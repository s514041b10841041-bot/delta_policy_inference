# model/internvl3/internvl3_embedder.py
import torch
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from transformers import GenerationConfig
from torchvision.transforms.functional import to_pil_image
from typing import Any, Dict, Optional, Union, List
from torch import nn
import logging
import functools

from .cross_view import CrossViewAdapter, CrossViewPredictiveBranch, TextGroundingAttention
from .bottleneck_fusion import BottleneckFusion

logger = logging.getLogger(__name__)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def flash_attn_is_available() -> bool:
    try:
        import flash_attn  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True

# === Image Transformations ===
def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

# === Aspect Ratio Handling ===
@functools.lru_cache(maxsize=10000)
def get_target_aspect_ratio(orig_width, orig_height, image_size, min_num, max_num):
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = orig_width * orig_height
    for ratio in target_ratios:
        target_ar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_ar)
        if diff < best_ratio_diff:
            best_ratio_diff = diff
            best_ratio = ratio
        elif diff == best_ratio_diff and area > 0.5 * image_size**2 * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


class InternVL3Embedder(nn.Module):
    def __init__(
        self,
        model_name="OpenGVLab/InternVL3-1B",
        image_size=448,
        device="cuda",
        enable_gradient_checkpointing: bool = False,
        gradient_checkpointing_use_reentrant: bool = False,
        use_cross_view_adapter: bool = False,
        use_cvp: bool = False,
        cross_view_num_heads: int = 14,
        cross_view_dropout: float = 0.0,
        cross_view_gamma_scale: float = 0.1,
        cross_view_gate_init: float = 0.2,
        cross_view_gate_per_channel: bool = True,
        cross_view_head_in_adapter: bool = True,
        cross_view_head_output_norm: bool = False,
        cross_view_orthogonal_residual: bool = False,
        cross_view_tp_index: int = 0,
        cross_view_wrist_index: int = 1,
        use_text_grounding: bool = False,
        cross_view_text_num_heads: Optional[int] = None,
        cvp_mask_ratio: float = 0.5,
        cvp_source_mode: str = "opposite_only",
        cvp_detach_source: bool = True,
        cvp_content_query_ratio: float = 0.0,
        use_bottleneck_fusion: bool = False,
        bottleneck_num_tokens: int = 16,
        bottleneck_num_heads: int = 14,
        bottleneck_mlp_ratio: int = 4,
        bottleneck_dropout: float = 0.0,
        bottleneck_view_partition: bool = True,
        use_action_query_adapter: bool = False,
        num_action_query_tokens: int = 32,
    ):
        super().__init__()
        # First statement in __init__ on purpose: this must fail before AutoModel loads a 1B
        # checkpoint, both so a mistyped launch fails in a second rather than two minutes and
        # so the unit test for it needs no network and no weights.
        if use_bottleneck_fusion:
            for name, enabled in (
                ("use_cross_view_adapter", use_cross_view_adapter),
                ("use_cvp", use_cvp),
                ("use_text_grounding", use_text_grounding),
            ):
                if enabled:
                    raise ValueError(
                        f"use_bottleneck_fusion=True is mutually exclusive with {name}=True: "
                        "they are two different fusion families occupying the same insertion "
                        "point, and enabling both measures neither. Pick one."
                    )
        self.device = device
        self.image_size = image_size
        self.max_text_length = 1024  # InternVL3 supports up to 1024 tokens
        self.enable_gradient_checkpointing = bool(enable_gradient_checkpointing)
        self.gradient_checkpointing_use_reentrant = bool(gradient_checkpointing_use_reentrant)
        self.transform = build_transform(image_size)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        use_flash_attn = flash_attn_is_available()
        if not use_flash_attn:
            logger.warning("flash_attn is not installed. Falling back to standard attention.")
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_flash_attn=use_flash_attn,
            low_cpu_mem_usage=True,
            _fast_init=False,
        ).to(self.device) 
        
        if hasattr(self.model.language_model, 'model'):
            layers = self.model.language_model.model.layers

        else:
            layers = self.model.language_model.layers
        layers = layers[:14]

        if hasattr(self.model.language_model, 'model'):
            self.model.language_model.model.layers = torch.nn.ModuleList(layers)
        else:
            self.model.language_model.layers = torch.nn.ModuleList(layers)
        self.model.language_model.lm_head = torch.nn.Identity()

        if use_cvp and not use_cross_view_adapter:
            raise ValueError("use_cvp=True requires use_cross_view_adapter=True")
        if use_text_grounding and not use_cross_view_adapter:
            raise ValueError("use_text_grounding=True requires use_cross_view_adapter=True")
        if use_text_grounding and use_cvp:
            raise ValueError(
                "use_text_grounding=True with use_cvp=True is not supported: CVP would "
                "train the shared tp_from_wrist/wrist_from_tp attention on raw (ungrounded) "
                "features while the main path feeds it grounded ones at inference -- the "
                "same train/inference query-distribution mismatch cross-view.md section 6.7 "
                "documents, just moved to the K/V side. Disable one."
            )
        if cross_view_tp_index == cross_view_wrist_index:
            raise ValueError("TP and Wrist must use different view indices")
        if min(cross_view_tp_index, cross_view_wrist_index) < 0:
            raise ValueError("Cross-view indices must be non-negative")

        self.use_cross_view_adapter = bool(use_cross_view_adapter)
        self.use_cvp = bool(use_cvp)
        self.use_text_grounding = bool(use_text_grounding)
        self.cross_view_tp_index = int(cross_view_tp_index)
        self.cross_view_wrist_index = int(cross_view_wrist_index)
        self.cross_view_adapter = None
        self.cvp_branch = None
        self.text_grounding = None
        if self.use_cross_view_adapter:
            projected_dim = int(self.model.config.hidden_size)
            model_dtype = next(self.model.parameters()).dtype
            self.cross_view_adapter = CrossViewAdapter(
                embed_dim=projected_dim,
                num_heads=cross_view_num_heads,
                dropout=cross_view_dropout,
                gamma_scale=cross_view_gamma_scale,
                gate_init=cross_view_gate_init,
                gate_per_channel=cross_view_gate_per_channel,
                use_head=cross_view_head_in_adapter,
                head_output_norm=cross_view_head_output_norm,
                orthogonal_residual=cross_view_orthogonal_residual,
            ).to(device=self.device, dtype=model_dtype)
            if self.use_cvp:
                # The current wrapper always uses one 448x448 tile per view.
                # If dynamic tiling is enabled later, CVP raises a clear error
                # until tile-aware positional embeddings are configured.
                self.cvp_branch = CrossViewPredictiveBranch(
                    embed_dim=projected_dim,
                    max_position_tokens=int(self.model.num_image_token),
                    mask_ratio=cvp_mask_ratio,
                    source_mode=cvp_source_mode,
                    detach_source=cvp_detach_source,
                    content_query_ratio=cvp_content_query_ratio,
                    owns_heads=not cross_view_head_in_adapter,
                ).to(device=self.device, dtype=model_dtype)
            if self.use_text_grounding:
                self.text_grounding = TextGroundingAttention(
                    embed_dim=projected_dim,
                    num_heads=cross_view_text_num_heads or cross_view_num_heads,
                    dropout=cross_view_dropout,
                ).to(device=self.device, dtype=model_dtype)

        # Attention-bottleneck fusion: the sparse control arm for the adapter above, never
        # both at once (guarded in the exclusivity check). Constructed only when enabled, so
        # a baseline run adds no parameters and draws no RNG -- which matters because the
        # action head is seeded from the RNG state left after the embedder is built
        # (EVO1.__init__), and a stray draw here would silently re-randomize it.
        self.use_bottleneck_fusion = bool(use_bottleneck_fusion)
        self.bottleneck_fusion = None
        if self.use_bottleneck_fusion:
            projected_dim = int(self.model.config.hidden_size)
            model_dtype = next(self.model.parameters()).dtype
            self.bottleneck_fusion = BottleneckFusion(
                embed_dim=projected_dim,
                num_tokens=bottleneck_num_tokens,
                num_heads=bottleneck_num_heads,
                mlp_ratio=bottleneck_mlp_ratio,
                dropout=bottleneck_dropout,
                view_partition=bottleneck_view_partition,
                gamma_scale=cross_view_gamma_scale,
                gate_init=cross_view_gate_init,
                gate_per_channel=cross_view_gate_per_channel,
            ).to(device=self.device, dtype=model_dtype)

        # Action-query adapter (see EvoConfig.use_action_query_adapter). Guarded behind the
        # flag so that off, this constructs nothing -- no parameters, no RNG draw -- and the
        # module is bit-for-bit identical to before this feature existed.
        self.use_action_query_adapter = bool(use_action_query_adapter)
        self.num_action_query_tokens = int(num_action_query_tokens)
        self.action_query_embed = None
        if self.use_action_query_adapter:
            model_dtype = next(self.model.parameters()).dtype
            self.action_query_embed = nn.Embedding(
                self.num_action_query_tokens, int(self.model.config.hidden_size)
            ).to(device=self.device, dtype=model_dtype)
            nn.init.zeros_(self.action_query_embed.weight)

        self._configure_memory_features()

    @staticmethod
    def _make_padding_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
        return (
            torch.arange(max_length, device=lengths.device).unsqueeze(0)
            >= lengths.unsqueeze(1)
        )

    def _apply_cross_view_modules(
        self,
        vit_embeds: torch.Tensor,
        batch_num_tiles_list: List[List[int]],
        image_masks: torch.Tensor,
        compute_cvp: bool,
        text_embeds: Optional[torch.Tensor] = None,
        text_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        """Fuse semantic view slots while preserving InternVL flatten order.

        ``text_embeds``/``text_padding_mask`` (``[B, L, D]`` / ``[B, L]``, one row per
        sample -- see ``_extract_text_only``) are only consumed when ``self.text_grounding``
        is set; every other configuration ignores them.
        """

        # getattr, not attribute access: several tests build this class through __new__
        # and set only the attributes their case needs, and an off-by-default module must
        # not force every one of them to be updated.
        bottleneck = getattr(self, "bottleneck_fusion", None)
        if self.cross_view_adapter is None and bottleneck is None:
            if compute_cvp:
                raise RuntimeError("CVP was requested but Cross-View Adapter is disabled")
            return vit_embeds, {}
        if compute_cvp and bottleneck is not None:
            raise RuntimeError("CVP is not supported with bottleneck fusion")
        if self.text_grounding is not None and text_embeds is None:
            raise RuntimeError("Text grounding is enabled but no text_embeds were provided")
        if vit_embeds.ndim != 3:
            raise ValueError(f"Expected projected features [tiles, tokens, dim], got {tuple(vit_embeds.shape)}")

        batch_size = len(batch_num_tiles_list)
        if image_masks.ndim != 2 or image_masks.shape[0] != batch_size:
            raise ValueError(
                f"image_masks must be [B, views] for B={batch_size}, got {tuple(image_masks.shape)}"
            )
        required_view_index = max(self.cross_view_tp_index, self.cross_view_wrist_index)
        if any(len(num_tiles) <= required_view_index for num_tiles in batch_num_tiles_list):
            raise ValueError(
                f"Cross-view fusion requires view slots {self.cross_view_tp_index} and "
                f"{self.cross_view_wrist_index} in every sample"
            )

        tokens_per_tile = vit_embeds.shape[1]
        embed_dim = vit_embeds.shape[2]
        tile_cursor = 0
        sample_views: List[List[torch.Tensor]] = []
        for num_tiles_list in batch_num_tiles_list:
            views = []
            for num_tiles_raw in num_tiles_list:
                num_tiles = int(num_tiles_raw)
                if num_tiles < 0:
                    raise ValueError(f"Tile count cannot be negative, got {num_tiles}")
                next_cursor = tile_cursor + num_tiles
                if next_cursor > vit_embeds.shape[0]:
                    raise ValueError("batch_num_tiles_list consumes more tiles than vit_embeds contains")
                views.append(vit_embeds[tile_cursor:next_cursor].reshape(-1, embed_dim))
                tile_cursor = next_cursor
            sample_views.append(views)
        if tile_cursor != vit_embeds.shape[0]:
            raise ValueError(
                f"batch_num_tiles_list consumed {tile_cursor} tiles, but vit_embeds has {vit_embeds.shape[0]}"
            )

        tp_list = [views[self.cross_view_tp_index] for views in sample_views]
        wrist_list = [views[self.cross_view_wrist_index] for views in sample_views]
        tp_lengths = torch.tensor(
            [features.shape[0] for features in tp_list], device=vit_embeds.device, dtype=torch.long
        )
        wrist_lengths = torch.tensor(
            [features.shape[0] for features in wrist_list], device=vit_embeds.device, dtype=torch.long
        )
        if int(tp_lengths.max().item()) == 0 or int(wrist_lengths.max().item()) == 0:
            if compute_cvp:
                raise RuntimeError(
                    "CVP was requested but this batch has zero valid TP/Wrist tokens"
                )
            return vit_embeds, {}

        x_tp = torch.nn.utils.rnn.pad_sequence(tp_list, batch_first=True)
        x_wrist = torch.nn.utils.rnn.pad_sequence(wrist_list, batch_first=True)
        tp_padding_mask = self._make_padding_mask(tp_lengths, x_tp.shape[1])
        wrist_padding_mask = self._make_padding_mask(wrist_lengths, x_wrist.shape[1])
        valid_mask = image_masks.to(device=vit_embeds.device, dtype=torch.bool)
        if valid_mask.shape[1] <= required_view_index:
            raise ValueError(
                f"image_masks only has {valid_mask.shape[1]} view slots; "
                f"cross-view fusion needs index {required_view_index}"
            )
        pair_valid = (
            valid_mask[:, self.cross_view_tp_index]
            & valid_mask[:, self.cross_view_wrist_index]
            & (tp_lengths > 0)
            & (wrist_lengths > 0)
        )

        aux: Dict[str, Any] = {}
        if compute_cvp:
            if self.cvp_branch is None:
                raise RuntimeError("compute_cvp=True but the CVP branch is disabled")
            cvp_result = self.cvp_branch(
                adapter=self.cross_view_adapter,
                x_tp=x_tp,
                x_wrist=x_wrist,
                tp_padding_mask=tp_padding_mask,
                wrist_padding_mask=wrist_padding_mask,
                pair_valid=pair_valid,
            )
            aux = {
                "cvp_loss": cvp_result.loss,
                "cvp_num_predictions": cvp_result.num_predictions,
                "cvp_num_tp_targets": cvp_result.num_tp_targets,
                "cvp_num_wrist_targets": cvp_result.num_wrist_targets,
            }

        attn_tp, attn_wrist = None, None
        if self.text_grounding is not None:
            # Grounding runs on every row regardless of pair_valid -- it only needs this
            # view's own tokens and the instruction, not the opposite view -- and the
            # padded/invalid rows it touches are exactly the ones CrossViewAdapter's own
            # pair_valid gating below discards, so no extra masking is needed here.
            attn_tp = self.text_grounding(x_tp, text_embeds, text_padding_mask)
            attn_wrist = self.text_grounding(x_wrist, text_embeds, text_padding_mask)

        # Same packing, same repacking, same pair_valid: the two fusion families differ
        # only in what happens between them.
        if bottleneck is not None:
            fused_tp, fused_wrist = bottleneck(
                x_tp=x_tp,
                x_wrist=x_wrist,
                tp_padding_mask=tp_padding_mask,
                wrist_padding_mask=wrist_padding_mask,
                pair_valid=pair_valid,
            )
        else:
            fused_tp, fused_wrist = self.cross_view_adapter(
                x_tp=x_tp,
                x_wrist=x_wrist,
                tp_padding_mask=tp_padding_mask,
                wrist_padding_mask=wrist_padding_mask,
                pair_valid=pair_valid,
                attn_tp=attn_tp,
                attn_wrist=attn_wrist,
            )
        for batch_index, views in enumerate(sample_views):
            tp_length = int(tp_lengths[batch_index].item())
            wrist_length = int(wrist_lengths[batch_index].item())
            views[self.cross_view_tp_index] = fused_tp[batch_index, :tp_length]
            views[self.cross_view_wrist_index] = fused_wrist[batch_index, :wrist_length]

        repacked_tiles = []
        for views, num_tiles_list in zip(sample_views, batch_num_tiles_list):
            for features, num_tiles_raw in zip(views, num_tiles_list):
                num_tiles = int(num_tiles_raw)
                repacked_tiles.append(features.reshape(num_tiles, tokens_per_tile, embed_dim))
        return torch.cat(repacked_tiles, dim=0), aux

    def _configure_memory_features(self) -> None:
        checkpoint_kwargs = {"use_reentrant": self.gradient_checkpointing_use_reentrant}
        
        # silencing the PyTorch warning and enforcing the correct config.
        import torch.utils.checkpoint
        if not hasattr(torch.utils.checkpoint, "_checkpoint_patched"):
            _orig_checkpoint = torch.utils.checkpoint.checkpoint
            def _checkpoint_patched(*args, **kwargs):
                if "use_reentrant" not in kwargs:
                    kwargs["use_reentrant"] = self.gradient_checkpointing_use_reentrant
                return _orig_checkpoint(*args, **kwargs)
            torch.utils.checkpoint.checkpoint = _checkpoint_patched
            torch.utils.checkpoint._checkpoint_patched = True

        def _enable_ckpt(module) -> bool:
            if module is None or not hasattr(module, "gradient_checkpointing_enable"):
                return False
            try:
                module.gradient_checkpointing_enable(gradient_checkpointing_kwargs=checkpoint_kwargs)
            except TypeError:
                module.gradient_checkpointing_enable()
            return True

        if not self.enable_gradient_checkpointing:
            if hasattr(self.model, "vision_model") and hasattr(self.model.vision_model, "encoder"):
                self.model.vision_model.encoder.gradient_checkpointing = False
            return

        enabled_any = False
        enabled_any = _enable_ckpt(self.model) or enabled_any

        if hasattr(self.model, "vision_model") and hasattr(self.model.vision_model, "encoder"):
            self.model.vision_model.encoder.gradient_checkpointing = True
            enabled_any = True

        if hasattr(self.model, "language_model"):
            language_model = self.model.language_model
            enabled_any = _enable_ckpt(language_model) or enabled_any
            if hasattr(language_model, "model"):
                enabled_any = _enable_ckpt(language_model.model) or enabled_any
            if hasattr(language_model, "config"):
                language_model.config.use_cache = False

        if hasattr(self.model, "config"):
            self.model.config.use_cache = False

        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

        if enabled_any:
            logger.info("Gradient checkpointing enabled for InternVL3 embedder.")
        else:
            logger.warning("Requested gradient checkpointing, but model does not expose checkpointing controls.")
        

    def _dynamic_preprocess_tensor(self, image_t, min_num=1, max_num=1, use_thumbnail=False):
        C, orig_height, orig_width = image_t.shape
        
        target_aspect_ratio = get_target_aspect_ratio(
            orig_width, orig_height, self.image_size, min_num, max_num)
            
        ratio_w, ratio_h = target_aspect_ratio[0], target_aspect_ratio[1]
        target_width = self.image_size * ratio_w
        target_height = self.image_size * ratio_h
        blocks = ratio_w * ratio_h
        
        resized_img = TF.resize(
            image_t, 
            size=[target_height, target_width], 
            interpolation=InterpolationMode.BICUBIC, 
            antialias=True
        )
        
        
        # 1. view to -> [C, ratio_h, image_size, ratio_w, image_size]
        reshaped = resized_img.view(C, ratio_h, self.image_size, ratio_w, self.image_size)
        # 2. permute to -> [ratio_h, ratio_w, C, image_size, image_size]
        permuted = reshaped.permute(1, 3, 0, 2, 4)
        # 3. reshape to -> [blocks, C, image_size, image_size]
        stacked_tiles = permuted.reshape(blocks, C, self.image_size, self.image_size)
            
        if use_thumbnail and blocks != 1:
            thumbnail_img = TF.resize(
                image_t, 
                size=[self.image_size, self.image_size], 
                interpolation=InterpolationMode.BICUBIC, 
                antialias=True
            )
            stacked_tiles = torch.cat([stacked_tiles, thumbnail_img.unsqueeze(0)], dim=0)
            
        return stacked_tiles

    def _preprocess_images(
        self,
        image_tensors_batch: List[List[Union[Image.Image, torch.Tensor]]]
    ) -> tuple[torch.Tensor, List[List[int]]]:

        pixel_values_list = []
        batch_num_tiles_list = []
        
        dtype = torch.bfloat16
        mean = torch.tensor(IMAGENET_MEAN, device=self.device, dtype=dtype).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=self.device, dtype=dtype).view(1, 3, 1, 1)

        for image_tensors in image_tensors_batch:
            num_tiles_list = []
            for image in image_tensors:
                if isinstance(image, torch.Tensor):
                    image_t = image.to(self.device, dtype=torch.float32)
                    if image_t.ndim == 3 and image_t.shape[1] == self.image_size and image_t.shape[2] == self.image_size:
                        tile_tensors = (image_t.unsqueeze(0).to(dtype=dtype) - mean) / std
                    else:
                        tiles = self._dynamic_preprocess_tensor(image_t).to(dtype=dtype)
                        tile_tensors = (tiles - mean) / std
                else:
                    image_t = TF.to_tensor(image).to(self.device, dtype=torch.float32)
                    tiles = self._dynamic_preprocess_tensor(image_t).to(dtype=dtype)
                    tile_tensors = (tiles - mean) / std
                
                pixel_values_list.append(tile_tensors)
                num_tiles_list.append(tile_tensors.shape[0])
            batch_num_tiles_list.append(num_tiles_list)

        if len(pixel_values_list) > 0:
            pixel_values = torch.cat(pixel_values_list, dim=0)
        else:
            pixel_values = torch.empty(0, 3, self.image_size, self.image_size, dtype=torch.bfloat16, device=self.device)

        return pixel_values, batch_num_tiles_list

    def _build_multimodal_prompt(
        self,
        batch_num_tiles_list: List[List[int]],
        text_prompts: List[str]
    ) -> List[str]:

        IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
        IMG_START_TOKEN = "<img>"
        IMG_END_TOKEN = "</img>"
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

        prompts = []
        for num_tiles_list, text_prompt in zip(batch_num_tiles_list, text_prompts):
            prompt_segments = []
            for i, tile_count in enumerate(num_tiles_list):
                token_count = self.model.num_image_token * tile_count
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * token_count + IMG_END_TOKEN
                prompt_segments.append(f"Image-{i+1}: {image_tokens}\n")
            
            final_prompt = "".join(prompt_segments) + text_prompt.strip()
            prompts.append(final_prompt)

        return prompts
    
    def _tokenize_and_embed_text(
        self,
        prompts: List[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize the multimodal prompt and look up its (pre-Qwen) embedding table rows.

        Split out of what used to be the front half of ``_prepare_and_fuse_embeddings`` so
        text embeddings exist before cross-view fusion runs, not only after -- see
        ``_extract_text_only``. At the <IMG_CONTEXT> positions this embedding is a
        placeholder, later overwritten by ``_splice_visual_into_embeds``; nothing here reads
        it at those positions.
        """

        untruncated_ids = self.tokenizer(prompts, padding=False, truncation=False)["input_ids"]
        true_sequence_length = max(len(ids) for ids in untruncated_ids) if len(untruncated_ids) > 0 else 0

        if true_sequence_length > self.max_text_length:
            print("\n" + "="*80)
            print(f" WARNING: Input prompt was TRUNCATED in batch!")
            print(f"   - Max Length Allowed    : {self.max_text_length}")
            print(f"   - Actual Max Length     : {true_sequence_length}")
            print("="*80 + "\n")

        # 'longest', not 'max_length': the fused sequence is returned to the action head WITHOUT
        # this attention_mask (see the return of the caller, and Evo1.py:97 -> flow_matching.py:284,
        # whose cross-attention takes no key_padding_mask). Padding is therefore invisible to the
        # LLM but fully visible to the action head, which spends softmax mass on it. Measured on
        # CALVIN at max_views=3: 1024 positions of which 226 were pure padding.
        # true_sequence_length above was already computed and only used for the warning.
        model_inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding='longest',
            truncation=True,
            max_length=self.max_text_length
        ).to(self.device)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        img_token_mask = (input_ids == self.img_context_token_id)
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids).clone()
        return input_ids, attention_mask, img_token_mask, input_embeds

    def _extract_text_only(
        self,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        img_token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The real instruction tokens' embeddings, repacked to ``[B, L, D]`` with padding.

        Same embedding-table rows Qwen will eventually see -- just the subset that is
        neither a padding position nor an <IMG_CONTEXT> placeholder -- re-padded per sample
        since removing the image placeholders leaves a different real length per row.
        """

        text_mask = attention_mask.bool() & ~img_token_mask
        lengths = text_mask.sum(dim=1)
        rows = [input_embeds[b][text_mask[b]] for b in range(input_embeds.shape[0])]
        text_embeds = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True)
        text_padding_mask = self._make_padding_mask(lengths, text_embeds.shape[1])
        return text_embeds, text_padding_mask

    def _splice_visual_into_embeds(
        self,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        img_token_mask: torch.Tensor,
        vit_embeds: torch.Tensor,
        image_masks: torch.Tensor,
        batch_num_tiles_list: List[List[int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, C = input_embeds.shape
        vit_embeds = vit_embeds.reshape(-1, C)

        tokens_per_tile = self.model.num_image_token
        vit_idx = 0

        actual_vis_tokens_list = img_token_mask.sum(dim=1).tolist()

        for b in range(B):
            expected_vis_tokens = sum(batch_num_tiles_list[b]) * tokens_per_tile
            mask_b = img_token_mask[b]
            actual_vis_tokens = actual_vis_tokens_list[b]
            
            item_vit_embeds = vit_embeds[vit_idx : vit_idx + expected_vis_tokens]
            vit_idx += expected_vis_tokens
            
            if actual_vis_tokens > 0:
                input_embeds[b, mask_b] = item_vit_embeds[:actual_vis_tokens]
            
            current_token_idx = 0
            img_token_locations = torch.where(mask_b)[0]
            
            for i in range(len(batch_num_tiles_list[b])):
                num_tiles_for_this_image = batch_num_tiles_list[b][i]
                num_tokens_for_this_image = num_tiles_for_this_image * tokens_per_tile
           
                if not image_masks[b][i]:
                    start_offset = current_token_idx
                    end_offset = min(current_token_idx + num_tokens_for_this_image, len(img_token_locations))
                    if start_offset < end_offset:
                        idxs = img_token_locations[start_offset:end_offset]
                        attention_mask[b, idxs] = 0
        
                current_token_idx += num_tokens_for_this_image
     
        return input_embeds, attention_mask

    def get_fused_image_text_embedding_from_tensor_images(
        self,
        image_tensors_batch: List[List[Union[Image.Image, torch.Tensor]]],
        image_masks: torch.Tensor,
        text_prompts: List[str],
        return_cls_only: bool = True,
        return_mask: bool = False,
        compute_cvp: bool = False,
        return_aux: bool = False,
    ):

        pixel_values, batch_num_tiles_list = self._preprocess_images(image_tensors_batch)
        aux: Dict[str, Any] = {}

        # Tokenize and embed text before cross-view fusion runs, not after: text_grounding
        # (when enabled) needs the instruction's embeddings as its K/V, and this is the only
        # point in the pipeline where they exist without depending on vit_embeds.
        prompts = self._build_multimodal_prompt(batch_num_tiles_list, text_prompts)
        input_ids, attention_mask, img_token_mask, inputs_embeds = self._tokenize_and_embed_text(prompts)
        # Stashed so a caller can map a position in the LLM sequence back to "this is the Nth
        # patch of view V" -- needed to draw the action head's cross-attention over the camera
        # frames. Overwritten every forward; nothing in the model reads it.
        self._last_img_token_mask = img_token_mask.detach()

        text_embeds, text_padding_mask = None, None
        if self.text_grounding is not None:
            text_embeds, text_padding_mask = self._extract_text_only(
                inputs_embeds, attention_mask, img_token_mask
            )

        if pixel_values.shape[0] == 0:
            print("Warning: No valid images to process after masking.")
            vit_embeds = torch.empty(0, self.model.config.hidden_size).to(self.device, dtype=torch.bfloat16)
        else:
            vit_embeds = self.model.extract_feature(pixel_values)
            vit_embeds, aux = self._apply_cross_view_modules(
                vit_embeds=vit_embeds,
                batch_num_tiles_list=batch_num_tiles_list,
                image_masks=image_masks,
                compute_cvp=compute_cvp,
                text_embeds=text_embeds,
                text_padding_mask=text_padding_mask,
            )

        inputs_embeds, attention_mask = self._splice_visual_into_embeds(
            inputs_embeds, attention_mask, img_token_mask, vit_embeds, image_masks, batch_num_tiles_list
        )

        # Action-query adapter: append the zero-init query tokens to InternVL3's OWN input
        # sequence so they walk through the same causal self-attention as the vision/text
        # tokens and pick up a grounded summary, then split them back off below. S is the
        # pre-append length -- everything downstream that isn't the adapter (return_cls_only,
        # the return_aux/CVP path) only ever sees the first S positions, unchanged.
        S = inputs_embeds.shape[1]
        llm_inputs_embeds, llm_attention_mask = inputs_embeds, attention_mask
        if self.use_action_query_adapter:
            B = inputs_embeds.shape[0]
            query_embeds = self.action_query_embed.weight.unsqueeze(0).expand(B, -1, -1)
            query_embeds = query_embeds.to(dtype=inputs_embeds.dtype)
            llm_inputs_embeds = torch.cat([inputs_embeds, query_embeds], dim=1)
            query_attn = attention_mask.new_ones((B, self.num_action_query_tokens))
            llm_attention_mask = torch.cat([attention_mask, query_attn], dim=1)

        outputs = self.model.language_model(
            inputs_embeds=llm_inputs_embeds,
            attention_mask=llm_attention_mask,
            use_cache=False,
            return_dict=True,
        )
        fused_hidden = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        action_query_hidden = None
        if self.use_action_query_adapter:
            fused_hidden, action_query_hidden = fused_hidden[:, :S, :], fused_hidden[:, S:, :]

        if return_cls_only:
            cls_hidden = fused_hidden[:, 0, :]
            if return_aux:
                return (cls_hidden, None, aux) if return_mask else (cls_hidden, aux)
            return (cls_hidden, None) if return_mask else cls_hidden
        # attention_mask marks which of these positions are real. The LLM already honoured it,
        # but the action head cannot: it receives only the hidden states, and its
        # cross-attention would otherwise spend softmax mass on <pad> and on any view whose
        # image_mask is 0. Returning it is what lets flow_matching.py mask those keys out.
        if return_aux:
            return (fused_hidden, attention_mask, aux) if return_mask else (fused_hidden, aux)
        if self.use_action_query_adapter:
            # img_token_mask (S positions, same as fused_hidden/attention_mask) tells the
            # action head's gated branch which of those positions are actual image patches --
            # see FlowmatchingActionHead.forward's use of action_query_img_mask.
            return (fused_hidden, attention_mask, action_query_hidden, img_token_mask) if return_mask \
                else (fused_hidden, action_query_hidden, img_token_mask)
        return (fused_hidden, attention_mask) if return_mask else fused_hidden
