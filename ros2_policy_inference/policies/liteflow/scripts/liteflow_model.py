import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import List, Union, Tuple
from PIL import Image
import torch
import torch.nn as nn
from model.internvl3.internvl3_embedder import InternVL3Embedder
from model.action_head.flow_matching import FlowmatchingActionHead
from config import EvoConfig
import logging

class EVO1(nn.Module):
    def __init__(self, config: EvoConfig):
        super().__init__() 
        self.config = config
        self._device = config.device
        self.return_cls_only = config.return_cls_only
        vlm_name = config.vlm_name
        self.embedder = InternVL3Embedder(
            model_name=vlm_name,
            device=self._device,
            enable_gradient_checkpointing=config.enable_gradient_checkpointing,
            gradient_checkpointing_use_reentrant=config.gradient_checkpointing_use_reentrant,
            use_cross_view_adapter=config.use_cross_view_adapter,
            use_cvp=config.use_cvp,
            cross_view_num_heads=config.cross_view_num_heads,
            cross_view_dropout=config.cross_view_dropout,
            cross_view_gamma_scale=config.cross_view_gamma_scale,
            cross_view_gate_init=config.cross_view_gate_init,
            cross_view_gate_per_channel=config.cross_view_gate_per_channel,
            cross_view_head_in_adapter=config.cross_view_head_in_adapter,
            cross_view_head_output_norm=config.cross_view_head_output_norm,
            cross_view_orthogonal_residual=config.cross_view_orthogonal_residual,
            cross_view_tp_index=config.cross_view_tp_index,
            cross_view_wrist_index=config.cross_view_wrist_index,
            use_text_grounding=config.use_text_grounding,
            cross_view_text_num_heads=config.cross_view_text_num_heads,
            cvp_mask_ratio=config.cvp_mask_ratio,
            cvp_source_mode=config.cvp_source_mode,
            cvp_detach_source=config.cvp_detach_source,
            cvp_content_query_ratio=config.cvp_content_query_ratio,
            use_bottleneck_fusion=config.use_bottleneck_fusion,
            bottleneck_num_tokens=config.bottleneck_num_tokens,
            bottleneck_num_heads=config.bottleneck_num_heads,
            bottleneck_mlp_ratio=config.bottleneck_mlp_ratio,
            bottleneck_dropout=config.bottleneck_dropout,
            bottleneck_view_partition=config.bottleneck_view_partition,
            use_action_query_adapter=config.use_action_query_adapter,
            num_action_query_tokens=config.num_action_query_tokens,
        )
        self._configure_language_lora()

        # Re-seed before the action head so its initialization does not depend on how many
        # random numbers the embedder happened to consume.
        #
        # Measured, not hypothetical: building CrossViewAdapter + CrossViewPredictiveBranch
        # consumes RNG, and they are constructed above -- so merely enabling cross-view gave
        # the action head a completely different init. Comparing the first cross-view run
        # against its baseline, the VLM weights agreed to within 1% (median relative
        # difference 0.0101) while the action head's differed by 1.4127 -- sqrt(2), the exact
        # signature of two statistically independent random vectors. A synthetic repro with
        # the same seed and cross-view on/off gives 1.4126. That confound, not the module,
        # is the leading explanation for that run's -34.8 points on libero_spatial_occluded:
        # both models fit the training distribution equally well (action loss within 2.3%),
        # but occlusion is OOD, where two unrelated solutions diverge freely, and forcing
        # gamma to exactly 0 at inference did not recover the loss.
        if config.seed is not None:
            torch.manual_seed(config.seed + 1)

        action_head_type = config.action_head.lower()
        
        if action_head_type == "flowmatching":
           
            horizon = config.action_horizon if config.action_horizon is not None else config.horizon
            per_action_dim = config.per_action_dim
            action_dim = config.action_dim
            
            if action_dim != horizon * per_action_dim:
                raise ValueError(f"action_dim ({action_dim}) ≠ horizon ({horizon}) × per_action_dim ({per_action_dim})")
            
            self.horizon = horizon
            self.per_action_dim = per_action_dim
            
            self.action_head = FlowmatchingActionHead(config=config).to(self._device)
        else:
            raise NotImplementedError(f"Unknown action_head: {action_head_type}")

    def get_vl_embeddings(
        self,
        images: List[List[Union[Image.Image, torch.Tensor]]],
        image_mask: torch.Tensor,  
        prompt: List[str] = None,
        return_cls_only: Union[bool, None] = None,
        return_mask: bool = False,
        compute_cvp: bool = False,
        return_aux: bool = False,
    ) -> torch.Tensor:

        if return_cls_only is None:
            return_cls_only = self.return_cls_only

        if images is None or len(images) == 0:
            raise ValueError("Must provide at least one batch of images. Got `images=None` or empty list.")
        
        if prompt is None:
            prompt = [""] * len(images)
            
        return self.embedder.get_fused_image_text_embedding_from_tensor_images(
            image_tensors_batch=images,
            image_masks=image_mask,
            text_prompts=prompt,
            return_cls_only=return_cls_only,
            return_mask=return_mask,
            compute_cvp=compute_cvp,
            return_aux=return_aux,
        )

    def prepare_state(self, state_input: Union[list, torch.Tensor]) -> torch.Tensor:

        if isinstance(state_input, list):
            state_tensor = torch.tensor(state_input)
        elif isinstance(state_input, torch.Tensor):
            state_tensor = state_input
        else:
            raise TypeError("Unsupported state input type")

        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)

        return state_tensor.to(self._device)

    
    def predict_action(
        self,
        fused_tokens: torch.Tensor,
        state: torch.Tensor,
        actions_gt: torch.Tensor = None,
        action_mask: torch.Tensor = None,
        embodiment_ids: torch.Tensor = None,
        context_mask: torch.Tensor = None,
        action_query_tokens: torch.Tensor = None,
        action_query_img_mask: torch.Tensor = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        if actions_gt is None:
            return self.action_head.get_action(fused_tokens, state=state, action_mask=action_mask, embodiment_id=embodiment_ids, context_mask=context_mask,
                                               action_query_tokens=action_query_tokens, action_query_img_mask=action_query_img_mask)
        else:
            return self.action_head(fused_tokens, state=state, actions_gt=actions_gt, action_mask=action_mask, embodiment_id=embodiment_ids, context_mask=context_mask,
                                    action_query_tokens=action_query_tokens, action_query_img_mask=action_query_img_mask)


    @torch.no_grad()
    def run_inference(
        self,
        images: List[Union[Image.Image, torch.Tensor]],
        image_mask: torch.Tensor,
        prompt: str,
        state_input: Union[list, torch.Tensor],
        return_cls_only: Union[bool, None] = None,
        action_mask: Union[torch.Tensor, None] = None
    ) -> torch.Tensor:

        # Wrap inputs into a batch of size 1 for the embedder
        if image_mask.dim() == 1:
            image_mask = image_mask.unsqueeze(0)
            
        # return_mask=True keeps inference consistent with training: the action head's
        # cross-attention must ignore <pad> and masked-out image slots in both, otherwise
        # the two see structurally different context.
        vl_out = self.get_vl_embeddings(
                        images=[images],
                        image_mask=image_mask,
                        prompt=[prompt],
                        return_cls_only=return_cls_only,
                        return_mask=True,
                    )
        if self.embedder.use_action_query_adapter:
            fused_tokens, context_mask, action_query_tokens, action_query_img_mask = vl_out
        else:
            fused_tokens, context_mask = vl_out
            action_query_tokens, action_query_img_mask = None, None

        state_tensor = self.prepare_state(state_input)

        action = self.predict_action(fused_tokens, state_tensor, action_mask=action_mask,
                                     context_mask=context_mask, action_query_tokens=action_query_tokens,
                                     action_query_img_mask=action_query_img_mask)

        if isinstance(action, torch.Tensor) and action.dtype == torch.bfloat16:
            action = action.to(torch.float32)
        
        return action
    

    def forward(self, fused_tokens, state=None, actions_gt=None, action_mask=None, embodiment_ids=None,
                context_mask=None, action_query_tokens=None, action_query_img_mask=None):

        return self.predict_action(fused_tokens, state, actions_gt, action_mask, embodiment_ids,
                                   context_mask=context_mask, action_query_tokens=action_query_tokens,
                                   action_query_img_mask=action_query_img_mask)

    def remove_training_only_modules(self):
        """Drop CVP parameters after loading a checkpoint for inference."""

        if getattr(self.embedder, "cvp_branch", None) is not None:
            self.embedder.cvp_branch = None
        self.embedder.use_cvp = False

    def _freeze_module(self, module: nn.Module, name: str):
        print(f"Freezing {name} parameters...")
        for p in module.parameters():
            p.requires_grad = False

    def _set_module_trainable(self, module: nn.Module, trainable: bool, name: str):
        action = "Finetuning" if trainable else "Freezing"
        print(f"{action} {name} parameters...")
        for p in module.parameters():
            p.requires_grad = trainable

    def _parse_lora_target_modules(self) -> List[str]:
        target_modules = self.config.lora_target_modules
        if isinstance(target_modules, str):
            target_modules = [name.strip() for name in target_modules.split(",")]
        else:
            target_modules = [str(name).strip() for name in target_modules]
        target_modules = [name for name in target_modules if name]
        if not target_modules:
            raise ValueError("lora_target_modules must contain at least one module name.")
        return target_modules

    def _configure_language_lora(self):
        """Wrap only the InternVL language branch with PEFT LoRA."""
        if not self.config.use_lora:
            return

        if self.config.lora_rank <= 0:
            raise ValueError(f"lora_rank must be positive, got {self.config.lora_rank}.")
        if self.config.lora_alpha <= 0:
            raise ValueError(f"lora_alpha must be positive, got {self.config.lora_alpha}.")
        if not 0.0 <= self.config.lora_dropout < 1.0:
            raise ValueError(f"lora_dropout must be in [0, 1), got {self.config.lora_dropout}.")

        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise ImportError(
                "LoRA training requires PEFT. Install Evo_1/requirements.txt "
                "or run `pip install peft==0.10.0`."
            ) from exc

        target_modules = self._parse_lora_target_modules()
        peft_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.embedder.model.language_model = get_peft_model(
            self.embedder.model.language_model,
            peft_config,
        )
        logging.info(
            "Enabled language-model LoRA: rank=%d, alpha=%d, dropout=%s, targets=%s",
            self.config.lora_rank,
            self.config.lora_alpha,
            self.config.lora_dropout,
            ",".join(target_modules),
        )

    def _enable_only_lora_parameters(self):
        language_model = self.embedder.model.language_model
        trainable_lora_params = 0
        for name, param in language_model.named_parameters():
            is_lora_parameter = "lora_" in name
            param.requires_grad = is_lora_parameter
            if is_lora_parameter:
                trainable_lora_params += param.numel()

        if trainable_lora_params == 0:
            raise RuntimeError(
                "use_lora=True, but no LoRA parameters were found in the language model. "
                "Check lora_target_modules against the InternVL language-model module names."
            )
        print(f"Finetuning language LoRA parameters ({trainable_lora_params / 1e6:.2f}M)...")

    def set_finetune_flags(self):
        config = self.config  
        legacy_finetune_vlm = config.finetune_vlm
        finetune_language_model = config.finetune_language_model
        finetune_vision_model = config.finetune_vision_model

        if config.use_lora:
            if legacy_finetune_vlm or finetune_language_model:
                raise ValueError(
                    "--use_lora cannot be combined with --finetune_vlm or "
                    "--finetune_language_model. LoRA already defines how the language branch is trained."
                )

            self._set_module_trainable(self.embedder, False, "VLM (InternVL3)")
            self._set_module_trainable(
                self.embedder.model.vision_model,
                finetune_vision_model,
                "VLM vision_model",
            )
            if hasattr(self.embedder.model, "mlp1"):
                self._set_module_trainable(
                    self.embedder.model.mlp1,
                    finetune_vision_model,
                    "VLM visual projector (mlp1)",
                )
            self._enable_only_lora_parameters()
        elif finetune_language_model or finetune_vision_model:
            self._set_module_trainable(self.embedder, False, "VLM (InternVL3)")
            self._set_module_trainable(self.embedder.model.language_model, finetune_language_model, "VLM language_model")
            self._set_module_trainable(self.embedder.model.vision_model, finetune_vision_model, "VLM vision_model")
            if hasattr(self.embedder.model, "mlp1"):
                self._set_module_trainable(self.embedder.model.mlp1, finetune_vision_model, "VLM visual projector (mlp1)")
        else:
            if not legacy_finetune_vlm:
                self._freeze_module(self.embedder, "VLM (InternVL3)")
            else:
                print("Finetuning VLM (InternVL3)...")

        if not config.finetune_action_head:
            self._freeze_module(self.action_head, "Action Head")
        else:
            print("Finetuning Action Head...")

        # Baseline flags above intentionally keep their previous meaning.  New
        # cross-view modules are always trainable when explicitly enabled, even
        # when the baseline freezes the rest of the embedder as one module.
        cross_view_adapter = getattr(self.embedder, "cross_view_adapter", None)
        if cross_view_adapter is not None:
            self._set_module_trainable(cross_view_adapter, True, "Cross-View Adapter")
        cvp_branch = getattr(self.embedder, "cvp_branch", None)
        if cvp_branch is not None:
            self._set_module_trainable(cvp_branch, True, "CVP branch")
        text_grounding = getattr(self.embedder, "text_grounding", None)
        if text_grounding is not None:
            self._set_module_trainable(text_grounding, True, "Text-Grounding Attention")
        bottleneck_fusion = getattr(self.embedder, "bottleneck_fusion", None)
        if bottleneck_fusion is not None:
            self._set_module_trainable(bottleneck_fusion, True, "Bottleneck Fusion")
