import dataclasses
from dataclasses import dataclass, field, asdict
from typing import Optional

@dataclass
class EvoConfig:
    # Basic config
    device: str = "cuda"
    run_name: str = "default_run"
    vlm_name: str = "OpenGVLab/InternVL3-1B"
    action_head: str = "flowmatching"
    return_cls_only: bool = False
    disable_wandb: bool = False
    wandb_project: str = "default_run"
    debug: bool = False

    # Dataset
    dataset_type: str = "lerobot"
    data_paths: Optional[str] = None
    dataset_config_path: str = ""
    image_size: int = 448
    binarize_gripper: bool = False
    use_augmentation: bool = False
    # Per-sample probability of erasing a random patch of the TP (slot-0) view during
    # training, via torchvision's RandomErasing. Independent of use_augmentation/
    # aug_transform: this is training-time exposure to "TP present but its content is
    # locally wrong" -- the LIBERO-Occ-style failure mode -- not the geometric/color jitter
    # aug_transform already does. 0.0 never constructs the transform, so RNG stream and
    # behaviour are unchanged when off. Dataset-loading-only: does not affect what the
    # server reconstructs from a checkpoint, so no EvoConfig.from_dict generation shim
    # is needed (absent field just resolves to the dataclass default, 0.0).
    tp_occlusion_prob: float = 0.0
    vision_masked: bool = False
    max_samples_per_file: Optional[int] = None
    horizon: int = 16
    video_backend: str = "av"
    cache_dir: Optional[str] = None
    # Resolved from the dataset config at startup (see train.prepare_dataset), never set on
    # the command line -- the dataset config is the single source of truth. Persisted into
    # the checkpoint's config.json so Evo1_server denormalizes actions the same way training
    # normalized them. None means "checkpoint predates this field"; the server then warns
    # and assumes "bounds".
    normalization_type: Optional[str] = None

    # Training
    lr: float = 1e-5
    batch_size: int = 8
    num_workers: int = 8
    prefetch_factor: Optional[int] = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    max_steps: int = 600
    warmup_steps: int = 300
    final_lr_scale: float = 0.0
    grad_clip_norm: float = 1.0
    weight_decay: float = 1e-5
    fused_adamw: bool = True
    # AdamW betas. The default is torch's (0.9, 0.999); FLOWER runs (0.9, 0.95)
    # (conf/model/flower.yaml optimizer.betas). 0.999 gives the second moment a ~1000-step
    # memory, which is most of a 20k-step budget.
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999

    # --- EMA of the weights, ported from FLOWER's flower/callbacks/ema.py ---
    # FLOWER evaluates the EMA copy, not the trained one (`evaluate_ema_weights_instead:
    # True`), so its published CALVIN score is an EMA number. Off by default: it costs an
    # fp32 shadow of every parameter and no existing checkpoint here was trained with it.
    use_ema: bool = False
    # The yaml's `decay: 0.999` is NOT what FLOWER's runs used -- that value is only read on
    # the apex multi_tensor path, and apex is not installed in the env those runs used, so
    # apply_ema()'s get_decay() schedule is what produced every published number:
    #   decay(step) = clamp(1 - (1 + step/inv_gamma) ** -power, min_decay, max_decay)
    # It ramps the averaging window from ~22 steps at step 100 to ~1170 at step 40k.
    ema_power: float = 2.0 / 3.0
    ema_inv_gamma: float = 1.0
    ema_min_decay: float = 0.0
    ema_max_decay: float = 0.9999
    ema_start_step: int = 0
    non_finite_max_streak: int = 5
    log_interval: int = 10
    ckpt_interval: int = 10
    save_dir: str = "./checkpoints"
    enable_gradient_checkpointing: bool = True
    gradient_checkpointing_use_reentrant: bool = False

    # Resume
    resume: bool = False
    resume_path: Optional[str] = None
    # Explicit step counter for a resume whose checkpoint carries a non-numeric step
    # (step_best / step_final saved before 2026-09-05 wrote "best" / "final" into client_state).
    resume_step: Optional[int] = None
    resume_pretrain: bool = False

    # Finetuning
    finetune_vlm: bool = False
    finetune_language_model: bool = False
    finetune_vision_model: bool = False
    finetune_action_head: bool = False
    use_lora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    lora_lr: Optional[float] = None

    # Cross-view fusion. Both features are opt-in so checkpoints and training
    # commands written before this branch retain their exact architecture.
    use_cross_view_adapter: bool = False
    use_cvp: bool = False
    cross_view_num_heads: int = 14
    cross_view_dropout: float = 0.0
    cross_view_gamma_scale: float = 0.1
    # Gate initialization, in alpha (pre-tanh) units: gamma = gamma_scale * tanh(alpha).
    # 0.0 was the original zero-init, and it left the gate dead -- the first run ended at
    # alpha ~ 0.004, exactly the sqrt(steps)*lr displacement of a random walk, because
    # gamma ~ 0 scales the action gradient reaching the attention to ~0. 0.2 starts gamma at
    # 0.1*tanh(0.2) = 0.0197, ~2% of the residual: enough for the attention to receive real
    # action gradient from step 0, small enough not to disturb the pretrained features.
    cross_view_gate_init: float = 0.2
    # One gate per channel instead of one scalar per direction. A single scalar has to be
    # right for all embed_dim channels at once; per-channel lets the useful ones open while
    # the harmful ones stay shut, and gives the gate embed_dim independent gradients instead
    # of one. Costs 2*embed_dim parameters.
    cross_view_gate_per_channel: bool = True
    # Learning rate for the gate parameters only. Under AdamW a parameter moves about `lr`
    # per step, so at the base 2e-5 the gate needs ~50k steps to travel alpha 0->1 even with
    # a perfectly consistent gradient -- longer than the whole 40k budget. 1e-3 lets it reach
    # a decided value within the first few percent of training. None = use the base lr.
    cross_view_gate_lr: Optional[float] = 1e-3
    # CVP trains head(raw_delta) ~ target via cosine loss, which -- LayerNorm at head's
    # input strips raw_delta's scale, the two Linears can rotate it freely -- constrains
    # nothing about raw_delta itself. The first run's main path added raw_delta directly,
    # unheaded, so CVP could satisfy its own loss while raw_delta pointed somewhere the
    # residual add never benefited from. True moves the head onto CrossViewAdapter and
    # makes the main path add head(raw_delta) too, so both consumers optimize the same
    # quantity. False reproduces the original mismatch for an A/B against this fix.
    cross_view_head_in_adapter: bool = True
    # The head's front LayerNorm only bounds its input; the final Linear's output -- what
    # gamma actually scales and adds -- has no bound at all, so on content the shared
    # attention never trained on (an occluded view) the injected perturbation's magnitude
    # is whatever that Linear produces, not "small because gamma is small". True appends a
    # trailing LayerNorm to the head so gamma always scales a fixed-norm quantity: a hard
    # cap on the residual instead of a tendency measured only on the training distribution.
    # False reproduces every checkpoint so far (no such weight exists on their head).
    cross_view_head_output_norm: bool = False
    # Project the gated residual onto the orthogonal complement of the token before adding
    # it. CVP's target is the token itself, so its pull drives the residual toward a rescale
    # of x (redundant -- x already carries that); the projection keeps only the component
    # x does not have. Guarantees: <fused,x>/<x,x> == 1 exactly (the original feature is
    # untouched along its own direction) and the added vector is 100% new directions -- the
    # design goal "each feature gains a little new information without disturbing what it
    # already encodes" made literal. Removes the residual_cos->1 degeneration structurally,
    # at the stated cost that the adapter cannot suppress x along itself. False reproduces
    # every checkpoint so far; adds no parameters, so no from_dict generation shim.
    cross_view_orthogonal_residual: bool = False
    cross_view_tp_index: int = 0
    cross_view_wrist_index: int = 1
    # Ground each view against the instruction before cross-view fusion: x_tp/x_wrist each
    # independently attend to the instruction tokens (one shared TextGroundingAttention
    # instance across both views), and the *grounded* result is what the bidirectional
    # cross-view attention reads as Q/K -- the gated residual still adds onto the untouched
    # original x_tp/x_wrist (CrossViewAdapter's attn_tp/attn_wrist parameters), so language
    # never bypasses the existing gate. Requires use_cross_view_adapter=True. Independent of
    # use_cvp: CVP's reconstruction target is language-independent by design (cross-view.md
    # 6.10), so this and CVP compose freely. False adds no parameters and changes no
    # existing forward pass, so no from_dict generation shim is needed.
    use_text_grounding: bool = False
    # Heads for the text-grounding attention. None reuses cross_view_num_heads.
    cross_view_text_num_heads: Optional[int] = None
    cvp_mask_ratio: float = 0.5
    cvp_loss_weight: float = 0.1
    cvp_source_mode: str = "opposite_only"
    # Detaching CVP K/V keeps Visual Encoder/Projector training identical to
    # the baseline; they still receive action-loss gradients when enabled.
    cvp_detach_source: bool = True
    # Fraction of CVP's sampled positions whose query is the token's own detached clean
    # feature (bare -- no mask/position/view embedding) instead of a mask query. Mask
    # queries never train the shared attention's q_norm/q_proj on the content inputs the
    # main path actually feeds them; that side otherwise only sees the action gradient
    # through the gate, the weak channel the dead-gate run exposed. Content queries make
    # CVP train head(attend(x_i, opposite)) ~ x_i, the main path's literal inference
    # computation. 0.0 = mask-query-only (the behaviour of every checkpoint so far; adds
    # no parameters, so no from_dict generation shim is needed).
    cvp_content_query_ratio: float = 0.0

    # Attention-bottleneck fusion (Nagrani et al. 2021): the sparse counterpart to the
    # Cross-View Adapter, and the second family in the "does explicit two-view fusion buy
    # anything at 700M" comparison. Same insertion point (projected features, before the
    # <IMG_CONTEXT> splice) and same gated residual as the adapter; the difference is that
    # the two views never attend to each other -- everything crosses through K learned
    # bottleneck tokens. Mutually exclusive with use_cross_view_adapter / use_cvp /
    # use_text_grounding (InternVL3Embedder.__init__ raises), because a control arm that
    # runs two fusion families at once measures neither.
    #
    # Gate behaviour deliberately reuses the adapter's fields rather than duplicating them
    # (cross_view_gamma_scale / cross_view_gate_init / cross_view_gate_per_channel /
    # cross_view_gate_lr), so both arms are gated identically by construction.
    #
    # Off by default and it constructs nothing when off, so no from_dict generation shim is
    # needed: a checkpoint written before this field resolves it to False and rebuilds
    # exactly the module tree it stored.
    use_bottleneck_fusion: bool = False
    # K. 16 tokens against 2x256 patches is the bottleneck; the whole point is that it is
    # far too narrow to pass the views through, so only what is worth sharing survives.
    bottleneck_num_tokens: int = 16
    # 896 / 14 = 64-dim heads, the same as the adapter's.
    bottleneck_num_heads: int = 14
    bottleneck_mlp_ratio: int = 4
    bottleneck_dropout: float = 0.0
    # Step-1 attention mask: the first K//2 bottleneck tokens read only TP, the rest only
    # Wrist (all still see the z keys). No parameters, so the state_dict is unchanged, but
    # the forward is not: a bottleneck checkpoint written before this field was trained
    # WITHOUT the mask, and from_dict resolves an absent field to False for that reason.
    # Why on by default: the two cameras are asymmetric (fixed vs wrist-mounted), so an
    # unconstrained z has a cheap degenerate solution -- read the stable camera only -- and
    # that would leave the wrist->TP direction dead, which is exactly the direction the
    # occluded evaluation needs (cross-view.md 13.1).
    bottleneck_view_partition: bool = True

    # Action-query adapter (VLA-Adapter-style ablation). InternVL3Embedder appends
    # `num_action_query_tokens` extra learnable, zero-initialized tokens to its own input
    # sequence (after the vision+text tokens), so they pass through the same 14 layers of
    # causal self-attention as everything else and arrive at the action head already
    # carrying a language+vision-grounded summary -- unlike the plain vision/text hidden
    # states, which the LLM never specifically curated for the action head. The action
    # head then cross-attends into two sources per block instead of one: this summary
    # (+ the state token) is always attended to, and the VLM's own raw image-patch hidden
    # states are attended to through a per-block gate that starts closed
    # (action_query_raw_gate_init) and may open during training. Off reproduces the exact
    # current architecture bit-for-bit: the embedding is never constructed, so it adds no
    # parameters, consumes no RNG, and no from_dict generation shim is needed.
    use_action_query_adapter: bool = False
    # K: size of the query-token summary buffer. Independent of horizon/action_dim -- this
    # is a fixed-size scratchpad for the LLM to fill, not a per-timestep slot.
    num_action_query_tokens: int = 32
    # Pre-tanh gate on the raw image-patch cross-attention branch, applied to its attention
    # logits before they join the always-on branch's logits in one softmax (not a
    # post-hoc residual scale -- see FlowerGatedCrossAttention). 0.0 = tanh(0) = 0: at
    # step 0 the raw branch's logits are flattened to a constant, so softmax gives it only
    # a small, content-independent share of attention and the block's output is
    # dominated by the always-on branch, exactly like cross_view_gate_init's role for the
    # cross-view adapter.
    action_query_raw_gate_init: float = 0.0

    # Global RNG seed. None reproduces the old behaviour: nothing seeded anything, so every
    # run started from a different random action head and no run was reproducible. Worse, any
    # config change that alters how much RNG module construction consumes -- enabling the
    # cross-view adapter, for one -- silently re-randomizes everything built afterwards, so an
    # A/B against a baseline was never a controlled comparison. Set this on both arms of any
    # comparison; EVO1.__init__ additionally re-seeds before the action head so its init is
    # identical whether or not cross-view is enabled.
    seed: Optional[int] = None

    # Misc / Model params
    per_action_dim: int = 7
    state_dim: int = 7
    action_horizon: Optional[int] = None
    num_layers: int = 8
    dropout: float = 0.0
    # Self-attention over the action chunk inside each action-head block. Off reproduces the
    # original architecture, where the horizon steps never attend to each other (see
    # flow_matching.BasicTransformerBlock). Checkpoints written before 2026-08-26 have no
    # such weights, so loading one requires action_self_attn=False.
    action_self_attn: bool = True
    # RMSNorm + square Linear on the VLM features before the action head cross-attends them,
    # mirroring FLOWER's cond_norm/cond_linear. Off reproduces the raw-hidden-state path.
    action_cond_proj: bool = True
    # "flower" = the ported FlowBlock (adaLN-modulated gated self-attn / cross-attn / SwiGlu,
    # RMSNorm, RoPE on the action chunk). "basic" = the original block, kept so the two can be
    # compared. Checkpoints are not interchangeable between them.
    action_block: str = "flower"
    action_adaln_dim: int = 256      # FLOWER's lora_dim (transformers.py:522)
    # FLOWER's parameter-saving scheme for adaLN: one shared full-rank controller
    # (self.global_adaln, dim->6*dim) computed once per batch and added to every block's
    # own low-rank adapter (action_adaln_dim: dim->lora_dim->6*dim) -- far fewer parameters
    # than giving every layer its own full-rank adaLN. False ablates the whole scheme: no
    # shared controller, and each block's adaLN_modulation becomes an ordinary independent
    # dim->6*dim projection (action_adaln_dim is then unused).
    action_global_adaln: bool = True
    # conf/model/flower.yaml says 32.0, but every FLOWER run this model is measured against
    # overrides it: logs/runs/2026-08-16/calvin_ibadapter_*/.hydra/overrides.yaml all carry
    # model.rope_theta=1000.0. The yaml is not the last word -- check the run's overrides.
    action_rope_theta: float = 1000.0
    # FLOWER's use_adaln_cond (conf/model/flower.yaml) is false, so its global_cond is the
    # timestep embedding alone. True adds the mask-weighted mean of the VLM context to it,
    # which is flower.py:1133-1141's other branch and what this head used to do unconditionally.
    action_adaln_cond: bool = False
    # "mlp" = FLOWER's TimestepEmbedder: sinusoidal features of the continuous t through a
    # learned two-layer MLP (transformers.py:638). "table" = the original lookup into a fixed
    # sinusoidal table at int(t*999), which has no learnable parameters at all -- and adaLN is
    # the only path the timestep takes into the model.
    action_time_embed: str = "mlp"          # mlp | table
    # "per_token" = FLOWER's decoder: one Linear over each action token (flower.py:1688).
    # "pooled" = the original: flatten the chunk, project to one embed_dim vector, expand it
    # back to the whole chunk -- a single bottleneck after self-attention built per-step features.
    action_decode: str = "per_token"        # per_token | pooled
    # Additive sinusoidal position on the action tokens. FLOWER only adds one when RoPE is off
    # (flower.py:1087); with RoPE on this is a second position signal on the same tokens.
    action_token_pos_embed: bool = False
    # FLOWER RMSNorms the cross-attention keys as well as the queries (transformers.py:280,
    # applied at :431). q/k norm is a pair: normalising only q leaves the logit scale riding on
    # the raw magnitude of the VLM features, which is what cond_norm/cond_linear exists to control.
    action_cross_k_norm: bool = True
    # FLOWER's use_causal_attention is true. Kept false here on purpose: flow matching denoises
    # the whole chunk in one pass, so masking token k from the tokens after it withholds part of
    # the trajectory being jointly denoised. FLOWER's own yaml comment makes the same argument.
    action_causal_attn: bool = False
    # FLOWER's sample_actions returns z.clamp(-1, 1). Under bounds normalisation the actions are
    # in [-1, 1] by construction, so without this an overshoot denormalises past the data range.
    action_clamp_output: bool = True
    # Flow-matching objective. "gaussian"/"stratified_uniform" match FLOWER; "uniform"/"beta"
    # reproduce Evo-1's original. These define what the model learns, not how it is computed,
    # so a checkpoint is only valid with the setting it was trained under.
    flow_noise: str = "gaussian"            # gaussian | uniform
    flow_time_sampling: str = "stratified_uniform"   # stratified_uniform | beta
    # Action-generation objective, for the ablation "is flow matching smoother than diffusion".
    # "flow" is everything above. "ddpm" trains the SAME DiT the way Diffusion Policy trains
    # its DDPM (integer step k ~ U{0..K-1}, x_k = sqrt(abar_k) a + sqrt(1-abar_k) eps, epsilon
    # regression) and integrates it with DDIM or ancestral DDPM (flow_matching.py
    # DDPMSchedule). The parameter table is identical between the two, so the arms differ
    # only in what the network is regressed onto and how it is integrated. A checkpoint is
    # only valid under the objective it was trained with. Absent in an old config.json means
    # "flow": every checkpoint written before this field existed was flow matching.
    action_objective: str = "flow"          # flow | ddpm
    ddpm_train_steps: int = 100             # Diffusion Policy: num_train_timesteps 100
    ddpm_beta_schedule: str = "squaredcos_cap_v2"   # squaredcos_cap_v2 | linear
    ddpm_prediction: str = "epsilon"        # epsilon | sample
    # Inference-only, like num_inference_timesteps: which reverse process integrates the
    # trained network, and DDIM's stochasticity (0 = deterministic). Evo1_server overrides
    # them from EVO1_DDPM_SAMPLER / EVO1_DDIM_ETA so one checkpoint is evaluated under several.
    ddpm_sampler: str = "ddim"              # ddim | ddpm
    ddim_eta: float = 0.0
    
    # Model defaults
    embed_dim: int = 896
    hidden_dim: int = 1024
    state_hidden_dim: int = 1024
    num_heads: int = 8
    num_categories: int = 1
    num_inference_timesteps: int = 50

    def to_dict(self):
        return asdict(self)
    
    @property
    def action_dim(self) -> int:
        ah = self.action_horizon if self.action_horizon is not None else self.horizon
        return ah * self.per_action_dim

    @classmethod
    def from_dict(cls, d: dict):
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        filtered_d = {k: v for k, v in d.items() if k in valid_keys}
        # A checkpoint's config.json only carries the fields that existed when it was written.
        # Filling the rest from the dataclass defaults would hand an old checkpoint the CURRENT
        # architecture and fail load_state_dict, so absent fields fall back to the behaviour of
        # the generation that wrote them. Two markers, newest first:
        #   `action_cross_k_norm` absent -> written by the first FLOWER port
        #   `action_block` absent        -> written before the FLOWER port at all
        # train.py passes vars(args), which carries every field, so training is never affected.
        if "action_block" not in d:
            legacy = _PRE_FLOWER_DEFAULTS
        elif "action_cross_k_norm" not in d:
            legacy = _FLOWER_V1_DEFAULTS
        else:
            legacy = {}
        for k, v in legacy.items():
            if k in valid_keys:
                filtered_d.setdefault(k, v)
        # Cross-view gates are versioned independently of the action head. The first
        # cross-view checkpoint (libero_spatial_crossview) wrote one 0-dim scalar gate per
        # direction, zero-initialized; today's defaults build a per-channel [embed_dim] gate,
        # which is a different state_dict shape and would fail strict loading. Absent field
        # means that first generation.
        if "cross_view_gate_per_channel" not in d:
            filtered_d.setdefault("cross_view_gate_per_channel", False)
            filtered_d.setdefault("cross_view_gate_init", 0.0)
        # Independent of the gate markers above: both the first run and the gate-fixed v2
        # run wrote CVP's head inside CrossViewPredictiveBranch
        # (cvp_branch.cvp_head_tp/cvp_head_wrist), never on the adapter. Absent field means
        # that layout -- CrossViewAdapter must not expect head_tp/head_wrist weights.
        if "cross_view_head_in_adapter" not in d:
            filtered_d.setdefault("cross_view_head_in_adapter", False)
        # Bottleneck checkpoints written before the view partition existed trained without
        # the step-1 mask. Same state_dict shape, different forward -- resolve to what they
        # were trained with.
        if d.get("use_bottleneck_fusion") and "bottleneck_view_partition" not in d:
            filtered_d.setdefault("bottleneck_view_partition", False)
        return cls(**filtered_d)


# What this head was before the FLOWER port. See EvoConfig.from_dict.
_PRE_FLOWER_DEFAULTS = {
    "action_block": "basic",
    # Verified against both checkpoints on this machine that predate the port (the released
    # Evo1_LIBERO and calvin_stage1): neither carries transformer_blocks.*.self_attn weights.
    "action_self_attn": False,
    "action_cond_proj": False,
    "flow_noise": "uniform",
    "flow_time_sampling": "beta",
    "action_adaln_cond": True,
    "action_time_embed": "table",
    "action_decode": "pooled",
    "action_token_pos_embed": True,
    "action_cross_k_norm": False,
    "action_causal_attn": False,
    "action_clamp_output": False,
}

# What the first FLOWER port produced: the block, the objective and cond_proj were already
# FLOWER's, the rest of the head was not.
_FLOWER_V1_DEFAULTS = {
    "action_rope_theta": 32.0,
    "action_adaln_cond": True,
    "action_time_embed": "table",
    "action_decode": "pooled",
    "action_token_pos_embed": True,
    "action_cross_k_norm": False,
    "action_causal_attn": False,
    "action_clamp_output": False,
}
