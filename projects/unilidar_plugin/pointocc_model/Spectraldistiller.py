'''
Description: Spectral distillation modules of LiEqualizer -- the low-frequency branch (TPV plane
features) and the high-frequency branch (voxel logits).

================================================================================
Overview
================================================================================
This file implements the two spectral distillation branches of LiEqualizer. They share one set of
graph-spectral operators (_GraphSpectralBase):

  * Graph Laplacian L: the grid version is the normalized Laplacian with 4 neighbors (2D) / 6
    neighbors (3D) and replicate (Neumann) boundary, whose spectrum lies in [0, 2]; the
    support-subgraph version L_Ω is defined only on support voxels/cells and their support
    neighbors (zero outside the support),
    L_Ω x = m ⊙ (deg ⊙ x − Σ_neighbors x) / (total number of neighbors).
  * Diffusion (low-pass) D = I − αL with 0 < α ≤ 1/2: this guarantees that ρ(λ) = 1 − αλ is
    non-negative and monotone on [0, 2], so different frequencies never receive the same energy
    response. Multi-scale diffusion g_b = D^b x (b = 1..S).
  * Band-pass L g_b (the local curvature at a given scale) and high-pass residual h_b = x − D^b x.
  * Energy e_b = ‖g_b‖², contracted moments, multi-scale energy-decay profiles (compared with L1),
    flatness, and related statistics.

Both branches follow a "gated match + suppress" structure: the gate is given by the teacher and
does not back-propagate gradients; where the gate is open (the teacher has content there, i.e. at
class boundaries) the student is aligned to the teacher (match / detail); where the gate is closed
(the teacher is flat inside a class) the student's spurious high frequencies are suppressed
(suppress). The two branches use different suppression targets:

  * LF branch (TPV features): where the gate is closed, the student's band-pass energy is pushed
    towards 0 (shrinkage solution).
  * HF branch (voxel logits): where the gate is closed, only the part of the student exceeding the
    teacher is penalized; the teacher itself is the optimum.

--------------------------------------------------------------------------------
LF branch TPVLowFreqSpectralDistiller (the final method uses the v5 path _match_view_v5)
--------------------------------------------------------------------------------
Input: teacher and student TPV tri-plane features (FPN/neck outputs, the layer that LF distillation
      actually acts on), xy [B,C,240,180], yz [B,C,180,16], zx [B,C,16,240]; view weights
      xy:yz:zx = 3:1:1.
Each plane undergoes multi-scale diffusion g_b = D^b x (diffuse first, then compare; the
unsmoothed raw field is not compared), and is then cut into 30×30 blocks (block_pad='fit': the
16-wide z axis is blocked by its actual width, without replicate padding). Terms:
  * match: inside each block, weighted by the gate v and normalized within the block, the
    teacher's and student's multi-scale diffusion responses g_b (low-pass responses; the
    band-pass L g_b is used only for the gate and the suppression) are compared.
  * Energy term, profile term (L1 difference of the multi-scale energy-decay curves,
    lambda_profile=8), contracted-moment term.
  * suppress (only on gated planes): with weight 1 − v, the student's band-pass energy (L g_s)²
    is pushed towards 0 (suppress_target='zero'); the reference quantity
    suppress_ref_mode='log_tail' depends only on the teacher (ref = max(E_t, ρ·P_t),
    ρ = ref_floor_rel = 1e-4); the loss value is compressed by f(u) = u (u ≤ 1) /
    1 + log u (u > 1), and the gradient back-propagated is the true derivative of the forward
    value; lambda_suppress=0.25.
Gate (only on xy, plane_gate=(True, False, False)): the teacher plane class-probability map p
(projected onto the plane from the teacher voxel logits) goes through the same multi-scale
band-pass on the teacher support subgraph, e = Σ_k (L_Ω D_Ω^b p)_k²,
τ = gate_kappa · mean_R(e) (gate_kappa=0.5), v = e/(e + τ); R is the projection of the teacher
support onto the plane with isolated cells removed, and v = 0 outside R (gate_domain='support',
gate_source='prob').
yz/zx have no gate and no suppression term; match, profile and the other terms are computed only
on the teacher support (ungated_domain='support').
After successful construction, one line "[LF v5] {json}" is printed stating what actually runs
on each of the three planes, so that one can check that the configuration has taken effect.
When no v5 switch is set, the default path _match_view (not the final method) is used.

--------------------------------------------------------------------------------
HF branch TPVHighFreqSpectralDistiller (the final method uses the support-subgraph path
_match_view_support)
--------------------------------------------------------------------------------
Input: teacher and student 8-class voxel logits on the teacher support Ω_T (a common support
field; structural zeros outside the support).
Operator: hf_operator='support', i.e. the 6-neighbor support-subgraph Laplacian L_Ω (steps at the
support boundary are not treated as high frequency).
Terms:
  * Gate: given by the high-pass residual of the teacher class probabilities
    (gate_source='prob'); all channels of the same voxel share one gate (gate_share='voxel'),
    τ = gate_kappa · mean_R(e) (gate_kappa=0.5), w = e/(e + τ), with statistics taken only
    over the gated region R (= Ω_T) (gate_domain='support').
  * detail (match where the gate is open) and suppress (where the gate is closed): residuals are
    measured in units of the teacher residual scale (hf_norm='teacher'; no denominator on the
    student side, so the gradient is the true gradient of the forward loss); suppress penalizes
    only relu(n_s² − n_t²), i.e. the part of the student exceeding the teacher
    (suppress_target='teacher'). The weight of the gate-open match is lambda_detail
    (default 1.0).
  * Value matching L_V (lambda_value=20): teacher and student logits, with the channel mean
    removed and normalized by the RMS over Ω_T, are compared voxel by voxel inside Ω_T
    (class-weighted).
  * Continuity L_C (lambda_cont=20): on adjacent voxel pairs whose both ends lie where the
    teacher gate is closed, the difference of the student's class probabilities is penalized,
    i.e. the Dirichlet energy of the student probability field on the "teacher non-boundary
    subgraph".
  * Spectral statistics: energy (2.0), contracted moments (2.0), profile L1 (5.0), flatness
    hinge (1.0), local Euler-characteristic curve ECC (4.0, softmax-probability input); after
    the 8-scale pyramid, diffusion is further accumulated to 16 and 32 steps (extra_scales),
    and the gated detail/suppress terms are reused on those residuals.
  * Class weights class_weights = [1,1,1,1,1,2,2,1].
hf_operator='grid' is the grid-operator path (not the final method).

--------------------------------------------------------------------------------
Theoretical correspondence (the two cases of Theorem 1(iv))
--------------------------------------------------------------------------------
Treating the residual as a free variable, the optimum of the gated match + suppress objective is:
  * LF branch (push to 0): pointwise in the suppressed quantity the optimum is the teacher's value
    shrunk towards 0 by the factor a/(a + λ_sup(1 − a)) (close to the teacher where the gate is
    open, close to 0 where it is closed); at field level only the complementary curvature-energy
    bound holds, since T_L = L couples neighbouring coordinates. For intra-class detail that the student cannot predict from sparse
    input, the MSE-optimal estimate is its conditional expectation 0.
  * HF branch (penalize only the part exceeding the teacher): the optimum is the teacher itself,
    independent of the gate; the gate only decides which side is penalized (both sides are pulled
    towards the teacher when the gate is open; only the excess over the teacher is penalized
    when it is closed), which coincides with the optimum of KD.

--------------------------------------------------------------------------------
Ablation switches (not the final method; the code is kept to support ablations)
--------------------------------------------------------------------------------
  * enable_hf_distill / enable_lf_distill = False: remove an entire branch (set in the
    upper-level model).
  * No gating: HF gate off, LF lf_gate=False, lambda_cont=0.
  * "suppress only": HF lambda_detail=0, LF lambda_match_gated=0.
  * "match only": HF/LF lambda_suppress=0.

Note on invariance: norm normalization only gives "per-sample positive-gain invariance" (network
temperature); the multi-scale energies {e_b} are invariant to orthogonal transforms that commute
with L. Neither removes the effects of sparse sampling itself (attenuation and aliasing);
robustness to sampling comes from estimator design choices -- "diffuse first, then compare",
contracted moments, and gated suppression.
================================================================================
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from typing import List, Dict, Tuple, Optional


# =============================================================================
# Non-reentrant segment recomputation
# =============================================================================

class _RecomputeSegment(torch.autograd.Function):
    """Non-reentrant segment recomputation (replacement for torch.utils.checkpoint.checkpoint).

    Purpose: the spectral-operator segments (diffusion pyramid, band-pass, energy) are
    memory-heavy, so intermediate results are not saved in the forward pass and are recomputed in
    the backward pass. Why not the built-in checkpoint: in torch<=1.10, CheckpointFunction raises
    an error directly under torch.autograd.grad(), and the detector's dynamic loss balancer uses
    exactly autograd.grad to measure the gradient norm of the distillation losses -- the built-in
    implementation would make that measurement fail and leave the distillation weight stuck at 0.
    This implementation rebuilds the segment's subgraph inside backward with a nested
    autograd.grad and computes the input gradients, so it is transparent to both an outer
    .backward() and .grad().

    Constraint: the segment function must be pure -- no trainable parameters, no random
    operators, no external side effects (all spectral-operator segments in this file satisfy
    this; read-only constant buffers are allowed)."""

    @staticmethod
    def forward(ctx, fn, *args):
        # Record the segment function and whether each input requires grad; save the inputs so
        # the segment can be recomputed in backward.
        ctx.fn = fn
        ctx.input_requires_grad = [torch.is_tensor(a) and a.requires_grad for a in args]
        ctx.save_for_backward(*args)
        # The forward runs under no_grad: no graph is built and no intermediate activations are
        # saved -- this is where the memory saving comes from.
        with torch.no_grad():
            out = fn(*args)
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        # Retrieve the saved inputs, cut them from the original graph, and re-attach the original
        # requires_grad flags.
        saved = ctx.saved_tensors
        detached = []
        for a, req in zip(saved, ctx.input_requires_grad):
            d = a.detach()
            d.requires_grad_(req)
            detached.append(d)
        # Recompute the segment function with grad enabled to obtain a differentiable local
        # subgraph.
        with torch.enable_grad():
            out = ctx.fn(*detached)
        outputs = (out,) if torch.is_tensor(out) else tuple(out)
        # Back-propagate only through outputs that do require grad after recomputation AND for
        # which the outer graph actually passed a gradient (teacher-branch outputs carry no
        # gradient in the recomputed graph, and the corresponding outer grad is None as well).
        diff_out, diff_go = [], []
        for o, go in zip(outputs, grad_outputs):
            if torch.is_tensor(o) and o.requires_grad and go is not None:
                diff_out.append(o)
                diff_go.append(go)
        # Compute the gradient of each input with a nested autograd.grad; inputs that do not
        # require grad get None.
        grad_inputs = [None] * len(detached)
        need = [d for d, req in zip(detached, ctx.input_requires_grad) if req]
        if diff_out and need:
            gs = torch.autograd.grad(
                diff_out, need, grad_outputs=diff_go, allow_unused=True)
            it = iter(gs)
            for i, req in enumerate(ctx.input_requires_grad):
                if req:
                    grad_inputs[i] = next(it)
        # The first return value corresponds to fn (not a tensor, no gradient).
        return (None,) + tuple(grad_inputs)


# =============================================================================
# Shared spectral-operator base class
# =============================================================================

class _GraphSpectralBase(nn.Module):
    """
    Shared spectral-operator base class: the symmetric Neumann graph Laplacian on a regular grid,
    and the diffusion, energy, profile and moment statistics built on top of it. Both
    TPVLowFreqSpectralDistiller (LF, 2D tri-plane features) and TPVHighFreqSpectralDistiller
    (HF, 3D voxel logits) inherit from it.

    Grid Laplacian (Lemma 1):
        2D: L = I - A, where A is the 4-neighbor mean (replicate padding folds ghost neighbors
            back onto the diagonal);
        3D: the 6-neighbor version, with weights -1/6.
    This guarantees L = Lᵀ ⪰ 0, spec(L) ⊂ [0,2], xᵀLx = (1/2d)·Σ_edges (x_u - x_v)², and that
    constant vectors lie in the kernel -- this is what legitimizes all subsequent "spectral"
    language.
    These properties rely on pad_mode='replicate' (Neumann boundary), which is the default of
    both distillers.

    In the final method:
    - LF's multi-scale low-pass g_b = D^b x, the band-pass L g_b in the gate and suppression
      terms, and the per-scale Dirichlet energy all use the 2D grid operator here;
    - HF uses hf_operator='support', replacing diffusion and energy with the subclass's own
      support-subgraph operator L_Ω (_lap_op / _diffuse_op / _dirichlet_energy_op) and not
      calling _apply_laplacian / _diffuse / _dirichlet_energy here; it still shares α, eps, the
      segment recomputation _ckpt, per-sample/per-channel energies, profile L1, profile flatness,
      and moment alignment _moment_mse.
    """

    def __init__(self, alpha: float = 0.25, pad_mode: str = "replicate", eps: float = 1e-6,
                 use_checkpoint: bool = True, moment_norm: str = "per_scale_channel"):
        # Arguments:
        #   alpha          diffusion step α, D_α = I − αL; both distillers default to 0.25,
        #                  which is the value used by the final method.
        #   pad_mode       boundary padding of the grid Laplacian; 'replicate' folds ghost
        #                  neighbors back onto the cell itself, i.e. a Neumann boundary.
        #   eps            numerical floor for all normalizations, logarithms and divisions.
        #   use_checkpoint whether to apply segment recomputation to per-scale/per-threshold
        #                  segment functions (see _ckpt).
        #   moment_norm    normalization axis for contracted-moment alignment (see _moment_mse).
        super().__init__()
        # Normalization axis of the contracted moments: "joint" = normalize once over the joint
        # (channel × scale) axis (final method; both HF and LF set moment_norm='joint' explicitly
        # in the config); "per_scale_channel" = normalize once along the channel axis for each
        # scale (not the final method: constructor default / ablation). See _moment_mse for the
        # rationale.
        assert moment_norm in ("joint", "per_scale_channel"), moment_norm
        self.moment_norm = moment_norm
        # Only α ∈ (0, 1/2] guarantees that ρ(λ) = 1 − αλ is non-negative and monotonically
        # decreasing on spec(L) ⊂ [0,2], so that the powers of D_α form a monotone low-pass filter
        # bank. For α ∈ (1/2, 1), D_α is still a contraction, but ρ changes sign at the
        # high-frequency end and ρ² is no longer monotone, so different frequencies receive the
        # same response (band aliasing) and the multi-scale responses lose the ability to
        # distinguish frequencies (identifiability fails).
        assert 0.0 < alpha <= 0.5, (
            f"alpha={alpha}: need 0 < alpha <= 1/2 so that rho(l)=1-alpha*l stays in "
            f"[0,1] and monotone on spec(L) ⊂ [0,2] (Remark 1). alpha in (1/2,1) gives a "
            f"contraction but NOT a low-pass filter bank."
        )
        self.alpha = float(alpha)
        self.pad_mode = pad_mode
        self.eps = float(eps)
        # Segment-wise gradient checkpointing: the segment functions of both distillers are pure
        # functions without trainable parameters (read-only constant buffers only), but they
        # perform multi-scale diffusion, multi-order moments and multi-threshold ECC on
        # full-resolution tensors, so the intermediate activations kept by autograd are huge
        # (a single fp32 HF voxel-logit tensor [1,8,480,360,32] is about 177MB, and each scale
        # produces several of them). With recomputation per scale / per threshold, the forward
        # keeps only the segment-boundary tensors and the backward recomputes segment by segment
        # (no random operators and no parameters inside a segment, so the recomputation matches
        # the forward exactly; DDP compatible).
        self.use_checkpoint = bool(use_checkpoint)

        # 2D Laplacian kernel [1,1,3,3]: center 1, the 4 neighbors (up/down/left/right) −1/4
        # each, i.e. (Lx)_u = x_u − mean_{v~u} x_v.
        k2 = torch.zeros(1, 1, 3, 3)
        k2[0, 0, 1, 1] = 1.0
        for dy, dx in ((0, 1), (2, 1), (1, 0), (1, 2)):      # positions of the up/down/left/right neighbors in the 3×3 kernel
            k2[0, 0, dy, dx] = -0.25
        self.register_buffer("K_lap_2d", k2)

        # 3D Laplacian kernel [1,1,3,3,3]: center 1, the 6 neighbors (±H, ±W, ±Z) −1/6 each.
        # Both kernels are registered as buffers in __init__, so they automatically live on the
        # correct device under .to(device) / DataParallel / torch.compile. In the final method HF
        # uses the support-subgraph operator and does not read the 3D kernel; it only serves the
        # hf_operator='grid' path.
        k3 = torch.zeros(1, 1, 3, 3, 3)
        k3[0, 0, 1, 1, 1] = 1.0
        w = -1.0 / 6.0
        for idx in ((0, 1, 1), (2, 1, 1), (1, 0, 1), (1, 2, 1), (1, 1, 0), (1, 1, 2)):   # 6-neighbor positions
            k3[0, 0, idx[0], idx[1], idx[2]] = w
        self.register_buffer("K_lap_3d", k3)

    def _ckpt(self, fn, *tensors):
        """Segment-recomputation wrapper: during training, when any tensor input requires grad,
        the segment function fn is run through _RecomputeSegment with non-reentrant segment
        recomputation (intermediate activations inside the segment are not kept in the forward
        and are recomputed in the backward); otherwise (inference, use_checkpoint=False, or no
        input requiring grad) fn is called directly. The return value is the same as
        fn(*tensors) (a tensor or a tuple of tensors).
        fn must be pure (no trainable parameters, no random operators, no side effects);
        non-tensor arguments are passed via closure.
        The custom _RecomputeSegment is used instead of torch.utils.checkpoint because the latter,
        in torch<=1.10, is incompatible with the dynamic loss balancer measuring gradient norms via
        autograd.grad; the former is transparent to both .backward() and autograd.grad. In the
        final method, every per-scale segment of LF/HF, HF's contracted moments, and every ECC
        threshold go through it."""
        # Recomputation is only worthwhile when back-propagation will actually happen; if any
        # input requires grad, take the recomputation path
        if self.use_checkpoint and torch.is_grad_enabled() and any(
                torch.is_tensor(a) and a.requires_grad for a in tensors):
            return _RecomputeSegment.apply(fn, *tensors)
        return fn(*tensors)

    # ---- Basic operators ------------------------------------------------------

    def _apply_laplacian(self, x: torch.Tensor) -> torch.Tensor:
        """Grid Laplacian L·x, applied independently per channel (depthwise convolution); the
        output has the same shape as x.
        Input 2D [B,C,H,W] (4 neighbors) or 3D [B,C,H,W,Z] (6 neighbors). The border is first
        padded by one cell according to pad_mode and then convolved without padding; with
        replicate this is a Neumann boundary. The kernel is cast to the input dtype (AMP safe).
        In the final method: LF's diffusion D, the teacher band-pass L g_t in the gate and the
        suppression reference, and the student band-pass L g_s in the suppression term all use it
        (2D branch); HF uses the support-subgraph operator L_Ω and does not call it (the 3D branch
        is only used when hf_operator='grid')."""
        if x.dim() == 4:
            # 2D: expand the single-channel kernel to [C,1,3,3]; with groups=C the same Laplacian
            # is applied independently to every channel
            C = x.shape[1]
            k = self.K_lap_2d.expand(C, 1, 3, 3).to(dtype=x.dtype)
            xp = F.pad(x, (1, 1, 1, 1), mode=self.pad_mode)          # pad 1 at both ends of W and H → [B,C,H+2,W+2]
            return F.conv2d(xp, k, bias=None, stride=1, padding=0, groups=C)   # → [B,C,H,W]
        assert x.dim() == 5, f"Expect [B,C,H,W] or [B,C,H,W,Z], got {tuple(x.shape)}"
        # 3D: expand the kernel to [C,1,3,3,3]; the 6-tuple of F.pad pads both ends of Z, W, H in
        # that order → [B,C,H+2,W+2,Z+2]
        C = x.shape[1]
        k = self.K_lap_3d.expand(C, 1, 3, 3, 3).to(dtype=x.dtype)
        xp = F.pad(x, (1, 1, 1, 1, 1, 1), mode=self.pad_mode)
        return F.conv3d(xp, k, bias=None, stride=1, padding=0, groups=C)      # → [B,C,H,W,Z]

    def _diffuse(self, x: torch.Tensor) -> torch.Tensor:
        """One step of stable diffusion (low-pass): D_α x = x − α·Lx, with spectral response
        ρ(λ) = 1 − αλ ∈ [0,1] (α ≤ 1/2).
        Calling it b times in a row gives D^b x with response ρ(λ)^b; the larger b, the lower the
        retained band. Input and output have the same shape (2D/3D).
        In the final method it is used for LF's multi-scale low-pass pyramid g_b = D^b x; HF uses
        the support-subgraph version _diffuse_op instead."""
        return x - self.alpha * self._apply_laplacian(x)

    @staticmethod
    def _spatial_dims(x: torch.Tensor) -> Tuple[int, ...]:
        # Spatial axis indices (2, 3) or (2, 3, 4): the tensor layout is [B,C,spatial...];
        # reducing over these axes gives per-(B,C) statistics.
        return tuple(range(2, x.dim()))

    def _dirichlet_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Per-channel Dirichlet energy xᵀLx = (1/2d)Σ_edges(x_u − x_v)² ≥ 0 (Lemma 1),
        [B,C,...] → [B,C].
        Measures how much each channel field oscillates spatially.
        In the final method: LF computes it on g_b at every scale, L2-normalizes both sides along
        the channel axis and takes the MSE (the λ_energy term); HF uses the support-subgraph
        version _dirichlet_energy_op instead."""
        return torch.sum(x * self._apply_laplacian(x), dim=self._spatial_dims(x))   # Σ_u x_u·(Lx)_u

    def _channel_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Per-channel energy ‖x_c‖², [B,C,...] → [B,C]. On the diffusion pyramid g_b = D^b x it is
        the b-th contracted moment e_b (Lemma 2).
        In the final method: LF records e_b of g_b at every scale for moment alignment; HF
        computes contracted moments on the diffusion chain of the final residual in
        _contracted_moment_loss_m."""
        return torch.sum(x * x, dim=self._spatial_dims(x))

    def _sample_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample energy ‖x‖² (summed over channels and space together), [B,C,...] → [B].
        In the final method: HF computes it on the residual h_b = x − D^b x at every scale,
        giving the b-th component of the residual scale-energy profile (used by _profile_l1 and
        _profile_flatness)."""
        return torch.sum(x * x, dim=tuple(range(1, x.dim())))

    def _normalize_response(self, x: torch.Tensor, per_channel: bool = False) -> torch.Tensor:
        """
        Shape normalization: x / sqrt(Σx² + eps), output has the same shape as x. With
        per_channel=False all (channel, space) entries of each sample are normalized together;
        with True each (sample, channel) is normalized separately.
        It only provides invariance to a per-sample positive gain and cannot remove the
        attenuation and aliasing caused by sampling; the latter is handled by the design of the
        loss terms (gated alignment, suppression term, profiles and moments).
        Not called by the final method: only used in LF's ungated, unblocked basic path, and in
        HF's match term when ungated and gate_norm ≠ 'mean_rms'.
        """
        # Axes to normalize over: spatial axes only when per channel, otherwise channel + spatial
        dims = self._spatial_dims(x) if per_channel else tuple(range(1, x.dim()))
        denom = torch.sqrt(torch.sum(x * x, dim=dims, keepdim=True) + self.eps)   # eps avoids division by zero for an all-zero field
        return x / denom

    # ---- Profile statistics ---------------------------------------------------

    def _js_divergence(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Jensen–Shannon divergence between row-normalized distributions (symmetric, bounded by
        log 2), p,q: [B,S] → scalar (batch mean).
        p and q are each first row-normalized into distributions, so only the shape is compared,
        independent of magnitude; eps avoids division by zero and log(0).
        Not the final method: used for profile alignment when profile_metric='js' (the class
        default); the final method uses _profile_l1."""
        p = p / (p.sum(dim=-1, keepdim=True) + self.eps)            # row-normalize into a distribution
        q = q / (q.sum(dim=-1, keepdim=True) + self.eps)
        m = 0.5 * (p + q)                                            # mixture distribution
        kl_pm = torch.sum(p * (torch.log(p + self.eps) - torch.log(m + self.eps)), dim=-1)   # KL(p‖m) [B]
        kl_qm = torch.sum(q * (torch.log(q + self.eps) - torch.log(m + self.eps)), dim=-1)   # KL(q‖m) [B]
        return (0.5 * (kl_pm + kl_qm)).mean()

    def _profile_l1(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """First-order profile distance: total variation after row normalization,
        0.5·Σ|p̂ − q̂| ∈ [0,1], p,q: [B,S] → scalar (batch mean).
        Why this instead of JS/KL: JS and both forward and reverse KL are second-order small
        quantities near p≈q (≈ χ²/2), so their gradients vanish as teacher and student align;
        L1 is first order and its gradient magnitude does not vanish with alignment. Row
        normalization ensures that only the profile shape is compared, independent of the overall
        magnitude.
        In the final method both HF and LF use it (profile_metric='l1'): HF compares the residual
        scale-energy profile, LF compares the energy profile of the multi-scale low-pass
        responses."""
        p = p / (p.sum(dim=-1, keepdim=True) + self.eps)            # row-normalize into a distribution
        q = q / (q.sum(dim=-1, keepdim=True) + self.eps)
        return (0.5 * torch.sum(torch.abs(p - q), dim=-1)).mean()   # per-sample total variation, then batch mean

    def _profile_flatness(self, p: torch.Tensor) -> torch.Tensor:
        """
        Geometric mean / arithmetic mean GM/AM of the normalized profile along the **diffusion
        scale axis**, p: [B,n] → [B].
        The input is HF's residual energy E_b = ‖x − D^b x‖² at n diffusion scales (n=8; b is the
        number of diffusion steps), not the energy over graph-Fourier frequencies; the axes
        differ, so do not read it as spectral flatness, and do not use flatness values on the
        graph-Fourier axis to calibrate λ_flat.

        Properties:
        - E_b = Σ_λ (1 − ρ(λ)^b)²·S(λ), with ρ(λ) = 1 − αλ and S(λ) the graph-Fourier energy
          spectrum; i.e. E_b is a linear functional of S.
        - After row normalization Σp = 1 ⇒ AM ≡ 1/n, and this quantity = n·GM(p). For α = 0.25,
          ρ ∈ [0.5, 1], and every component with λ > 0, (1 − ρ^b)², is strictly increasing in b;
          a convex combination of them is still strictly increasing, so a uniform profile is
          unreachable: this quantity is strictly less than 1 and only takes values in a narrower
          interval within (0,1].
        - This quantity depends only on n linear functionals of S: one can construct two signals
          with the same profile but vastly different graph-Fourier flatness, so it is neither
          equivalent to graph-Fourier flatness nor a bound on it in either direction.
        - Numerical edge case: when Σp ≲ eps, both GM and AM are dominated by eps and the result
          degenerates to ≈1; the residual energy of real voxel logits is many orders of magnitude
          larger than eps, so this degeneracy is not triggered.

        In the final method: HF computes this quantity for both student and teacher and uses the
        one-sided hinge relu(F_s − F_t) to penalize only "the student profile being flatter than
        the teacher's" (energy saturating earlier, i.e. too large a share of high-frequency
        residual), λ_flat = 1.0. When the constraint is satisfied, the term and its gradient are
        exactly 0.
        """
        p = p / (p.sum(dim=-1, keepdim=True) + self.eps)            # row-normalize into a distribution
        gm = torch.exp(torch.mean(torch.log(p + self.eps), dim=-1))   # geometric mean exp(mean log p)
        am = torch.mean(p, dim=-1)                                     # arithmetic mean (≈ 1/n)
        return gm / (am + self.eps)

    # ---- Contracted-moment alignment ------------------------------------------

    def _moment_mse(self, es: List[torch.Tensor], et: List[torch.Tensor]) -> torch.Tensor:
        """
        Alignment term for the contracted-moment sequence {e_b}_{b=1..K} (per-channel energies).
        es/et are lists of length K, each item [B,C]; returns a scalar.

        The normalization is determined by moment_norm:
        - "joint" (final method): flatten [B,C,K] into [B, C·K], L2-normalize once over the joint
          (channel × scale) axis and then take the MSE. It removes only one positive gain per
          sample, which is exactly the global-scaling invariance of Proposition 5(ii), no more and no less;
          the energy decay along the scale axis -- i.e. the moment sequence itself -- is fully
          preserved.
        - "per_scale_channel" (not the final method: constructor default / ablation): normalize
          each scale separately along the channel axis. Its invariance group is "per-sample ×
          per-scale" positive gains, larger than the global-scaling invariance of Proposition 5(ii), and what is additionally quotiented out
          is precisely the decay along the scale axis: any field in which "all channels share the
          same spatial pattern and differ only in channel amplitude" (channel⊗space rank-1) gives
          loss ≡ 0 and gradient ≡ 0, and for C=1 it is identically 0 for any two fields.
        joint is multiplied by K at the end, so that the per-element average magnitude (1/C)
        matches per_scale_channel and both modes share the same lambda_moment.
        Note: a finite number K of moments cannot fully identify a general high-dimensional
        spectrum in the first place; joint normalization only removes the "whole class identically
        zero" non-identifiability and does not claim completeness.
        In the final method both LF (_match_view_v5, reusing the per-channel energies of the
        multi-scale pyramid) and HF (_contracted_moment_loss_m) call it.
        """
        E_s = torch.stack(es, dim=-1)                      # [B,C,K]
        E_t = torch.stack(et, dim=-1)
        B, _, K = E_s.shape
        if self.moment_norm == "per_scale_channel":        # not the final method: per-scale normalization along the channel axis
            loss = E_s.new_tensor(0.0)
            for k in range(K):
                # k-th moment: teacher and student each L2-normalized along channels, then MSE
                loss = loss + F.mse_loss(F.normalize(E_s[..., k], p=2, dim=1),
                                         F.normalize(E_t[..., k], p=2, dim=1))
            return loss / float(max(1, K))                 # average over the K scales
        # Final method, joint: [B,C,K] → [B,C·K], only one L2 normalization per sample
        ms = F.normalize(E_s.reshape(B, -1), p=2, dim=1)
        mt = F.normalize(E_t.reshape(B, -1), p=2, dim=1)
        return F.mse_loss(ms, mt) * float(K)               # mse averages over B·C·K elements; ×K gives the same magnitude as the per-scale version

    def _contracted_moment_loss(self, xs: torch.Tensor, xt: torch.Tensor,
                                num_moments: int) -> torch.Tensor:
        """
        Contracted spectral-moment alignment (Theorem 1(iii)): e_k = ‖D_α^k x‖² (per channel),
        k = 1..K, compared between teacher and student with _moment_mse.
        xs/xt: same shape [B,C,...]; num_moments = K; returns a scalar.
        Compared with the raw L^k moments tr(xᵀL^k x) (_legacy_lk_moment_loss): the latter amplifies
        the top frequency band, where sampling-aliasing noise concentrates, at the geometric rate
        (λ_hi/λ_lo)^k, while the former suppresses it at the geometric rate (ρ_hi/ρ_lo)^{2k},
        which makes it robust to noise.
        Caller: the HF grid-operator path (hf_operator='grid') calls it on the final residual
        h = x − D^n x. The final method's HF uses the structurally identical support-subgraph
        version _contracted_moment_loss_m; LF directly reuses the per-channel energies of the
        multi-scale pyramid and does not call this function either.
        """
        es: List[torch.Tensor] = []
        et: List[torch.Tensor] = []
        gs, gt = xs, xt
        # Advance the diffusion chain g_k = D^k x step by step, recording the per-channel energy
        # e_k [B,C] at each step
        for _ in range(int(num_moments)):
            gs = self._diffuse(gs)
            gt = self._diffuse(gt)
            es.append(self._channel_energy(gs))
            et.append(self._channel_energy(gt))
        return self._moment_mse(es, et)

    def _legacy_lk_moment_loss(self, xs: torch.Tensor, xt: torch.Tensor,
                               num_moments: int) -> torch.Tensor:
        """Raw L^k moment alignment: m_k = xᵀL^k x (per channel), k = 1..K; for each order,
        teacher and student are each L2-normalized along channels and compared with MSE, then
        averaged over K. xs/xt: same shape [B,C,...]; returns a scalar. Uses only the grid
        operator _apply_laplacian.
        Not the final method: only used for ablation when legacy_lk_moments=True; its noise
        sensitivity grows exponentially with k (Theorem 1(iii)), and it is not allowed with the
        HF support-subgraph operator."""
        loss = xs.new_tensor(0.0)
        Ls, Lt = xs, xt
        for _ in range(int(num_moments)):
            Ls = self._apply_laplacian(Ls)                 # L^k x
            Lt = self._apply_laplacian(Lt)
            # m_k = Σ x·L^k x → [B,C]; L is symmetric positive semi-definite, so in theory this is
            # ≥ 0; the clamp only removes negative values caused by floating-point error
            ms = torch.clamp(torch.sum(xs * Ls, dim=self._spatial_dims(xs)), min=0.0)
            mt = torch.clamp(torch.sum(xt * Lt, dim=self._spatial_dims(xt)), min=0.0)
            loss = loss + F.mse_loss(F.normalize(ms, p=2, dim=1), F.normalize(mt, p=2, dim=1))
        return loss / float(max(1, num_moments))


# =============================================================================
# Low-frequency distillation (TPV tri-plane views)
# =============================================================================

# Legal values (LF_V5_CHOICES) and defaults (LF_V5_DEFAULTS) of the structural switches of
# TPVLowFreqSpectralDistiller.
# Module-level constants, not part of the state_dict. Values are validated one by one at
# construction (illegal values raise ValueError); when all effective values equal the defaults,
# forward takes the basic path _match_view, and if any differs it takes _match_view_v5. The final
# method takes _match_view_v5.
# Meaning of each switch (final-method value in parentheses):
#   block_pad         how to pad when the plane size is not divisible by the block size during
#                     block alignment: 'replicate' replicates edges according to pad_mode;
#                     'mask' pads with 0 and gives padded cells weight 0, so they enter neither
#                     the block norm nor the loss; 'fit' first clips the block size to at most the
#                     plane size and then behaves like 'mask', so that yz/zx blocks fit the
#                     16-wide z axis ('fit').
#   gate_domain       statistics domain of the gate and suppression term on gated planes: 'plane'
#                     the whole plane; 'support' the teacher support (with isolated cells that
#                     have no support neighbor removed) ('support').
#   gate_source       gate criterion: 'feature' the teacher feature band-pass (L g_t)²; 'prob' the
#                     multi-scale band-pass energy of the teacher plane class probabilities on the
#                     support subgraph ('prob').
#   profile_domain    weighting of the profile energy on gated planes: 'plane' unweighted; 'open'
#                     weighted by the gate v ('open').
#   plane_gate        whether each of the (xy, yz, zx) planes is gated ((True, False, False): only
#                     xy is gated).
#   suppress_target   target of the suppression term: 'zero' pushes the student band-pass energy
#                     towards 0 where the gate is closed; 'teacher' penalizes only the part of the
#                     student exceeding the teacher ('zero').
#   suppress_ref_mode reference quantity of the suppression term: 'detach_floor'
#                     ref = max(detach E_s, E_t); 'log_tail' the reference depends only on the
#                     teacher, T = max(E_t, ρ·P_t), and the suppression amount takes 1 + log u
#                     when u > 1 ('log_tail', ρ = ref_floor_rel).
#   block_weight      weighting across blocks: 'equal' equal block weights (when there is a
#                     statistics domain, only blocks containing in-domain cells count); 'support'
#                     weighted by the number of statistics-domain cells in the block (the default
#                     'equal' is used).
#   ungated_domain    statistics domain on ungated planes: 'plane' the whole plane; 'support' the
#                     teacher support with isolated cells removed, R_u ('support').
LF_V5_CHOICES = dict(block_pad=("replicate", "fit", "mask"), gate_domain=("plane", "support"),
                     gate_source=("feature", "prob"), profile_domain=("plane", "open"),
                     suppress_target=("zero", "teacher"), suppress_ref_mode=("detach_floor", "log_tail"),
                     block_weight=("equal", "support"), ungated_domain=("plane", "support"))
LF_V5_DEFAULTS = dict(block_pad="replicate", gate_domain="plane", gate_source="feature",
                      profile_domain="plane", plane_gate=(True, True, True), suppress_target="zero",
                      suppress_ref_mode="detach_floor", block_weight="equal", ungated_domain="plane")
# Switches that are all reset to their defaults when lf_gate=False (no-gate ablation): the gate
# itself (plane_gate) and the switches that act only on the gate/suppression term.
# The reset switches are recorded in lf_v5_ignored and are visible in the construction log. After
# the reset, the gate and the suppression term are not computed at all.
# gate_domain / profile_domain act only on gated planes and do not affect ungated planes.
LF_V5_GATE_SWITCHES = ("gate_domain", "gate_source", "profile_domain", "plane_gate",
                       "suppress_target", "suppress_ref_mode")
# Non-gating switches that keep their effective values when lf_gate=False: they change the ungated
# block alignment / profile itself, do not read the teacher curvature, and do not compute the gate
# or the suppression term. ungated_domain belongs to this class: the no-gate ablation arm must use
# the same value as the final method, so that the two arms differ only in the gate.
# (The construction logic only iterates over LF_V5_GATE_SWITCHES; switches not in it are kept.
# This tuple is not read anywhere in this file and serves as documentation.)
LF_V5_UNGATED_ACTIVE = ("block_pad", "block_weight", "ungated_domain")

# Fixed prefix tag of the LF construction log (for grep); the log line format is tag + space +
# one line of JSON (see lf_v5_log_line).
LF_V5_LOG_TAG = "[LF v5]"


def _lf_v5_is_rank0() -> bool:
    """Whether the current process is responsible for printing: when DDP is initialized, only
    rank 0 returns True; when not initialized (single GPU / CPU / tests) returns True.
    Used by _lf_v5_emit to avoid every process printing the construction log in multi-GPU
    training."""
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def _lf_v5_emit(msg: str) -> None:
    """Emit one line of the LF construction log (rank 0 only). Called once at the end of
    TPVLowFreqSpectralDistiller.__init__; msg is lf_v5_log_line(): the LF variant actually in
    effect (per-plane gating state, statistics domains, profile weighting, suppression settings,
    etc.), used to check the run configuration.
    The training entry tools/train.py has already created the root logger with
    get_root_logger(name='mmdet') before build_model (console + work_dir/<timestamp>.log), so this
    is then equivalent to mmcv.utils.print_log(msg, logger=get_root_logger()); if the logger is
    not initialized (tests, analysis scripts) or INFO is not visible, it falls back to
    print(flush=True).
    It only queries (loggerDict), does not create a logger, and does not change any logging
    configuration."""
    if not _lf_v5_is_rank0():
        return                                             # non-rank-0 processes do not print
    import logging
    # Only look up an existing "mmdet" logger; loggerDict.get does not create one on the fly the
    # way getLogger does
    logger = logging.Logger.manager.loggerDict.get("mmdet")
    # loggerDict may hold a placeholder (PlaceHolder) instead of a Logger; require a real Logger
    # with handlers attached and INFO level visible
    if isinstance(logger, logging.Logger) and logger.handlers and logger.isEnabledFor(logging.INFO):
        logger.info(msg)                                   # via the mmdet root logger: console + work_dir log file
    else:
        print(msg, flush=True)                             # fallback: print directly to stdout


class TPVLowFreqSpectralDistiller(_GraphSpectralBase):
    """
    LF spectral distillation: aligns the low-frequency structure of the teacher and student TPV
    tri-plane features (neck outputs; xy [B,C,240,180], yz [B,C,180,16], zx [B,C,16,240]).
    forward computes one loss per plane; total loss =
    (w_xy·L_xy + w_yz·L_yz + w_zx·L_zx)/view_divisor, which is (3·xy + yz + zx)/3 in the final
    method.

    All comparisons are made on S = num_scales diffusion scales g_b = D^b x (b=1..S,
    D = I − αL), i.e. diffuse first, then compare; the raw field b=0 is not compared
    (corresponding to Theorem 1(iii)). Per-plane loss:
      (a) match: shape alignment of the multi-scale low-pass responses. The plane is cut into
          block_size blocks; each block is normalized over (C,bh,bw) and
          ‖n_s − n_t‖² = 2−2cosθ is computed, then averaged over blocks. What is compared is the
          low-pass response g_b itself, not the band-pass.
      (b) suppress (only on gated planes, only at suppress_scales): where the gate is closed
          (where the teacher is flat), the student band-pass energy (L g_s)² is suppressed, i.e.
          the spurious structure that the student produces where the teacher has none.
      (c) alignment of the relative inter-channel distribution of the per-channel Dirichlet
          energy g_bᵀ L g_b (lambda_energy).
      (a)(b)(c) form the single-scale loss, averaged with weights eta_b.
      (d) relative multi-scale response-profile alignment: q_b = e_b/Σ_b e_b,
          e_b = Σ_c ‖g_b,c‖², i.e. the DRD statistic used in the evaluation (corresponding to
          Theorem 1(i)(ii)), independent of magnitude; the distance is L1 total variation or JS
          (lambda_profile). For the spatial weighting of e_b see profile_domain (gated planes)
          and ungated_domain (ungated planes).
      (e) alignment of the contracted spectral moments e_k = ‖D^k x‖², directly reusing the
          per-channel energies of the pyramid, without extra convolutions (lambda_moment).

    The gate (lf_gate=True) is computed from the teacher and does not back-propagate gradients:
    where the gate is open (v≈1, the teacher has structure) a v-weighted match is applied, and
    where it is closed (v≈0) suppress is applied.

    Two implementation paths:
      - _match_view: used when all switches in LF_V5_DEFAULTS take their default values (not
        the final method).
      - _match_view_v5: used when the effective value of any switch is non-default; the final
        method takes this path. Only the xy plane is gated: gate = multi-scale band-pass energy
        e of the teacher plane class-probability map on the teacher support subgraph,
        v = e/(e+τ), τ = gate_kappa·mean_R(e); yz/zx are ungated and only matched on the
        teacher support.
    """

    def __init__(
        self,
        # ---- Diffusion pyramid and basic loss terms ----
        # num_scales: number of diffusion scales S (compare g_b = D^b x, b=1..S). Final method 8.
        num_scales: int = 8,
        # alpha: diffusion step α, D = I − αL. The base class requires 0 < α ≤ 1/2, so that
        #   ρ(λ)=1−αλ is non-negative and monotone on [0,2] (a true low-pass). Final method 0.25.
        alpha: float = 0.25,
        # eta: weights η_b of the per-scale single-scale loss (match + energy + suppress); None
        #   gives 0.5 + 0.25·b (smoother scales get larger weights). The final method uses None.
        eta: Optional[List[float]] = None,
        lambda_energy: float = 0.5,        # weight of the Dirichlet-energy distribution term (c); final method 0.5
        num_moments: int = 8,              # contracted-moment order K; final method 8 (= num_scales, all reused from the pyramid)
        lambda_moment: float = 0.5,        # weight of the contracted-moment term (e); final method 0.5
        eps: float = 1e-6,                 # numerical floor for all normalization denominators
        detach_teacher: bool = True,       # cut the gradient of the teacher features at the start of forward; final method True
        pad_mode: str = "replicate",       # boundary padding of the grid Laplacian (replicate = Neumann boundary); final method replicate
        lambda_profile: float = 0.5,       # weight of the profile term (d); final method 8.0
        # per_channel_norm: whether the whole-plane match is normalized per channel. Only read on
        #   the _match_view path with lf_gate=False and block_size=None; not read by the final
        #   method.
        per_channel_norm: bool = False,
        # cosine_resp: the whole-plane match uses the per-sample ‖n_s−n_t‖² = 2−2cosθ (True, O(1)
        #   magnitude) or the mean over all elements (False).
        #   Read under the same condition as per_channel_norm; not read by the final method.
        cosine_resp: bool = True,
        # include_raw_scale: when True, the raw field b=0 is also compared ("compare first, then
        #   diffuse", for ablation). _match_view_v5 requires False. Final method False.
        include_raw_scale: bool = False,
        # legacy_lk_moments: when True, the moment term uses the raw L^k moments xᵀL^k x instead
        #   (for ablation; its noise sensitivity grows exponentially with k).
        #   Final method False.
        legacy_lk_moments: bool = False,
        view_weights: Tuple[float, float, float] = (3.0, 1.0, 1.0),   # (xy, yz, zx) plane weights; final method uses the default
        view_divisor: float = 3.0,         # divisor of the total loss; final method 3.0, i.e. (3·xy + yz + zx)/3
        use_checkpoint: bool = True,       # one recomputation segment per scale, keeping only segment-boundary tensors, to reduce memory of the distillation graph
        # ---- Block-local match ----
        # block_size=(bh,bw): match becomes a per-spatial-block alignment; each block is
        #   normalized independently, 2−2cosθ is computed, and then averaged over blocks, so that
        #   shape mismatch in blocks containing small objects gets the same weight as large ground
        #   blocks (whole-plane normalization lets large areas dominate the loss).
        #   The energy/profile/moment terms are per-channel statistics and are not affected.
        #   None = the whole plane as a single block. _match_view_v5 requires non-None.
        #   Final method (30,30).
        block_size: Optional[Tuple[int, int]] = None,
        # ---- LF gating: division of labor between detail (match where the gate is open) and
        #      suppress (suppression where the gate is closed) ----
        # lf_gate: when True, the teacher gate v ∈ [0,1) is computed on gated planes (selected by
        #   plane_gate) (teacher side, no gradient).
        #   Where v≈1 (the teacher has low-frequency structure) a v-weighted block alignment is
        #   applied; where v≈0 (the teacher is flat) the suppress term suppresses the student's
        #   local variation (L g_s)². What is suppressed is the local variation, not the low-pass
        #   amplitude: a flat teacher only implies L g_t≈0, not g_t≈0, so legitimate constant/DC
        #   components are not suppressed. False: no gate is computed and there is no suppress
        #   term (no-gate ablation).
        #   Final method True.
        lf_gate: bool = False,
        gate_kappa: float = 0.5,           # gate threshold τ = gate_kappa·(mean saliency over the statistics domain); final method 0.5
        lambda_suppress: float = 0.5,      # suppress-term weight; final method 0.25 (0 in the "match only" ablation arm)
        # suppress_scales: scale indices b at which suppress is applied (0-based, corresponding to
        #   g_{b+1}). None = range(2, num_scales), i.e. suppression only on the smoother scales
        #   (folded aliasing falls into the low/mid bands). Final method None, i.e. {2..7}.
        suppress_scales: Optional[List[int]] = None,
        # profile_metric: profile distance. "l1" = total variation 0.5·Σ|p̂−q̂| (first order; the
        #   gradient does not vanish when teacher and student are close);
        #   "js" = JS divergence (second order; the gradient tends to 0 after alignment). Final
        #   method "l1".
        profile_metric: str = "js",
        # moment_norm: normalization axis of the contracted moments (passed to the base class;
        #   see _moment_mse). "joint" = normalize once over the joint (channel × scale) axis,
        #   quotienting out only the per-sample positive gain; "per_scale_channel" = normalize once
        #   per scale along the channel axis (which also quotients out the energy decay along the
        #   scale axis; not the final method). Final method "joint".
        moment_norm: str = "per_scale_channel",
        # suppress_ref: amplitude reference of suppress. "teacher_floor": ref = max(detach E_s, E_t),
        #   where E_t is the teacher band-pass energy at the same scale; homogeneous of degree 2 in
        #   the student amplitude; "self_rms": ref = the student's own E_s; homogeneous of degree 0
        #   in the student amplitude, so it constrains only the proportion and not the amplitude
        #   (available only on the _match_view path, for ablation). The gated path of
        #   _match_view_v5 requires "teacher_floor"; the concrete reference formula is then chosen
        #   by suppress_ref_mode. Final method "teacher_floor".
        suppress_ref: str = "self_rms",
        # ---- Switch group (legal values LF_V5_CHOICES, defaults LF_V5_DEFAULTS) ----
        # If the effective value of any switch is non-default, forward takes _match_view_v5,
        # otherwise _match_view.
        # Planes fall into two classes: gated planes (lf_gate=True and plane_gate[k]=True) and
        # ungated planes (the remaining planes; all three when lf_gate=False). Gate-related
        # switches (LF_V5_GATE_SWITCHES) act only on gated planes; block_pad / block_weight /
        # ungated_domain (LF_V5_UNGATED_ACTIVE) do not read the teacher gate and do not compute the
        # gate or the suppress term, and also take effect when lf_gate=False.
        # block_pad: handling when the plane size is not divisible by the block length.
        #   "replicate": replicate padding (border rows/columns are copied several times and are
        #   counted repeatedly in the block norm and the in-block alignment);
        #   "fit": per-axis block length = min(block length, plane size) (yz → (30,16),
        #   zx → (16,30)); any remaining non-divisible remainder is handled as "mask";
        #   "mask": padded cells are filled with 0 and have weight 0, entering neither the block
        #   norm nor the loss.
        #   It only changes the blocking and does not read the teacher, so it also takes effect
        #   when lf_gate=False. Final method "fit".
        block_pad: str = "replicate",
        # lambda_match_gated: weight of the blocked match on gated planes. Final method 1.0; 0 in
        #   the "gated, suppress only" ablation arm (only the match on gated planes is removed;
        #   the gate, suppress and the in-support match on ungated planes are kept).
        #   Defined only on the gated path of _match_view_v5; in other cases a value other than 1
        #   raises an error at construction.
        lambda_match_gated: float = 1.0,
        # gate_domain: statistics domain on gated planes. "support": τ, the per-channel teacher
        #   band-pass energy e_t, the student E_s, the suppress reference and weights, the block
        #   norms and the block average are all computed only over the plane projection R of the
        #   teacher support (with gate_source='prob', R additionally excludes isolated cells that
        #   have no support neighbor); outside R the gate and suppress weights are 0, and blocks
        #   without R cells do not enter the block average. "plane": the whole plane.
        #   "support" requires plane_support. Final method "support".
        gate_domain: str = "plane",
        # gate_source: source of the gate saliency.
        #   "prob": the multi-scale band-pass energy of the teacher plane class-probability map p
        #   (the teacher voxel softmax averaged over support voxels along the collapsed axis),
        #   e_b = Σ_k (L_Ω D_Ω^b p)_k², where L_Ω/D_Ω are the support-induced 4-neighbor Neumann
        #   subgraph operators; summed over classes, all channels share one gate, which opens
        #   where the teacher class probabilities vary spatially (semantic boundaries). Computed
        #   only on the teacher side, without gradient.
        #   With gate_domain='support', isolated cells without support neighbors are additionally
        #   removed (under the subgraph operator their band-pass energy is identically 0).
        #   "feature": the per-channel teacher feature band-pass energy (L g_t)².
        #   In both cases match/suppress act on the TPV features. "prob" requires plane_prob and
        #   plane_support.
        #   Final method "prob".
        gate_source: str = "feature",
        # profile_domain: weighting of the profile energy e_b on gated planes. "open": weighted by
        #   the gate actually used at that scale (Σ v·g_b², same weights for teacher and student,
        #   the gate does not back-propagate); closed-gate regions do not enter the profile term,
        #   so the profile term cannot inject structure into the student there and counteract
        #   suppress;
        #   "plane": the unweighted whole-plane Σ g_b². The profile weighting of ungated planes is
        #   determined by ungated_domain; without a gate it is a no-op.
        #   Final method "open".
        profile_domain: str = "plane",
        # plane_gate (xy, yz, zx): whether each plane is gated. Planes set to False compute no
        #   gate and have no suppress term; match uses the ungated block alignment (statistics
        #   domain determined by ungated_domain). Each yz/zx cell is an average over a whole ρ/φ
        #   line and has no definable local semantics; zero-target suppression on them would
        #   flatten the features as a whole, so only xy is gated. Final method
        #   (True, False, False).
        plane_gate: Tuple[bool, bool, bool] = (True, True, True),
        # suppress_target: target of suppress where the gate is closed. "zero": push the student
        #   band-pass energy (L g_s)² towards 0; "teacher": penalize only the part of the student
        #   exceeding the teacher, relu((L g_s)² − (L g_t)²)/ref, for which the teacher itself is
        #   the optimum. Final method "zero".
        suppress_target: str = "zero",
        # suppress_ref_mode: form of the suppress reference ref (let A = weighted mean of the
        #   student band-pass energy where the gate is closed).
        #   "detach_floor": ref = max(detach E_s, E_t). When E_s > E_t the forward is roughly
        #   homogeneous of degree 0 in the student amplitude, yet the backward only gives ∇A/E_s,
        #   which is not the derivative of the forward.
        #   "log_tail": the reference depends only on the teacher, T = max(E_t, ref_floor_rel·P_t)
        #   (P_t = teacher low-pass power at the same scale, mean g_t²),
        #   u = Σw·q/(T+eps)/(Σw+eps), and the loss is f(u) = u (u≤1) | 1+log u (u>1). f is C¹
        #   and monotone, and the gradient = ∇A/max(A,T) is everywhere the true derivative of
        #   the forward. It must be written as a branch selection with torch.where
        #   (u − relu(u−1) + log1p(relu(u−1)) suffers 1−1 cancellation in the fp32 backward, which
        #   makes the gradient inaccurate at large u).
        #   The loss value has no fixed upper bound; the dynamic loss balancer reads only gradient
        #   norms and is not affected.
        #   When lf_gate=False there is no suppress term and this switch is a no-op. Final method
        #   "log_tail".
        suppress_ref_mode: str = "detach_floor",
        ref_floor_rel: float = 1e-4,       # floor coefficient of the teacher reference in log_tail (T does not degenerate to 0 when the teacher is flat); final method 1e-4
        # block_weight: weighting of the average over blocks. "equal": participating blocks get
        #   equal weight; "support": blocks are weighted by cell count -- when the plane has a
        #   statistics domain (R of a gated plane with gate_domain='support', or R_u of an ungated
        #   plane with ungated_domain='support') the in-domain cells are counted, the same region
        #   as used for alignment; otherwise the support cells m are counted. It does not read the
        #   teacher gate and also takes effect when lf_gate=False; "support" requires
        #   plane_support. Not the final method (the final method uses "equal").
        block_weight: str = "equal",
        # ungated_domain: statistics domain of ungated planes (plane_gate[k]=False; all three
        #   planes when lf_gate=False).
        #   "plane": whole-plane ungated block alignment + unweighted profile Σ g_b².
        #   "support": R_u = m·1[deg>0] (deg = number of 4-neighbors within the support;
        #   independent of gate_source, isolated cells are always removed).
        #   Match weight v_det = R_u, block statistics domain Mb = R_u (blocks without R_u cells do
        #   not enter the block average), profile energy weighted by R_u (Σ R_u·g_b², same weights
        #   for teacher and student). It is only a spatial restriction to the teacher support; it
        #   does not read the teacher gate and does not compute the gate or the suppress term,
        #   so it also takes effect when lf_gate=False: both the final method and the no-gate
        #   ablation arm use 'support', and the two run the same code on yz/zx, differing only in
        #   the gate and suppress term on xy. The energy term and contracted moments are not
        #   affected. Requires plane_support.
        #   Final method "support".
        ungated_domain: str = "plane",
    ):
        # Base class: checks the range of alpha, registers the 2D/3D Laplacian kernels, sets the
        # segment-recomputation switch and the contracted-moment normalization axis.
        super().__init__(alpha=alpha, pad_mode=pad_mode, eps=eps,
                         use_checkpoint=use_checkpoint, moment_norm=moment_norm)
        assert suppress_ref in ("teacher_floor", "self_rms"), suppress_ref
        self.suppress_ref = suppress_ref
        # Normalize block_size into an int tuple; None means the whole plane as a single block
        self.block_size = tuple(int(v) for v in block_size) if block_size else None
        self.lf_gate = bool(lf_gate)
        self.gate_kappa = float(gate_kappa)
        self.lambda_suppress = float(lambda_suppress)
        # suppress scales default to b = 2..S−1; stored as a frozenset for the per-scale check
        # b in self._suppress_scales
        if suppress_scales is None:
            suppress_scales = list(range(2, int(num_scales)))
        self._suppress_scales = frozenset(int(b) for b in suppress_scales)
        assert profile_metric in ("js", "l1"), profile_metric
        self.profile_metric = profile_metric
        # Remaining hyperparameters are stored after type normalization
        self.num_scales = int(num_scales)
        self.lambda_energy = float(lambda_energy)
        self.num_moments = int(num_moments)
        self.lambda_moment = float(lambda_moment)
        self.detach_teacher = bool(detach_teacher)
        self.lambda_profile = float(lambda_profile)
        self.per_channel_norm = bool(per_channel_norm)
        self.cosine_resp = bool(cosine_resp)
        self.include_raw_scale = bool(include_raw_scale)
        self.legacy_lk_moments = bool(legacy_lk_moments)
        self.view_weights = tuple(float(v) for v in view_weights)
        self.view_divisor = float(view_divisor)

        # Scale weights η_b: default 0.5 + 0.25·b. The length must equal num_scales, otherwise the
        # per-scale weighting would be silently misaligned.
        if eta is None:
            eta = [0.5 + 0.25 * b for b in range(self.num_scales)]
        assert len(eta) == self.num_scales, (
            f"len(eta)={len(eta)} must equal num_scales={self.num_scales}"
        )
        # Registered as a buffer: moves with .to(device) and is saved in the state_dict
        self.register_buffer("eta", torch.tensor(eta, dtype=torch.float32))

        # ---- Switch value checks: always raise ValueError at construction (not assert:
        #      python -O strips asserts) ----
        req = dict(block_pad=block_pad, gate_domain=gate_domain, gate_source=gate_source,
                   profile_domain=profile_domain, suppress_target=suppress_target,
                   suppress_ref_mode=suppress_ref_mode, block_weight=block_weight,
                   ungated_domain=ungated_domain)
        for k_, v_ in req.items():
            if v_ not in LF_V5_CHOICES[k_]:
                raise ValueError(f"{k_}={v_!r} not in {LF_V5_CHOICES[k_]}")
        # plane_gate must have 3 elements (xy, yz, zx) and accept only true bool / 0 / 1: if the
        # config mistakenly writes the string "False", bool("False") is True and would silently
        # open the gate.
        pg = tuple(plane_gate) if isinstance(plane_gate, (list, tuple)) else ()
        if len(pg) != 3 or not all(isinstance(x, (bool, int)) and x in (0, 1) for x in pg):
            raise ValueError(f"plane_gate={plane_gate!r}: need 3 bools (xy, yz, zx)")
        req["plane_gate"] = tuple(bool(x) for x in pg)
        # ref_floor_rel must be finite and ≥ 0
        rho = float(ref_floor_rel)
        if not (math.isfinite(rho) and rho >= 0.0):
            raise ValueError(f"ref_floor_rel={ref_floor_rel}: need a finite value >= 0")
        self.ref_floor_rel = rho               # only read when suppress_ref_mode='log_tail'
        # lambda_match_gated must be finite and ≥ 0
        lmg = float(lambda_match_gated)
        if not (math.isfinite(lmg) and lmg >= 0.0):
            raise ValueError(f"lambda_match_gated={lambda_match_gated}: need a finite value >= 0")
        self.lambda_match_gated = lmg
        # Switch values requested by the config (recorded as is, written to the construction log)
        self.lf_v5_requested = dict(req)
        # Effective values eff. When lf_gate=False (no-gate ablation: ungated match, no suppress
        # term), the gate-related switches (LF_V5_GATE_SWITCHES) are all reset to their defaults
        # and the gate and suppress term are not computed at all; the names of the reset switches
        # are recorded in lf_v5_ignored. Only LF_V5_UNGATED_ACTIVE (block_pad / block_weight /
        # ungated_domain) still take effect on the ungated path, and none of the three reads the
        # teacher gate. profile_domain='open' has no gate to weight with and is likewise reset to
        # 'plane'.
        # Gate-related switches act only on gated planes, so resetting them does not change
        # ungated planes: yz/zx of the final method and the same planes of the no-gate ablation
        # arm run the same code (looking only at ungated_domain / block_pad / block_weight).
        eff = dict(req)
        self.lf_v5_ignored: List[str] = []
        if not self.lf_gate:
            for k_ in LF_V5_GATE_SWITCHES:
                if eff[k_] != LF_V5_DEFAULTS[k_]:
                    self.lf_v5_ignored.append(k_)
                    eff[k_] = LF_V5_DEFAULTS[k_]
        # Effective values are written to attributes of the same name (self.block_pad /
        # self.gate_domain / self.plane_gate, etc.)
        for k_, v_ in eff.items():
            setattr(self, k_, v_)
        # All effective switches at their defaults → forward takes _match_view (does not read
        # plane_support / plane_prob); otherwise it takes _match_view_v5. True for the final
        # method.
        self._lf_v5 = any(eff[k_] != LF_V5_DEFAULTS[k_] for k_ in LF_V5_DEFAULTS)
        if self._lf_v5:
            # _match_view_v5 is implemented only for the form "diffuse first, then compare +
            # blocked alignment + teacher_floor reference"; other combinations raise an error
            # immediately instead of silently running an unimplemented path.
            if self.include_raw_scale:
                raise ValueError("LF v5 switches require include_raw_scale=False (diffuse first, then compare)")
            if self.block_size is None:
                raise ValueError("LF v5 switches are only implemented on the block-local path and require block_size (main config (30,30))")
            if self.lf_gate and self.suppress_ref != "teacher_floor":
                raise ValueError("LF v5 switches require suppress_ref='teacher_floor' (main config); "
                                 f"the current {self.suppress_ref!r} is a legacy ablation reference")
        # lambda_match_gated ≠ 1 is meaningful only with _match_view_v5 and at least one gated
        # plane; otherwise it would be silently ignored, hence the error
        if self.lambda_match_gated != 1.0 and not (self._lf_v5 and self.lf_gate and any(self.plane_gate)):
            raise ValueError("lambda_match_gated is only defined on the v5 gated path (requires lf_gate=True, at least one "
                             "gated plane, and an active v5 switch); otherwise it would be silently ignored")
        # After successful construction, emit one log line recording the LF variant actually run
        # (only reads attributes already set, creates no tensors, changes no state; rank 0 only).
        # The training log uses it to verify which LF configuration is actually running (a
        # misspelled key in the upper-level config would be swallowed by **kwargs and silently run
        # the default LF).
        _lf_v5_emit(self.lf_v5_log_line())

    # ---- Construction log --------------------------------------------------------------------
    def lf_v5_summary(self) -> dict:
        """Return a description of the LF variant actually run (a dict, written to the construction
        log). Only reads attributes already set; the decision logic is the same as forward's branch
        selection. Top-level fields: path ("v5" = _match_view_v5, "production" = _match_view),
        lf_gate, lf_v5_requested (values requested by the config), lf_v5_ignored (gate-related
        switches reset to their defaults when lf_gate=False), planes, needs_plane_inputs,
        block_size, suppress_ref, ref_floor_rel, lambda_suppress, suppress_scales;
        lambda_match_gated is written only when it takes a non-default value. planes, per plane
        (xy / yz / zx):
        status   gated (lf_gate=True and plane_gate[k]=True) | ungated;
        gate_source the gate source actually used on a gated plane (only feature on the
                 _match_view path); None on ungated planes;
        region   statistics domain of the block alignment / gate / suppress: plane (whole plane) |
                 support R (gated plane with gate_domain='support'; R = m·1[deg>0] when
                 gate_source='prob', otherwise R = m) | support R_u (ungated plane with
                 ungated_domain='support', R_u = m·1[deg>0]);
        profile  profile energy weighting: plane (unweighted Σ g_b²) | gate v (gated plane with
                 profile_domain='open') | support R_u;
        suppress on | off; when on, suppress_target and suppress_ref_mode are the target and
                 reference actually run (on the _match_view path: suppress_ref='teacher_floor' is
                 recorded as detach_floor, 'self_rms' as self_rms);
        block_pad / block_weight effective values (None when block_size=None, i.e. no blocking).
        Final method: path=v5; xy is gated / prob / support R / gate v / suppress on (zero,
        log_tail); yz and zx are ungated / support R_u / suppress off; block_pad=fit,
        block_weight=equal."""
        v5 = bool(self._lf_v5)                         # whether _match_view_v5 is taken
        blocks = self.block_size is not None           # whether blocked alignment is used
        # A gated plane has a suppress term only if at least one scale lies in suppress_scales
        has_sup = any(b in self._suppress_scales for b in range(self.num_scales))
        planes = {}
        for k, name in enumerate(("xy", "yz", "zx")):
            gated = bool(self.lf_gate and self.plane_gate[k])
            if gated:
                # Gated plane: statistics domain from gate_domain, profile weighting from
                # profile_domain (both take effect only on the _match_view_v5 path)
                region = "support R" if (v5 and self.gate_domain == "support") else "plane"
                profile = "gate v" if (v5 and self.profile_domain == "open") else "plane"
            else:
                # Ungated plane: both the statistics domain and the profile weighting depend only
                # on ungated_domain
                region = "support R_u" if (v5 and self.ungated_domain == "support") else "plane"
                profile = region
            sup_on = gated and has_sup                 # suppress exists only on gated planes
            # Reference form actually used: None without suppress; on the _match_view_v5 path it
            # follows suppress_ref_mode; otherwise it follows suppress_ref (teacher_floor is the
            # max(detach E_s, E_t) formula, recorded as detach_floor)
            if not sup_on:
                ref_mode = None
            elif v5 and self.suppress_ref_mode == "log_tail":
                ref_mode = "log_tail"
            else:
                ref_mode = "detach_floor" if self.suppress_ref == "teacher_floor" else "self_rms"
            planes[name] = dict(
                status="gated" if gated else "ungated",
                gate_source=(self.gate_source if v5 else "feature") if gated else None,
                region=region,
                profile=profile,
                suppress="on" if sup_on else "off",
                suppress_target=(self.suppress_target if v5 else "zero") if sup_on else None,
                suppress_ref_mode=ref_mode,
                block_pad=self.block_pad if blocks else None,
                block_weight=self.block_weight if blocks else None,
            )
        # Convert tuples (plane_gate) in the requested values into lists for JSON output
        req = {k_: (list(v_) if isinstance(v_, tuple) else v_) for k_, v_ in self.lf_v5_requested.items()}
        out = dict(
            path="v5" if v5 else "production",
            lf_gate=bool(self.lf_gate),
            lf_v5_requested=req,
            lf_v5_ignored=list(self.lf_v5_ignored),
            planes=planes,
            needs_plane_inputs=bool(self.needs_plane_inputs),
            block_size=list(self.block_size) if blocks else None,
            suppress_ref=self.suppress_ref,
            ref_floor_rel=self.ref_floor_rel,
            lambda_suppress=self.lambda_suppress,
            suppress_scales=sorted(self._suppress_scales),
        )
        if self.lambda_match_gated != 1.0:   # written only for non-default values; the log line of the default config does not contain this field
            out["lambda_match_gated"] = self.lambda_match_gated
        return out

    def lf_v5_log_line(self) -> str:
        """Construction log line: LF_V5_LOG_TAG + space + single-line JSON (sort_keys; the tag can
        be located with grep).
        Parsing: json.loads(line.split(LF_V5_LOG_TAG + ' ', 1)[1])."""
        import json
        return LF_V5_LOG_TAG + " " + json.dumps(self.lf_v5_summary(), sort_keys=True, separators=(", ", ": "))

    # ---- Extra inputs: teacher support plane_support and teacher plane class probabilities plane_prob --
    # Whether an input is needed is decided by "the planes that actually read it": gate-related
    # switches read inputs only when a gated plane exists, and ungated_domain reads them only when
    # an ungated plane exists. For example, with lf_gate=True and plane_gate all True,
    # ungated_domain requires no input.
    # Whether a gated plane exists (lf_gate=True and at least one plane_gate True). True for the
    # final method (xy).
    @property
    def _any_gated_plane(self) -> bool:
        return bool(self.lf_gate and any(self.plane_gate))

    # Whether an ungated plane exists (lf_gate=False, or plane_gate not all True). True for the
    # final method (yz, zx).
    @property
    def _any_ungated_plane(self) -> bool:
        return bool((not self.lf_gate) or not all(self.plane_gate))

    @property
    def needs_plane_support(self) -> bool:
        """Whether plane_support (the tri-plane projection m of the teacher support) is needed. Can
        only be True on the _match_view_v5 path: gated plane with gate_domain='support' or
        gate_source='prob', block_weight='support', or ungated plane with
        ungated_domain='support'. Always False when all switches are at their defaults; True for
        the final method."""
        return bool(self._lf_v5 and (
            (self._any_gated_plane and (self.gate_domain == "support" or self.gate_source == "prob"))
            or self.block_weight == "support"
            or (self._any_ungated_plane and self.ungated_domain == "support")))

    @property
    def needs_plane_prob(self) -> bool:
        """Whether plane_prob (teacher plane class probabilities) is needed: True only on the
        _match_view_v5 path when a gated plane exists and gate_source='prob'. Always False when all
        switches are at their defaults; True for the final method."""
        return bool(self._lf_v5 and self._any_gated_plane and self.gate_source == "prob")

    # Extra inputs are needed if either of the two is True. The LF distillation call site in
    # pointtpv_seg.py uses this to decide whether to build plane_support / plane_prob from the
    # teacher distillation field with plane_inputs_from_field and pass them to forward.
    @property
    def needs_plane_inputs(self) -> bool:
        return self.needs_plane_support or self.needs_plane_prob

    def plane_input_switches(self) -> str:
        """List the effective switches that cause extra inputs to be needed (comma-separated, used
        in error messages); the conditions are the same as in needs_plane_*.
        Returns an empty string on the _match_view path."""
        if not self._lf_v5:
            return ""
        out = []
        if self._any_gated_plane:
            if self.gate_domain == "support":
                out.append("gate_domain='support'")
            if self.gate_source == "prob":
                out.append("gate_source='prob'")
        if self.block_weight == "support":
            out.append("block_weight='support'")
        if self._any_ungated_plane and self.ungated_domain == "support":
            out.append("ungated_domain='support'")
        return ", ".join(out)

    @staticmethod
    @torch.no_grad()
    def plane_inputs_from_field(logits_vox_t: torch.Tensor):
        """Build the extra forward inputs from the teacher distillation field (no network forward,
        no gradient; only the field already obtained by the teacher's no_grad forward is used).
        logits_vox_t: teacher voxel distillation field [B,K,240,180,16] (t_out[3] in
        pointtpv_seg.py); it is scattered onto an all-zero tensor, so it is exactly 0 outside the
        teacher support Ω_T = unique(floor(gv_t/2)) (gv_t are the voxel coordinates of the
        distillation points).
        Returns (plane_support, plane_prob), each a 3-element list [xy, yz, zx]:
          plane_support [B,1,H,W] bool = plane projection of Ω_T (same criterion as HF's
                        _support_geometry: a voxel with any non-zero channel belongs to the
                        support);
          plane_prob    [B,K,H,W] float32 = teacher voxel softmax averaged over support voxels
                        along the collapsed axis (0 outside the support).
        Planes: xy [240,180] (collapse z), yz [180,16] (collapse ρ), zx [16,240] (collapse φ,
        then transpose), with the same axis order as the TPV planes.
        Called by pointtpv_seg.py when needs_plane_inputs is True (in the final method, on every
        LF forward)."""
        x = logits_vox_t.detach()
        m = x.ne(0).any(dim=1, keepdim=True)                 # voxel support Ω_T [B,1,240,180,16]
        mf = m.to(torch.float32)
        p = torch.softmax(x.float(), dim=1) * mf             # class probabilities, set to 0 outside the support [B,K,240,180,16]
        sup, prob = [], []
        # (collapsed axis, transpose?): dim=4 collapses z → xy; dim=2 collapses ρ → yz; dim=3
        # collapses φ → [240,16], transposed into zx [16,240]
        for dim, transpose in ((4, False), (2, False), (3, True)):
            cnt = mf.sum(dim=dim)                            # number of support voxels on each line
            ps = p.sum(dim=dim) / cnt.clamp_min(1.0)         # mean probability over support voxels; 0 where there is no support
            s = cnt > 0                                      # plane support: at least one support voxel on the line
            if transpose:
                ps = ps.transpose(-1, -2)
                s = s.transpose(-1, -2)
            sup.append(s.contiguous())
            prob.append(ps.contiguous())
        return sup, prob

    def _lf_v5_inputs(self, ref: List[torch.Tensor], plane_support, plane_prob):
        """Validate and prepare the extra forward inputs (teacher-side constants, detached).
        ref = list of student tri-plane features [B,C,H,W], used as the shape/device/dtype
        reference. Returns (ms, ps): per-plane support masks [B,1,H,W] (0/1, feature dtype) and
        class probabilities [B,K,H,W] (feature dtype); positions of unneeded inputs are None.
        Missing inputs or shape mismatches always raise an error; there is no silent fallback.
        Called by forward on the _match_view_v5 path."""
        n = len(ref)
        ms: List[Optional[torch.Tensor]] = [None] * n
        ps: List[Optional[torch.Tensor]] = [None] * n
        # Class probabilities: detach, move to the feature's device/dtype, check batch and plane size
        if self.needs_plane_prob:
            if plane_prob is None or len(plane_prob) != n:
                raise ValueError(f"LF v5: {self.plane_input_switches()} requires plane_prob (3 x [B,K,H,W], "
                                 "see plane_inputs_from_field)")
            for i, (p, s) in enumerate(zip(plane_prob, ref)):
                p = p.detach().to(device=s.device, dtype=s.dtype)
                if p.dim() != 4 or p.shape[0] != s.shape[0] or p.shape[-2:] != s.shape[-2:]:
                    raise ValueError(f"LF v5: plane_prob[{i}] shape {tuple(p.shape)} vs plane {tuple(s.shape)}")
                ps[i] = p
        # Support masks: accept [B,H,W] or [B,1,H,W]; after the size check, convert into 0/1 masks
        # of the feature dtype
        if self.needs_plane_support:
            if plane_support is None or len(plane_support) != n:
                raise ValueError(f"LF v5: {self.plane_input_switches()} requires plane_support (3 x [B,1,H,W], "
                                 "see plane_inputs_from_field)")
            for i, (m, s) in enumerate(zip(plane_support, ref)):
                m = m.detach()
                if m.dim() == 3:
                    m = m[:, None]                           # [B,H,W] → [B,1,H,W]
                if m.dim() != 4 or m.shape[0] != s.shape[0] or m.shape[1] != 1 or m.shape[-2:] != s.shape[-2:]:
                    raise ValueError(f"LF v5: plane_support[{i}] shape {tuple(m.shape)} vs plane {tuple(s.shape)}")
                ms[i] = (m != 0).to(device=s.device, dtype=s.dtype)
        return ms, ps

    # ---- Support-subgraph operator and probability gate ----------------------------------------
    @staticmethod
    def _nbr_sum_2d(x: torch.Tensor) -> torch.Tensor:
        """Sum of the 4 neighbors [..., H, W] → [..., H, W]; cells outside the grid count as 0
        (F.pad pads with 0 by default). Shifted slices are added instead of using a convolution,
        which is exact addition without convolution rounding: applied to a 0/1 support mask it
        gives exact integer neighbor counts."""
        p = F.pad(x, (1, 1, 1, 1))
        # up neighbor + down neighbor + left neighbor + right neighbor
        return p[..., :-2, 1:-1] + p[..., 2:, 1:-1] + p[..., 1:-1, :-2] + p[..., 1:-1, 2:]

    @torch.no_grad()
    def _lf_prob_saliency(self, p: torch.Tensor, m: torch.Tensor, deg: torch.Tensor) -> List[torch.Tensor]:
        """Saliency of the probability gate (computed only on the teacher side, without gradient).
        Input: p [B,K,H,W] teacher plane class probabilities; m [B,1,H,W] support mask (0/1);
        deg [B,1,H,W] actual number of 4-neighbors within the support.
        Output: a list of length S = num_scales whose b-th item (b=1..S; "diffuse first, then
        compare", as for the feature pyramid) is e_b = Σ_k (L_Ω g^p_b)_k² [B,1,H,W] with
        g^p_b = D_Ω^b (p·m), i.e. the multi-scale band-pass energy of the class-probability map,
        summed over classes; it is large where the teacher class probabilities vary spatially
        (semantic boundaries).
        L_Ω x = m·(deg·x − Σ_neighbors x)/4 is the Neumann Laplacian of the support-induced
        subgraph (the 2D version of the 3D subgraph operator of the HF branch), D_Ω = I − αL_Ω;
        deg ≤ 4, so the spectrum lies in [0,2] and D_Ω is still a monotone low-pass for α ≤ 1/2.
        When the support is the whole plane, L_Ω equals the grid operator with replicate boundary.
        When the input is 0 outside the support, L_Ω outputs exactly 0 on isolated cells with no
        support neighbor.
        Called once by _match_view_v5 on a gated plane when gate_source='prob'; the results are
        passed per scale to _lf_gate_v5."""
        p = p.detach()
        g = p * m                                            # 0 outside the support; L_Ω requires the input to be 0 outside the support

        # Subgraph Laplacian: its output is 0 outside the support, so g stays 0 outside the support
        # after diffusion, and Σ_neighbors only accumulates neighbors inside the support
        def lap(x):
            return m * (deg * x - self._nbr_sum_2d(x)) / 4.0

        out = []
        for _ in range(self.num_scales):
            g = g - self.alpha * lap(g)                      # g ← D_Ω g (one low-pass step)
            Lg = lap(g)                                      # band-pass response L_Ω D_Ω^b (p·m)
            out.append((Lg * Lg).sum(dim=1, keepdim=True))   # sum over classes → [B,1,H,W]
        return out

    def _lf_gate_v5(self, gt: torch.Tensor, R: Optional[torch.Tensor], salp: Optional[torch.Tensor]):
        """Compute the gate actually used at this scale. Inputs: gt = teacher low-pass response at
        this scale g_t = D^b t [B,C,H,W]; R = statistics domain [B,1,H,W] (0/1; None means the
        whole plane); salp = probability-gate saliency e_b at this scale [B,1,H,W] (given by
        _lf_prob_saliency when gate_source='prob', otherwise None).
        Gate v = e/(e+τ), τ = gate_kappa·mean_R(e) + eps:
          probability source: e = salp, v is [B,1,H,W], shared by all channels;
          feature source: e = (L g_t)² per channel [B,C,H,W], τ = gate_kappa·e_t + eps.
        Returns (v, e_t, Lt): v is 0 outside the statistics domain; e_t [B,C,1,1] = mean of the
        per-channel teacher band-pass energy (L g_t)² over the statistics domain (the teacher
        reference of suppress); Lt = L g_t (detached, used when suppress_target='teacher').
        L here is the grid Laplacian acting on the TPV features (_apply_laplacian). The gate uses
        only the teacher and does not back-propagate gradients.
        With R=None and the feature source, the computation is the same as _lf_gate on the
        _match_view path. Called per scale by _scale_terms_v5 on gated planes."""
        gt_d = gt.detach()                                   # the gate uses only the teacher, no gradient
        Lt = self._apply_laplacian(gt_d)                     # teacher band-pass response L g_t
        sal = Lt ** 2                                        # per-point per-channel band-pass energy [B,C,H,W]
        sdim = self._spatial_dims(sal)
        # e_t: mean of the per-channel teacher band-pass energy over the statistics domain
        # [B,C,1,1]; clamp_min(1) prevents division by 0 when the domain is empty
        if R is None:
            e_t = sal.mean(dim=sdim, keepdim=True)
        else:
            e_t = (sal * R).sum(dim=sdim, keepdim=True) / R.sum(dim=sdim, keepdim=True).clamp_min(1.0)
        if self.gate_source == "prob":
            # Probability source: saliency e [B,1,H,W], τ = κ·(mean of e over the statistics
            # domain) + eps, [B,1,1,1]
            e = salp.to(sal.dtype)
            if R is None:
                tau = self.gate_kappa * e.mean(dim=sdim, keepdim=True) + self.eps
            else:
                tau = self.gate_kappa * (e * R).sum(dim=sdim, keepdim=True) / R.sum(
                    dim=sdim, keepdim=True).clamp_min(1.0) + self.eps
        else:
            # Feature source: per-channel saliency, τ = κ·e_t + eps, [B,C,1,1]
            e = sal
            tau = self.gate_kappa * e_t + self.eps
        v = e / (e + tau)                                    # v ∈ [0,1); eps guarantees τ > 0, so v = 0 when e≡0
        if R is not None:
            v = v * R                                        # the gate is 0 outside the statistics domain
        return v, e_t, Lt

    def _block_local_resp_v5(self, gs: torch.Tensor, gt: torch.Tensor, v: Optional[torch.Tensor],
                             Mb: Optional[torch.Tensor]) -> torch.Tensor:
        """Block-local (weighted) shape alignment, i.e. the LF match term. Used by the final method;
        called by _scale_terms_v5 at every scale.

        What is compared is this scale's multi-scale diffusion (low-pass) response g_b = D^b x, not
        the band-pass L g_b.
        gs/gt: student/teacher g_b, [B,C,H,W]. The plane is cut into non-overlapping (bh,bw)
        blocks; each block is (weighted) Frobenius-normalized over (C,bh,bw) and the shapes are
        compared, with per-block loss Σ v·(ns−nt)² = 2−2cos_v ∈ [0,4].
        v: per-cell weights within a block, with three possible values.
          * Gated plane (xy in the final method): the gate v actually used at this scale
            (probability source [B,1,H,W], feature source [B,C,H,W]; 0 outside the statistics
            domain), i.e. shapes are compared only where the gate is open;
          * ungated plane with ungated_domain='support' (yz/zx in the final method): the
            statistics domain R_u [B,1,H,W], i.e. comparison only on the teacher support (with
            isolated cells removed);
          * None: whole plane, unweighted (not the final method).
        Mb: cell mask [B,1,H,W] that determines the block weights. With None all blocks have equal
          weight (not the final method); otherwise only blocks containing Mb cells enter the
          average, and with block_weight='support' they are further weighted by the number of Mb
          cells in the block (the final method uses block_weight='equal').
        block_pad: handling when the plane size is not divisible by the block length.
          * 'fit' (final method): per-axis block length = min(block length, plane size); the
            16-wide z axis of yz/zx directly uses 16-wide blocks without replicate padding; any
            remaining non-divisible remainder is handled as 'mask';
          * 'mask': padded cells are filled with 0 and have weight 0, entering neither the block
            norm nor the loss (with v=None, the "is a real cell" indicator is used as the weight;
            it is independent of the teacher and is not a gate);
          * 'replicate' (not the final method: default): pads by replicating the border according
            to self.pad_mode; padded cells take part in the block norm and the alignment.
        Returns: scalar loss."""
        bh, bw = self.block_size
        B, C, H, W = gs.shape
        mode = self.block_pad
        # 'fit': block length does not exceed the plane size; if it is still not divisible
        # afterwards, the remainder is padded in 'mask' fashion
        if mode == "fit":
            bh, bw = min(bh, H), min(bw, W)
            mode = "mask"
        # Number of cells needed on each axis to pad to an integer multiple of the block length (0
        # when divisible). With block_size=(30,30), the three planes of the final method,
        # xy[240,180], yz[180,16], zx[16,240], are all exactly divisible under 'fit', so no padding
        # occurs
        ph = (bh - H % bh) % bh
        pw = (bw - W % bw) % bw
        if ph or pw:
            if mode == "replicate":
                # Replicate-border padding (consistent with the Laplacian's boundary handling); the
                # weights v are padded the same way
                gs = F.pad(gs, (0, pw, 0, ph), mode=self.pad_mode)
                gt = F.pad(gt, (0, pw, 0, ph), mode=self.pad_mode)
                if v is not None:
                    v = F.pad(v, (0, pw, 0, ph), mode=self.pad_mode)
            else:
                # 'mask': features padded with 0 and weights padded with 0, so padded cells
                # contribute to neither the block norm nor the loss
                gs = F.pad(gs, (0, pw, 0, ph))
                gt = F.pad(gt, (0, pw, 0, ph))
                if v is None:
                    # Padded cells must be excluded even when unweighted: use the real-cell
                    # indicator (1 for real cells, 0 for padded cells) as the weight
                    v = F.pad(gs.new_ones(B, 1, H, W), (0, pw, 0, ph))
                else:
                    v = F.pad(v, (0, pw, 0, ph))
            if Mb is not None:
                Mb = F.pad(Mb, (0, pw, 0, ph))            # padded cells are never support cells
        # Blocking: [B,C,Hp,Wp] → [B,C,nh,bh,nw,bw] → [B,nh,nw,C,bh,bw]; in-block dims are (3,4,5)
        Hp, Wp = H + ph, W + pw
        nh, nw = Hp // bh, Wp // bw
        gs = gs.view(B, C, nh, bh, nw, bw).permute(0, 2, 4, 1, 3, 5)
        gt = gt.view(B, C, nh, bh, nw, bw).permute(0, 2, 4, 1, 3, 5)
        dims = (3, 4, 5)
        if v is None:
            # Unweighted: in-block Frobenius normalization, per-block ‖ns−nt‖² = 2−2cosθ; eps
            # prevents division by zero for all-zero blocks
            ns = gs / torch.sqrt((gs * gs).sum(dim=dims, keepdim=True) + self.eps)
            nt = gt / torch.sqrt((gt * gt).sum(dim=dims, keepdim=True) + self.eps)
            lb = ((ns - nt) ** 2).sum(dim=dims)
        else:
            # Weighted: normalize by the v-weighted in-block norm (Σ_block v·ns² = 1); per-block
            # loss Σ_block v·(ns−nt)² = 2−2cos_v ∈ [0,4].
            # Normalized only once and not divided again by the weight sum, so every block is of
            # O(1) magnitude. Single-channel weights [B,1,...] are broadcast along channels
            vb = v.reshape(B, v.shape[1], nh, bh, nw, bw).permute(0, 2, 4, 1, 3, 5)   # [B,nh,nw,C|1,bh,bw]
            ns = gs / torch.sqrt((vb * gs * gs).sum(dim=dims, keepdim=True) + self.eps)
            nt = gt / torch.sqrt((vb * gt * gt).sum(dim=dims, keepdim=True) + self.eps)
            lb = (vb * (ns - nt) ** 2).sum(dim=dims)                  # [B,nh,nw]
        if Mb is None:
            return lb.mean()                                          # no block mask: equal-weight average over all blocks and the batch
        # With a block mask (final method: Mb = the plane's statistics domain): cnt = number of Mb
        # cells in each block. With block_weight='support' blocks are weighted by cell count; with
        # 'equal' (final method) a block counts with equal weight as long as it contains at least
        # one Mb cell, and blocks lying entirely outside the domain get weight 0, so they do not
        # dilute the block average. A weighted mean over blocks is taken per sample
        # (clamp_min(1) prevents division by zero when a sample has no valid block), then averaged
        # over the batch
        cnt = Mb.reshape(B, 1, nh, bh, nw, bw).sum(dim=(1, 3, 5))     # support cells per block [B,nh,nw]
        wblk = cnt if self.block_weight == "support" else (cnt > 0).to(lb.dtype)
        return ((wblk * lb).sum(dim=(1, 2)) / wblk.sum(dim=(1, 2)).clamp_min(1.0)).mean()

    def _lf_suppress_v5(self, gs, gt, v, e_t, Lt, R):
        """Suppression term (suppress) where the gate is closed. Used by the final method on the
        gated plane (xy) at the suppress_scales scales; called by _scale_terms_v5 when
        do_suppress=True.

        Effect: in regions where the teacher's low-frequency structure is flat (gate closed,
        v≈0), the student's band-pass energy (L g_s)² is suppressed, i.e. the spurious local
        structure that the student produces in those regions. What is suppressed is the band-pass
        L g_b, not the low-pass amplitude g_b: a flat teacher only means L g_t≈0, not g_t≈0 (road
        surfaces, walls and object interiors are non-zero constants), so g_s itself must not be
        pushed towards 0.
        Inputs: gs/gt are this scale's g_b [B,C,H,W]; v is the gate actually used at this scale (0
        outside the statistics domain); e_t is the per-channel teacher band-pass energy
        [B,C,1,1] (mean over the statistics domain); Lt = L g_t (already detached); R is the
        statistics domain [B,1,H,W] or None (whole plane).
        Weight w = (1−v)·1_R: cells where the gate is closed and that lie in the statistics
        domain. Element-wise quantity q:
          * suppress_target='zero' (final method): q = (L g_s)², pushing the student band-pass
            energy towards 0 where the gate is closed;
          * 'teacher' (not the final method): q = relu((L g_s)² − (L g_t)²), penalizing only the
            part of the student exceeding the teacher; the teacher is the optimum.
        u = Σ w·q / ref / Σ w (per-sample weighted mean, [B]; eps is added to both denominators
        to prevent division by zero). Reference ref:
          * suppress_ref_mode='log_tail' (final method): ref = T = max(E_t, ref_floor_rel·P_t),
            depending only on the teacher. E_t = channel mean of e_t, P_t = teacher low-pass power
            at the same scale, mean g_t² (element-wise mean over the statistics domain); the
            floor P_t ensures that ref does not degenerate to eps even when the teacher is
            completely flat. The loss is f(u) = u (u≤1) | 1+log u (u>1), C¹ and monotone;
            with A = Σ w·q / Σ w, the gradient = ∇A/max(A,T). Because ref depends only on the
            teacher, what is back-propagated is the true derivative of the forward;
          * 'detach_floor' (not the final method: default): ref = max(detach E_s, E_t), where E_s
            is the student band-pass energy at the same scale. When E_s > E_t, ref varies with the
            student but is treated as a constant in the backward, so the gradient is not the
            derivative of the forward value.
        Returns: a scalar averaged over the batch."""
        sdims = tuple(range(1, gs.dim()))                 # (C,H,W): dims for per-sample sums/means
        r_s = self._apply_laplacian(gs)                   # student band-pass response L g_s [B,C,H,W]
        if self.suppress_ref_mode == "log_tail":
            # Final method: the reference T is computed only from the teacher (detached). E_t:
            # channel mean of the per-channel teacher band-pass energy [B,1,1,1];
            # P_t: teacher low-pass power mean g_t², element-wise mean over the statistics domain
            # [B,1,1,1]
            gt_d = gt.detach()
            E_t = e_t.mean(dim=1, keepdim=True)
            if R is None:
                P_t = (gt_d * gt_d).mean(dim=sdims, keepdim=True)
            else:
                # Denominator = number of in-domain cells × number of channels (R is
                # single-channel); clamp_min(1) prevents division by zero for an empty domain
                P_t = (gt_d * gt_d * R).sum(dim=sdims, keepdim=True) / (
                    R.sum(dim=sdims, keepdim=True) * gt_d.shape[1]).clamp_min(1.0)
            ref = torch.maximum(E_t, self.ref_floor_rel * P_t)   # T = max(E_t, ρ·P_t)
        else:
            # Not the final method, 'detach_floor': the student band-pass energy E_s at the same
            # scale (element-wise mean over the statistics domain), detached and maxed with E_t
            if R is None:
                e_s = (r_s * r_s).mean(dim=sdims, keepdim=True)
            else:
                e_s = (r_s * r_s * R).sum(dim=sdims, keepdim=True) / (
                    R.sum(dim=sdims, keepdim=True) * r_s.shape[1]).clamp_min(1.0)
            ref = torch.maximum(e_s.detach(), e_t.mean(dim=1, keepdim=True))
        # Suppression weight w = (1−v)·1_R: cells where the gate is closed and that lie in the
        # statistics domain; 0 outside the statistics domain
        w_sup = 1.0 - v
        if R is not None:
            w_sup = w_sup * R
        if self.suppress_target == "zero":
            # Final method: q/ref = (L g_s)²/ref, pushing the student band-pass energy towards 0
            # where the gate is closed
            r_n = r_s / torch.sqrt(ref + self.eps)
            num = w_sup * r_n * r_n
        else:
            # Not the final method: penalize only the part of the student band-pass energy
            # exceeding the teacher
            num = w_sup * torch.relu(r_s * r_s - Lt * Lt) / (ref + self.eps)
        # The weight sum is counted at the shape of r_s (single-channel weights are first expanded
        # along channels), so that u is an element-wise weighted mean
        den = (w_sup if w_sup.shape == r_s.shape else w_sup.expand_as(r_s)).sum(dim=sdims)
        u = num.sum(dim=sdims) / (den + self.eps)                    # [B]
        if self.suppress_ref_mode == "log_tail":
            # f(u) = u (u≤1) | 1+log u (u>1). It must be written with torch.where: the algebraically
            # equivalent form u − relu(u−1) + log1p(relu(u−1)) suffers 1 − 1 cancellation in the
            # fp32 backward, so for large u the gradient loses most of its precision or is even
            # exactly 0.
            # The clamp acts only on the branch that is not selected (for u ≤ 1 it prevents log
            # from producing inf/NaN gradients at u=0); the gradient of the selected branch is
            # exactly f'(u).
            u = torch.where(u > 1.0, 1.0 + torch.log(u.clamp(min=1.0)), u)
        return u.mean()

    def _scale_terms_v5(self, gs, gt, k: int, m, R, salp, do_suppress: bool):
        """Single-scale loss terms (the segment function of the segment recomputation _ckpt). Used
        by the final method; called per scale by _match_view_v5.

        gs/gt: student/teacher pyramid at the previous scale g_{b−1} [B,C,H,W]; inside the segment,
        one diffusion step is taken first to obtain this scale's g_b = D^b x, which is then
        compared (diffuse first, then compare, b ≥ 1). k = plane index (0 xy, 1 yz, 2 zx).
        m (support [B,1,H,W]), R (statistics domain) and salp (teacher probability saliency at
        this scale [B,1,H,W]) are all teacher-side constants, passed via closure together with k
        and do_suppress, and are not saved as segment inputs. R: on a gated plane, the gate
        statistics domain for gate_domain='support'; on an ungated plane, R_u for
        ungated_domain='support'; otherwise None (whole plane).
        This scale's loss step = match + λ_energy·energy term (+ λ_suppress·suppress).
        Returns (step, e_s, e_t, g_b^s, g_b^t), where e_s/e_t are the unweighted per-channel
        energies ‖g_b‖² [B,C] (used for the contracted moments, and for the profile when it is
        unweighted); when the profile needs weighting, (pes, pet) = Σ w·g_b² [B,C] are appended."""
        # One diffusion step: g_b = D g_{b−1}, D = I − αL (low-pass)
        gs = self._diffuse(gs)
        gt = self._diffuse(gt)
        gated = bool(self.lf_gate and self.plane_gate[k])   # in the final method only xy (k=0) is gated
        v = e_t = Lt = None
        if gated:                          # ungated planes / lf_gate=False never compute the gate
            # Gate v (0 outside the statistics domain), per-channel teacher band-pass energy e_t,
            # and teacher band-pass Lt = L g_t, all computed only from the detached teacher
            v, e_t, Lt = self._lf_gate_v5(gt, R, salp)
        # Per-cell weights of match: the gate v on gated planes; R_u on ungated planes (whole plane
        # unweighted when R=None)
        v_det = v if gated else R
        # The cells counted for block weights coincide with the match region: if there is a
        # statistics domain, in-domain cells are counted; without one, block_weight='support'
        # counts the support cells m, otherwise None (equal block weights). In the final method all
        # three planes have a statistics domain
        Mb = R if R is not None else (m if self.block_weight == "support" else None)
        loss_resp = self._block_local_resp_v5(gs, gt, v_det, Mb)   # match: shape alignment of in-block-normalized g_b
        # For ablation: match weight on gated planes (lambda_match_gated=0 is the "suppress only"
        # arm); the default 1.0 does not enter this branch, so behavior is the same as without the
        # switch
        if gated and self.lambda_match_gated != 1.0:
            loss_resp = self.lambda_match_gated * loss_resp
        l_sup = None
        if do_suppress:                    # do_suppress implies gated
            l_sup = self._lf_suppress_v5(gs, gt, v, e_t, Lt, R)
        # Energy term: the per-channel Dirichlet energy g_bᵀ L g_b [B,C] is L2-normalized along
        # channels and compared with MSE, so only the relative distribution of energy across
        # channels is compared; whole-plane statistics, unaffected by the gate and the statistics
        # domain
        Es = F.normalize(self._dirichlet_energy(gs), p=2, dim=1)
        Et = F.normalize(self._dirichlet_energy(gt), p=2, dim=1)
        loss_eng = F.mse_loss(Es, Et)
        step = loss_resp + self.lambda_energy * loss_eng
        if do_suppress:
            step = step + self.lambda_suppress * l_sup
        # Unweighted per-channel energy ‖g_b‖² [B,C]
        es = self._channel_energy(gs)
        et = self._channel_energy(gt)
        # Weights w of the profile energy (same for teacher and student; w is a teacher-side
        # constant and does not back-propagate):
        #   gated plane: with profile_domain='open' (final method), the gate v actually used at
        #     this scale, so the profile only counts gate-open regions, preventing the profile term
        #     from raising the student's local variation where the gate is closed and counteracting
        #     suppress;
        #   ungated plane: with ungated_domain='support' (final method), R_u.
        # When w is None, the profile uses the unweighted es/et
        if gated:
            w = v if self.profile_domain == "open" else None
        else:
            w = R if self.ungated_domain == "support" else None
        if w is None:
            return step, es, et, gs, gt
        sp = self._spatial_dims(gs)
        pes = (w * gs * gs).sum(dim=sp)    # Σ w·g_b² [B,C]
        pet = (w * gt * gt).sum(dim=sp)
        return step, es, et, gs, gt, pes, pet

    def _match_view_v5(self, s: torch.Tensor, t: torch.Tensor, k: int,
                       m: Optional[torch.Tensor], p: Optional[torch.Tensor]) -> torch.Tensor:
        """LF distillation loss of a single TPV plane. Used by the final method; forward calls it
        for each of the three planes when _lf_v5=True (some v5 switch is active).

        s/t: student/teacher plane features [B,C,H,W]; k = plane index (0 xy, 1 yz, 2 zx); m/p are
        the plane's plane_support [B,1,H,W] (0/1 float mask) / plane_prob [B,K,H,W] (None when not
        needed).
        Loss = Σ_b η_b·step_b / Σ_b η_b + λ_profile·profile term + λ_moment·contracted-moment
        term, with step_b as in _scale_terms_v5.
        Final method: xy is gated (teacher probability gate, statistics domain R = support with
        isolated cells removed, including suppress); yz/zx are ungated and only matched on R_u."""
        assert s.shape == t.shape and s.dim() == 4, (
            f"Expect matching [B,C,H,W], got {tuple(s.shape)} vs {tuple(t.shape)}"
        )
        gated = bool(self.lf_gate and self.plane_gate[k])
        salp = deg = R = None
        if gated:
            # Gated plane: the gate statistics domain is determined only by gate_domain /
            # gate_source
            if self.gate_source == "prob":
                # Probability gate (final method): number of 4-neighbors within the support deg,
                # and the teacher probability saliency at each scale Σ_k (L_Ω D_Ω^b p)_k² (teacher
                # side only, no gradient; a list of length num_scales, each item [B,1,H,W])
                deg = self._nbr_sum_2d(m) * m               # actual number of 4-neighbors within the support
                salp = self._lf_prob_saliency(p, m, deg)
            if self.gate_domain == "support":
                # With the probability source and the subgraph operator, the residual of isolated
                # cells without support neighbors is identically 0 and would only dilute τ and the
                # suppression denominator, so they are removed; with the feature source the
                # statistics domain is just the support m
                R = m * (deg > 0).to(m.dtype) if deg is not None else m
        elif self.ungated_domain == "support":
            # Ungated plane (yz/zx in the final method): statistics domain R_u = m·1[deg>0]
            # (support with isolated cells removed), determined only by ungated_domain and
            # independent of gate_source. No gate-related switch is read here, so the no-gate
            # ablation (lf_gate=False) and the final method compute exactly the same thing on
            # yz/zx, differing only in the gate and suppress on xy
            deg = self._nbr_sum_2d(m) * m
            R = m * (deg > 0).to(m.dtype)
        # The pyramid starts from the raw features g_0 = x
        gs, gt = s, t
        loss_ms = s.new_tensor(0.0)
        es_ch: List[torch.Tensor] = []     # unweighted per-channel energy ‖g_b‖² [B,C] at each scale (profile and contracted moments)
        et_ch: List[torch.Tensor] = []
        ps_ch: List[torch.Tensor] = []     # weighted per-channel energy Σ w·g_b² [B,C] at each scale (filled only when the profile needs weighting)
        pt_ch: List[torch.Tensor] = []
        for b in range(self.num_scales):
            # suppress is computed only on gated planes and when b ∈ suppress_scales (default
            # range(2, num_scales), i.e. the smoother scales)
            do_sup = gated and (b in self._suppress_scales)
            sp_b = salp[b] if salp is not None else None
            # One recomputation segment per scale: intermediate activations inside the segment are
            # not kept, only the boundaries gs/gt. This scale's salp and do_suppress are bound to
            # the current loop values via default arguments; k/m/R are teacher-side constants
            # passed via closure, not as segment inputs
            fn = (lambda a, c, _d=do_sup, _sp=sp_b: self._scale_terms_v5(a, c, k, m, R, _sp, _d))
            out = self._ckpt(fn, gs, gt)
            step, es, et, gs, gt = out[:5]
            # After the segment-recomputation wrapper the teacher boundary carries requires_grad;
            # cut it explicitly so that later segments do no needless gradient bookkeeping for the
            # teacher chain
            gt = gt.detach()
            eta_b = self.eta[b].to(step.dtype)     # cast the scale weight to the loss dtype (AMP safe)
            loss_ms = loss_ms + eta_b * step
            es_ch.append(es)
            et_ch.append(et)
            if len(out) > 5:
                ps_ch.append(out[5])
                pt_ch.append(out[6].detach())      # teacher-side weighted energy, no back-propagation

        # Scale-weighted average Σ_b η_b·step_b / Σ_b η_b (eps prevents division by zero)
        loss = loss_ms / (self.eta.sum().to(loss_ms.dtype) + self.eps)

        # Relative multi-scale response profile (i.e. the DRD statistic used in the evaluation):
        # summing over channels at each scale gives a [B,S] energy profile, which is compared after
        # row normalization, independent of magnitude. When the profile needs weighting (gate v on
        # a gated plane with profile_domain='open', R_u on an ungated plane with
        # ungated_domain='support') the weighted energies are used, otherwise the unweighted es/et
        src_s, src_t = (ps_ch, pt_ch) if ps_ch else (es_ch, et_ch)
        prof_s = torch.stack([e.sum(dim=1) for e in src_s], dim=-1)   # [B,S]
        prof_t = torch.stack([e.sum(dim=1) for e in src_t], dim=-1)
        if self.profile_metric == "l1":
            # Final method: first-order total variation 0.5·Σ|p̂−q̂|, whose gradient does not vanish
            # with alignment
            l_prof = self._profile_l1(prof_s, prof_t)
        else:
            # Not the final method: JS divergence (a second-order small quantity near p≈q)
            l_prof = self._js_divergence(prof_s, prof_t)
        loss = loss + self.lambda_profile * l_prof

        # Contracted spectral moments e_b = ‖D^b x‖²: directly reuse the unweighted per-channel
        # energies of the pyramid (whole plane, unaffected by the gate and the statistics domain),
        # with zero extra convolutions
        if self.legacy_lk_moments:
            # For ablation: raw L^k moments
            loss_mom = self._ckpt(
                lambda a, b: self._legacy_lk_moment_loss(a, b, self.num_moments), s, t)
        else:
            K = min(self.num_moments, len(es_ch))
            mom_s: List[torch.Tensor] = list(es_ch[:K])
            mom_t: List[torch.Tensor] = list(et_ch[:K])
            # When more moment orders than scales are needed, continue diffusing from the last
            # scale to fill them
            if self.num_moments > len(es_ch):
                for _ in range(self.num_moments - len(es_ch)):
                    gs = self._diffuse(gs)
                    gt = self._diffuse(gt)
                    mom_s.append(self._channel_energy(gs))
                    mom_t.append(self._channel_energy(gt))
            # The normalization axis is determined by moment_norm; 'joint' (joint channel × scale
            # axis) in the final method, see _moment_mse
            loss_mom = self._moment_mse(mom_s, mom_t)

        return loss + self.lambda_moment * loss_mom

    # -------------------------------------------------------------------------
    # The following _lf_gate / _block_local_resp / _scale_terms / _match_view form the default
    # path: used only when no v5 switch is active (_lf_v5=False). Not the final method.
    # -------------------------------------------------------------------------
    def _lf_gate(self, gt: torch.Tensor, return_energy: bool = False):
        """Not the final method (default path): teacher local-variation gate v ∈ (0,1),
        [B,C,H,W]. Called by _scale_terms when lf_gate=True and no v5 switch is active; the final
        method uses the teacher probability gate of _lf_gate_v5.

        saliency = (L g_t)²: non-negative point-wise (not (L g_t)·g_t, which can be negative
        point-wise).
        τ = gate_kappa·mean_{H,W}(saliency) (per B,C), v = saliency/(saliency+τ). The teacher side
        is detached; the gate does not back-propagate gradients.
        With return_energy=True, e_t = mean_{H,W}((L g_t)²) [B,C,1,1] is also returned. It is
        already computed inside the gate (no extra convolution) and serves suppress as an
        amplitude reference independent of the student."""
        gt_d = gt.detach()
        sal = self._apply_laplacian(gt_d) ** 2                       # (L g_t)² [B,C,H,W]
        e_t = sal.mean(dim=self._spatial_dims(sal), keepdim=True)    # [B,C,1,1]
        tau = self.gate_kappa * e_t + self.eps                       # eps prevents 0/0 for a completely flat teacher
        v = sal / (sal + tau)
        return (v, e_t) if return_energy else v

    def _block_local_resp(self, gs: torch.Tensor, gt: torch.Tensor,
                          v: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Not the final method (default path): block-local response shape alignment, called by
        _scale_terms; the final method uses _block_local_resp_v5.

        gs/gt: [B,C,H,W]. The plane is cut into non-overlapping (bh,bw) blocks (when not divisible,
        padded by replicating the border according to self.pad_mode, consistent with the
        Laplacian's boundary handling); each block is Frobenius-normalized over (C,bh,bw) and
        ‖ns−nt‖² = 2−2cosθ is computed, averaged over all blocks and the batch. When v is given,
        a v-weighted norm and weighted alignment are used instead (v must be per channel,
        [B,C,H,W]).
        Per-block normalization gives shape mismatch in blocks containing small objects the same
        weight as large ground blocks; whole-plane global normalization is the special case
        nh=nw=1.
        Note: the z axis of the yz/zx planes has only 16 cells; with replicate padding to a block
        length of 30, border rows are copied several times and their weight in the block norm and
        the alignment is amplified; the final method uses block_pad='fit' to avoid padding.
        """
        bh, bw = self.block_size
        B, C, H, W = gs.shape
        # Replicate-border padding to an integer multiple of the block length (none when divisible)
        ph = (bh - H % bh) % bh
        pw = (bw - W % bw) % bw
        if ph or pw:
            gs = F.pad(gs, (0, pw, 0, ph), mode=self.pad_mode)
            gt = F.pad(gt, (0, pw, 0, ph), mode=self.pad_mode)
            if v is not None:
                v = F.pad(v, (0, pw, 0, ph), mode=self.pad_mode)
        Hp, Wp = H + ph, W + pw
        nh, nw = Hp // bh, Wp // bw
        # [B,C,nh,bh,nw,bw] -> [B,nh,nw,C,bh,bw]
        gs = gs.view(B, C, nh, bh, nw, bw).permute(0, 2, 4, 1, 3, 5)
        gt = gt.view(B, C, nh, bh, nw, bw).permute(0, 2, 4, 1, 3, 5)
        dims = (3, 4, 5)
        if v is None:
            # Unweighted: 2−2cosθ per block, averaged over blocks and the batch; eps prevents
            # division by zero for all-zero blocks
            ns = gs / torch.sqrt((gs * gs).sum(dim=dims, keepdim=True) + self.eps)
            nt = gt / torch.sqrt((gt * gt).sum(dim=dims, keepdim=True) + self.eps)
            return ((ns - nt) ** 2).sum(dim=dims).mean()
        # Gated block-local alignment: normalize by the v-weighted in-block norm
        # (Σ_block v·ns² = 1), loss = Σ_block v·(ns−nt)² = 2−2cos_v ∈ [0,4].
        # Normalized only once and not divided again by the weight sum, so each block is O(1).
        # With v≡1 it is identical to the unweighted version
        vb = v.view(B, C, nh, bh, nw, bw).permute(0, 2, 4, 1, 3, 5)
        ns = gs / torch.sqrt((vb * gs * gs).sum(dim=dims, keepdim=True) + self.eps)
        nt = gt / torch.sqrt((vb * gt * gt).sum(dim=dims, keepdim=True) + self.eps)
        return (vb * (ns - nt) ** 2).sum(dim=dims).mean()

    def _scale_terms(self, gs: torch.Tensor, gt: torch.Tensor,
                     do_suppress: bool = False):
        """Single-scale loss terms of the default path (not the final method; the segment function
        of the segment recomputation), called per scale by _match_view; the final method uses
        _scale_terms_v5.

        Takes the pyramid at the previous scale gs/gt [B,C,H,W] and returns (unweighted loss, e_s,
        e_t, gs_next, gt_next), where e_s/e_t are the per-channel energies ‖g_b‖² [B,C]. The tensor
        outputs are compatible with segment recomputation; do_suppress is passed via closure (not
        a tensor).

        include_raw_scale=False (default): diffuse first, then compare; segment b (0-based)
        compares D^{b+1} x.
        include_raw_scale=True (for ablation): the diffusion is moved to the end of the segment,
        and segment b compares D^b x (b=0 is the raw field). In that case the pyramid has advanced
        to D^S x after the loop, and the extra moments start from D^{S+1} x; this only matters when
        num_moments > num_scales (the two are equal by default).
        """
        if not self.include_raw_scale:
            # Diffuse first, then compare: segment b compares G_{b+1} = D^{b+1} x
            gs = self._diffuse(gs)
            gt = self._diffuse(gt)

        l_sup = gs.new_tensor(0.0)
        if self.lf_gate:
            # Gated path: v = teacher local-variation gate. match aligns under the v-weighted norm;
            # suppress (only at do_suppress scales) suppresses the student's local-variation energy
            # in flat teacher regions
            v, e_t = self._lf_gate(gt, return_energy=True)
            if self.block_size is not None:
                loss_resp = self._block_local_resp(gs, gt, v=v)
            else:
                # No blocking: normalize the whole sample by the v-weighted norm, then weighted
                # alignment (2−2cos_v)
                sdims = tuple(range(1, gs.dim()))
                ns = gs / torch.sqrt((v * gs * gs).sum(dim=sdims, keepdim=True) + self.eps)
                nt = gt / torch.sqrt((v * gt * gt).sum(dim=sdims, keepdim=True) + self.eps)
                loss_resp = (v * (ns - nt) ** 2).sum(dim=sdims).mean()
            if do_suppress:
                # What suppress acts on is the student's local variation (L g_s)², not the low-pass
                # amplitude g_s²: a flat teacher only means L g_t≈0, not g_t≈0 (road surfaces,
                # walls and object interiors are non-zero constants), and suppressing g_s² would
                # push legitimate constant semantics towards 0 as well.
                # Loss = Σ(1−v)·r_n² / Σ(1−v) (weighted mean over the gate-closed region),
                # r_n = L g_s / √ref; using a weighted mean instead of dividing by the gate-open
                # energy means it does not blow up through a degenerate denominator even when the
                # teacher is completely flat (v≡0).
                # Reference ref:
                #   suppress_ref='teacher_floor': ref = max(detach E_s, E_t). E_t (teacher
                #     local-variation energy at the same scale) is independent of the student,
                #     making the loss homogeneous of degree 2 in the student amplitude; E_s acts as
                #     a floor so that ref does not degenerate to eps when the teacher is completely
                #     flat/zero, and guarantees loss ≤ 1/mean(1−v). Once the student's local
                #     variation drops below the teacher level, ref = E_t is fixed and the gradient
                #     decays linearly to 0 with the student's local variation. When E_s > E_t the
                #     back-propagated gradient is not the derivative of the forward value (the
                #     final method uses the log_tail reference of _lf_suppress_v5).
                #   suppress_ref='self_rms' (for ablation): ref = the student's own E_s; the loss
                #     is homogeneous of degree 0 in the student amplitude and constrains only the
                #     proportion of local variation in flat teacher regions, not its amplitude.
                sdims = tuple(range(1, gs.dim()))
                r_s = self._apply_laplacian(gs)
                e_s = (r_s * r_s).mean(dim=sdims, keepdim=True)
                if self.suppress_ref == 'self_rms':
                    ref = e_s
                else:
                    ref = torch.maximum(e_s.detach(), e_t.mean(dim=1, keepdim=True))
                r_n = r_s / torch.sqrt(ref + self.eps)
                w_sup = 1.0 - v
                l_sup = ((w_sup * r_n * r_n).sum(dim=sdims)
                         / (w_sup.sum(dim=sdims) + self.eps)).mean()
        elif self.block_size is not None:
            # Ungated: block-local response alignment, each (bh,bw) spatial block normalized
            # independently before comparing shapes;
            # 2−2cosθ ∈ [0,4] per block, averaged over blocks, same order of magnitude as the
            # global version
            loss_resp = self._block_local_resp(gs, gt)
        else:
            # Ungated, unblocked: compare after normalizing the whole sample (per channel when
            # per_channel_norm=True)
            ns = self._normalize_response(gs, per_channel=self.per_channel_norm)
            nt = self._normalize_response(gt, per_channel=self.per_channel_norm)
            if self.cosine_resp:   # O(1) magnitude: per-sample ‖ns−nt‖² = 2−2cosθ
                loss_resp = ((ns - nt) ** 2).sum(dim=tuple(range(1, ns.dim()))).mean()
            else:
                # Mean over all elements
                loss_resp = torch.mean((ns - nt) ** 2)

        # Energy term: per-channel Dirichlet energy L2-normalized along channels, then MSE (only
        # the relative distribution across channels is compared)
        Es = F.normalize(self._dirichlet_energy(gs), p=2, dim=1)
        Et = F.normalize(self._dirichlet_energy(gt), p=2, dim=1)
        loss_eng = F.mse_loss(Es, Et)
        step = loss_resp + self.lambda_energy * loss_eng
        if self.lf_gate and do_suppress:
            step = step + self.lambda_suppress * l_sup

        # Per-channel energy ‖g_b‖² [B,C], for the profile and the contracted moments
        es = self._channel_energy(gs)
        et = self._channel_energy(gt)

        if self.include_raw_scale:
            gs = self._diffuse(gs)   # for ablation: compare first, then diffuse (b=0 is the raw field)
            gt = self._diffuse(gt)
        return step, es, et, gs, gt

    # Not the final method (default path): LF distillation loss of a single TPV plane. forward
    # calls it for each of the three planes when no v5 switch is active (_lf_v5=False); the final
    # method uses _match_view_v5.
    # s/t: student/teacher plane features [B,C,H,W]. Loss = Σ_b η_b·step_b / Σ_b η_b +
    # λ_profile·profile term + λ_moment·contracted-moment term.
    def _match_view(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        assert s.shape == t.shape and s.dim() == 4, (
            f"Expect matching [B,C,H,W], got {tuple(s.shape)} vs {tuple(t.shape)}"
        )
        gs, gt = s, t
        loss_ms = s.new_tensor(0.0)
        es_ch: List[torch.Tensor] = []   # contracted moments e_b (per channel) [B,C], reused from the pyramid
        et_ch: List[torch.Tensor] = []

        for b in range(self.num_scales):
            # One recomputation segment per scale: intermediate activations inside the segment are
            # not kept, only the boundaries gs/gt.
            do_sup = self.lf_gate and (b in self._suppress_scales)
            fn = (lambda a, c, _d=do_sup: self._scale_terms(a, c, do_suppress=_d))
            step, es, et, gs, gt = self._ckpt(fn, gs, gt)
            # After the checkpoint wrapper the teacher boundary carries requires_grad; cut it
            # explicitly so that later segments do no needless gradient bookkeeping for the
            # teacher chain (its gradient would be discarded anyway).
            gt = gt.detach()
            eta_b = self.eta[b].to(step.dtype)
            loss_ms = loss_ms + eta_b * step
            es_ch.append(es)
            et_ch.append(et)

        # Scale-weighted average Σ_b η_b·step_b / Σ_b η_b
        loss = loss_ms / (self.eta.sum().to(loss_ms.dtype) + self.eps)

        # ---- Relative multi-scale response profile (i.e. the DRD statistic used in the
        #      evaluation), row-normalized, independent of magnitude ----
        prof_s = torch.stack([e.sum(dim=1) for e in es_ch], dim=-1)   # [B,S]
        prof_t = torch.stack([e.sum(dim=1) for e in et_ch], dim=-1)
        if self.profile_metric == "l1":
            l_prof = self._profile_l1(prof_s, prof_t)                 # first-order total variation
        else:
            l_prof = self._js_divergence(prof_s, prof_t)
        loss = loss + self.lambda_profile * l_prof

        # ---- Contracted spectral moments: directly reuse the per-channel energies of the
        #      pyramid, zero extra convolutions ----
        if self.legacy_lk_moments:
            # For ablation: raw L^k moments
            loss_mom = self._ckpt(
                lambda a, b: self._legacy_lk_moment_loss(a, b, self.num_moments), s, t)
        else:
            K = min(self.num_moments, len(es_ch))
            mom_s: List[torch.Tensor] = list(es_ch[:K])
            mom_t: List[torch.Tensor] = list(et_ch[:K])
            if self.num_moments > len(es_ch):   # continue diffusing when higher orders are needed
                for _ in range(self.num_moments - len(es_ch)):
                    gs = self._diffuse(gs)
                    gt = self._diffuse(gt)
                    mom_s.append(self._channel_energy(gs))
                    mom_t.append(self._channel_energy(gt))
            # The normalization axis is determined by moment_norm ('joint' is the joint channel ×
            # scale axis), see _moment_mse
            loss_mom = self._moment_mse(mom_s, mom_t)

        return loss + self.lambda_moment * loss_mom

    def forward(self, tpv_s: List[torch.Tensor], tpv_t: List[torch.Tensor],
                plane_support: Optional[List[torch.Tensor]] = None,
                plane_prob: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """Total LF distillation loss.

        tpv_s/t: [tpv_xy, tpv_yz, tpv_zx], each [B,C,H,W] (teacher/student TPV tri-plane
        features).
        plane_support / plane_prob: plane projection of the teacher support [B,1,H,W] / teacher
        plane class probabilities [B,K,H,W], each a list over the three planes, built by
        plane_inputs_from_field from the teacher distillation field; read only when
        needs_plane_inputs.
        When some v5 switch is active (_lf_v5=True, final method) each plane goes through
        _match_view_v5; otherwise the default path _match_view (not the final method) is taken and
        these two inputs are not read.
        Returns: (w_xy·L_xy + w_yz·L_yz + w_zx·L_zx) / view_divisor, by default
        (3·xy + yz + zx)/3."""
        # Teacher features do not back-propagate gradients
        if self.detach_teacher:
            tpv_t = [x.detach() for x in tpv_t]
        if self._lf_v5:
            # Final method: first validate and prepare the extra inputs (teacher side, detached;
            # missing inputs or shape mismatches raise an error directly), then compute the loss
            # per plane
            ms, ps = self._lf_v5_inputs(tpv_s, plane_support, plane_prob)
            losses = [self._match_view_v5(s, t, k, ms[k], ps[k])
                      for k, (s, t) in enumerate(zip(tpv_s, tpv_t))]
        else:
            # Default path (not the final method)
            losses = [self._match_view(s, t) for s, t in zip(tpv_s, tpv_t)]
        # Weighted sum over the three planes, divided by view_divisor
        w = self.view_weights
        total = (w[0] * losses[0] + w[1] * losses[1] + w[2] * losses[2]) / self.view_divisor
        return total


# =============================================================================
# High-frequency distillation (voxel logits): TPVHighFreqSpectralDistiller
# =============================================================================

class TPVHighFreqSpectralDistiller(_GraphSpectralBase):
    """
    High-frequency (HF) spectral distillation branch: teacher-gated multi-scale high-frequency
    alignment on voxel logits.

    Input: teacher and student per-class voxel logits [B,C,H,W,Z] (continuous values before
    softmax; C=8 classes in the final method).
    Computed on continuous logits rather than discrete labels, so it is differentiable
    everywhere. In the final method both are common-support fields: they have values only on the
    teacher support Ω_T, and are structural zeros outside Ω_T.

    Multi-scale high-pass residual: h_b = x − D^b x, where D = I − α·L is one diffusion step
    (low-pass), b = 1..num_scales.

    Why gating: the student's high-pass residual is dominated by aliasing noise caused by
    discrete sampling, and the residual channel has the worst signal-to-noise ratio at every
    scale (Theorem 1(iii)); matching residuals indiscriminately point by point would make the
    student fit the noise as if it were detail.
    Therefore a soft gate w ∈ (0,1) is determined by the teacher residual:
      - gate open (the teacher has high-frequency structure here, e.g. curbs, lane markings,
        object boundaries): the detail term aligns the shape of the student residual with the
        teacher's;
      - gate closed (the teacher is smooth): the suppress term suppresses the student's
        high-frequency energy here.
    The gate is computed only from the teacher and does not back-propagate gradients.

    Low-contrast regions (e.g. large continuous road surfaces): teacher residual ≈ 0 ⇒ gate ≈ 0,
    the detail term switches off automatically, and the suppress term pushes the student
    residual down. This is the intended behavior: there is no transferable high-frequency
    structure there, and teacher-student consistency in such regions is the responsibility of
    the low-frequency branch (TPVLowFreqSpectralDistiller) and the task loss; the suppress term
    also flattens sparse-sampling artifacts on the road surface as a side effect.
    If alignment inside low-contrast regions is also desired, lowering gate_kappa (lowering the
    noise floor τ) makes the gate open earlier.

    The final method takes the Ω_T subgraph path (hf_operator = gate_domain = "support", see
    _match_view_support), with total loss:
      L = [Σ_b η_b·(λ_det·L_detail + λ_sup·L_suppress + λ_eng·L_eng)_b
           + Σ_{m∈extra_scales} η_x·(λ_det·L_detail + λ_sup·L_suppress)_m] / (Σ_b η_b + |extra|·η_x)
        + λ_mom·L_mom + λ_prof·L_prof + λ_flat·L_flat + λ_V·L_V + λ_C·L_C + λ_ecc·L_ecc.
    Gated alignment only guarantees "like the teacher where the gate is open, not above the
    target where it is closed"; it does not guarantee that the student's graph-Fourier spectral
    flatness or entropy is ≤ the teacher's; flatness along the scale axis is constrained
    separately by L_flat.
    """

    def __init__(
        self,
        # ---- Multi-scale diffusion and spectral statistics ----
        num_scales: int = 8,               # number of base scales S: residual h_b = x − D^b x, b = 1..S (final method uses the default 8)
        alpha: float = 0.25,               # diffusion step α, D = I − α·L; the base class requires 0 < α ≤ 1/2 (final method uses the default 0.25)
        eta: Optional[List[float]] = None,  # per-scale weights η_b; None → 0.5 + 0.25·b (b = 0..S−1), coarser scales get larger weights
        lambda_energy: float = 0.5,        # weight of the per-scale channel Dirichlet-energy distribution matching L_eng (final method 2.0)
        num_moments: int = 8,              # contracted spectral-moment order K: diffuse the final residual h_S K more steps and take the channel energy at each step
        lambda_moment: float = 0.5,        # weight of the contracted spectral-moment term L_mom (final method 2.0)
        eps: float = 1e-6,                 # numerical safeguard for all divisions/square roots
        detach_teacher: bool = True,       # cut the gradient of the teacher logits in forward
        pad_mode: str = "replicate",       # boundary padding of the grid Laplacian; only used when hf_operator="grid"
        # ---- ECC: local soft Euler-characteristic curve (topological regularization) ----
        lambda_ecc: float = 0.2,           # ECC term weight (final method 4.0)
        ecc_taus: Optional[List[float]] = None,   # soft super-level-set thresholds; None → [0.2, 0.35, 0.5, 0.65, 0.8]
        ecc_kappa: float = 10.0,           # steepness of the soft threshold sigmoid(κ·(p − τ))
        ecc_block: Tuple[int, int, int] = (32, 32, 16),   # block size of the local Euler number; when tpv_aggregator does not give it explicitly, it is shrunk according to distill_res_scale
        ecc_input: str = "prob",           # "prob": super-level sets of the softmax probabilities (default, final method) | "logit": per-class sigmoid
        # ---- Teacher gating, scale profile and flatness ----
        gate_hf: bool = True,              # True: teacher-gated detail + suppress (final method); False: all-ones gate match and no suppress (no-gate ablation)
        gate_kappa: float = 1.0,           # noise floor τ = gate_kappa · mean_R(e), e is the gate energy; smaller opens the gate earlier (final method 0.5)
        lambda_suppress: float = 1.0,      # suppress-term weight γ: suppress student high-frequency energy where the gate is closed (final method 1.0; 0 for the "match only" ablation)
        lambda_profile: float = 0.5,       # weight of the residual scale-energy profile {‖h_b‖²}_b alignment term (final method 5.0)
        lambda_flat: float = 0.25,         # weight of the one-sided hinge relu(F_s − F_t) on scale-axis profile flatness (GM/AM) (final method 1.0)
        legacy_lk_moments: bool = False,   # True: use L^k moments instead of contracted spectral moments (for ablation; only defined on the grid operator); final method False
        # Normalization axis of the contracted moments: "joint" = normalize once over the joint
        # (channel × scale) axis (final method);
        # "per_scale_channel" = normalize once per scale along the channel axis (default, not the
        # final method). See _moment_mse.
        moment_norm: str = "per_scale_channel",
        use_checkpoint: bool = True,       # per-scale/per-threshold segment recomputation, to reduce memory of the distillation graph
        # ---- Per-class channel weighting (the channels of the logits are the classes, so this is
        #      exact per-class weighting) ----
        # Channels are weighted inside the integrals of the detail/suppress terms (and L_V);
        # internally normalized to mean 1, so the overall magnitude is unchanged and weight is
        # only redistributed among classes. Final method [1,1,1,1,1,2,2,1]: classes 5 and 6
        # (Living Being / Objects) are doubled.
        # None: all classes equally weighted.
        class_weights: Optional[List[float]] = None,
        # ---- Residual normalization, profile metric and extra scales ----
        # gate_norm: normalization of the residuals on the grid path and under hf_norm="gate_rms".
        #   "sum" (default, not the final method): residuals are normalized by the gate-weighted
        #       total energy (Σw·ns² = 1), and the loss is then divided by Σw (element-level weight
        #       sum); detail/suppress shrink as 1/N with the number of elements N and are close to
        #       0 at the actual resolution, so this is not used for training.
        #   "mean_rms" (final method): RMS normalization (Σw·ns² = Σw); dividing by Σw outside
        #       turns it into a weighted mean, so both terms are resolution-independent O(1)
        #       quantities. hf_norm="field"/"teacher" requires this value (checked at
        #       construction).
        gate_norm: str = "sum",
        # Metric of the scale-energy profile: "js" = JS divergence (default) | "l1" = first-order
        # total variation (final method)
        profile_metric: str = "js",
        # extra_scales: after the S base scales, diffusion is further accumulated to a total of m
        # steps (e.g. [16, 32]), and another set of gated detail/suppress terms is added on the
        # residual h = x − D^m x (without energy/profile/moments/flatness/ECC), extending the
        # high-pass coverage to lower bands: with α=0.25, 1 − ρ(λ)^32 is 0.56 at λ=0.1, whereas
        # the response of the 8 base scales there is much smaller. Final method [16, 32].
        extra_scales: Optional[List[int]] = None,
        # extra_eta: common weight η_x of the extra scales; None takes the η of the last base scale
        extra_eta: Optional[float] = None,
        # ---- Gating and operator on the Ω_T subgraph (the final method sets all four switches to
        #      the non-default values described below) ----
        # The input is a common-support field: teacher and student logits are non-zero only on the
        # teacher support Ω_T and are structural zeros elsewhere. If the gate statistics and the
        # diffusion are computed on the whole grid: τ is diluted by the large empty grid (the
        # gate is almost fully open inside Ω_T), the suppress weight falls mainly outside Ω_T, and
        # the residual also contains the diffusion halo beyond the support boundary, reflecting
        # occupancy geometry rather than logit variation. The switches below restrict all of this
        # to Ω_T:
        # gate_domain="support": τ, the weights and denominators of detail/suppress, and the RMS
        #      normalization are computed only over the gated region R ⊆ Ω_T; the no-gate ablation
        #      (gate_hf=False) is restricted in the same way. "grid" (default): the whole grid.
        # hf_operator="support": all diffusion/energy operators of the HF branch use the Neumann
        #      Laplacian L_Ω on the Ω_T-induced subgraph instead (missing neighbors at both the
        #      support boundary and the grid boundary are folded back onto the diagonal); the
        #      residual of a constant field is identically 0 on every connected component, so the
        #      residual reflects only logit variation along the surface, not occupancy geometry.
        #      "grid" (default): the 6-neighbor Laplacian on the whole grid.
        # gate_source="prob": the gate energy is taken from the residual of the teacher softmax
        #      probabilities h_p = p_t − D^b p_t (teacher side only, no gradient); detail/suppress
        #      still act on the logit residuals. Most of the teacher's logit variation along the
        #      surface is confidence fluctuation between voxels of the same class, so opening the
        #      gate by the logit residual cannot separate class boundaries from intra-class
        #      fluctuation; probabilities flatten such fluctuations in the saturated regime, so
        #      the gate opens mainly at class boundaries. "logit" (default): the gate energy is the
        #      teacher logit residual h_t².
        # gate_share="voxel": all channels share a per-voxel gate, with gate energy e_v = Σ_c e_c;
        #      "channel" (default): one gate per channel.
        # Non-default values of gate_source/gate_share are only defined when
        # gate_domain = hf_operator = "support".
        # Final method: gate_domain="support", hf_operator="support", gate_source="prob",
        # gate_share="voxel".
        gate_domain: str = "grid",
        hf_operator: str = "grid",
        gate_source: str = "logit",
        gate_share: str = "channel",
        # hf_norm: normalization of the residuals in detail/suppress ("field"/"teacher" require
        #   support mode and gate_norm="mean_rms"):
        #   "gate_rms" (default): teacher and student residuals are each normalized by their own
        #       gate-weighted RMS. The student's denominator depends on its own residual amplitude
        #       where the gate is open, so the student could amplify high frequencies where the
        #       gate is open to lower suppress overall.
        #   "field": the student residual is normalized by the student's overall logit amplitude
        #       a_s (RMS over Ω_T after removing the channel mean, gradient stopped) and then
        #       rescaled to the teacher's typical relative high-frequency level; the teacher
        #       residual is normalized by its own RMS over R. This closes the shortcut above and is
        #       invariant to temperature scaling.
        #   "teacher" (final method): the student residual is measured directly in units of the
        #       teacher residual RMS; there is no denominator on the student side, so the gradient
        #       is the true gradient of the forward loss.
        #   See _field_norm for details.
        # lambda_value: weight of the value-matching term L_V (>0 enables it; requires support
        #   mode). Teacher and student logits, with the channel mean removed and normalized by the
        #   RMS over Ω_T, are compared voxel by voxel inside Ω_T (class-weighted, every Ω_T voxel
        #   weighted equally, not weighted by occupancy geometry). Final method 20.
        #   See _value_match.
        hf_norm: str = "gate_rms",
        lambda_value: float = 0.0,
        # lambda_cont: weight of the class-continuity term L_C (>0 enables it; requires support
        #   mode). On adjacent voxel pairs (u,v) inside R, weighted by the degree to which the
        #   teacher gate is closed at both ends, c_uv = (1−w_u)(1−w_v) (teacher gate at b=1), the
        #   difference of the student class probabilities ‖p_s,u − p_s,v‖² is penalized, i.e. the
        #   weighted Dirichlet energy of the student probability field on the "teacher
        #   non-boundary subgraph". Probabilities rather than logits are used: it targets only
        #   class changes and ignores confidence fluctuations in the saturated regime; it only
        #   requires consistency within a region and does not prescribe which class. Final
        #   method 20.
        #   See _continuity.
        lambda_cont: float = 0.0,
        # suppress_target: target of the suppression term where the gate is closed (values other
        #   than "zero" require support mode).
        #   "teacher" (final method): penalize only the part of the student exceeding the teacher,
        #       relu(n_s² − n_t²); the teacher itself is the optimum of this term.
        #   "zero" (default): push the student's normalized residual energy towards 0; small
        #       teacher fluctuations where the gate is closed would also be pushed below.
        suppress_target: str = "zero",
        # lambda_detail: weight of the gated match (detail) term; the final method uses the
        #   default 1.0. 0 gives the "gated, suppress only" ablation arm: the match where the gate
        #   is open is removed, while the gate, suppress, L_V, L_C and the spectral terms are kept.
        #   Defined only in support mode with gate_hf=True; in other cases a value other than 1
        #   raises an error at construction instead of being silently ignored.
        lambda_detail: float = 1.0,
    ):
        # Base class: registers the grid Laplacian kernels and stores alpha / pad_mode / eps /
        # segment recomputation / contracted-moment normalization axis
        super().__init__(alpha=alpha, pad_mode=pad_mode, eps=eps,
                         use_checkpoint=use_checkpoint, moment_norm=moment_norm)
        assert gate_norm in ("sum", "mean_rms"), gate_norm
        self.gate_norm = gate_norm

        # ---- Configuration validity checks ----
        # Configuration conflicts always raise ValueError at construction (not assert: python -O
        # strips asserts)
        # Value ranges of the enumerated switches
        if gate_domain not in ("grid", "support"):
            raise ValueError(f"gate_domain={gate_domain!r}")
        if hf_operator not in ("grid", "support"):
            raise ValueError(f"hf_operator={hf_operator!r}")
        if gate_source not in ("logit", "prob"):
            raise ValueError(f"gate_source={gate_source!r}")
        if gate_share not in ("channel", "voxel"):
            raise ValueError(f"gate_share={gate_share!r}")
        # The probability gate / per-voxel shared gate are only defined when both gate_domain and
        # hf_operator are "support"
        if (gate_source != "logit" or gate_share != "channel") and not (
                gate_domain == "support" and hf_operator == "support"):
            raise ValueError("gate_source/gate_share are only defined for gate_domain=hf_operator='support'")
        # L^k moments only have a grid-operator version
        if legacy_lk_moments and hf_operator == "support":
            raise ValueError("legacy_lk_moments is only defined on the grid operator (hf_operator='grid')")
        if hf_norm not in ("gate_rms", "field", "teacher"):
            raise ValueError(f"hf_norm={hf_norm!r}")
        if suppress_target not in ("zero", "teacher"):
            raise ValueError(f"suppress_target={suppress_target!r}")
        # Non-default values of the following switches are all only defined in support mode (both
        # switches "support")
        if suppress_target != "zero" and not (gate_domain == "support" and hf_operator == "support"):
            raise ValueError("suppress_target is only defined for gate_domain=hf_operator='support'")
        if (hf_norm != "gate_rms" or float(lambda_value) != 0.0) and not (
                gate_domain == "support" and hf_operator == "support"):
            raise ValueError("hf_norm/lambda_value are only defined for gate_domain=hf_operator='support'")
        # The field/teacher normalizations replace the mean_rms residual normalization (both are
        # mean-type scales; dividing by the weight sum outside gives a weighted mean), and
        # gate_norm is required to be written as "mean_rms" to keep the configuration semantics
        # consistent; gate_norm itself then no longer takes part in the computation
        if hf_norm in ("field", "teacher") and gate_norm != "mean_rms":
            raise ValueError(f"hf_norm={hf_norm!r} replaces the mean_rms normalization and requires gate_norm='mean_rms'")
        # The weights of the additional terms must be finite and non-negative
        if not math.isfinite(float(lambda_value)) or float(lambda_value) < 0:
            raise ValueError(f"lambda_value={lambda_value}")
        if not math.isfinite(float(lambda_cont)) or float(lambda_cont) < 0:
            raise ValueError(f"lambda_cont={lambda_cont}")
        if float(lambda_cont) != 0.0 and not (gate_domain == "support" and hf_operator == "support"):
            raise ValueError("lambda_cont is only defined for gate_domain=hf_operator='support'")
        if not math.isfinite(float(lambda_detail)) or float(lambda_detail) < 0:
            raise ValueError(f"lambda_detail={lambda_detail}")
        if float(lambda_detail) != 1.0 and not (gate_domain == "support" and hf_operator == "support"):
            raise ValueError("lambda_detail is only defined for gate_domain=hf_operator='support'")

        # ---- Store the switches ----
        self.gate_domain = gate_domain
        self.hf_operator = hf_operator
        self.gate_source = gate_source
        self.gate_share = gate_share
        self.hf_norm = hf_norm
        self.lambda_value = float(lambda_value)
        self.lambda_cont = float(lambda_cont)
        self.suppress_target = suppress_target
        self.lambda_detail = float(lambda_detail)
        assert profile_metric in ("js", "l1"), profile_metric
        self.profile_metric = profile_metric
        # Extra scales: must be strictly increasing and all greater than num_scales (equal
        # neighbors would recompute the same residual with 0 diffusion steps)
        if extra_scales:
            xs = [int(m) for m in extra_scales]
            assert all(m > int(num_scales) for m in xs) and all(
                a < b for a, b in zip(xs, xs[1:])), (
                f"extra_scales {xs} must be strictly increasing and all > num_scales={num_scales}")
            self.extra_scales = xs
        else:
            self.extra_scales = None
        # The extra-scale weight defaults to the η of the last base scale (consistent with the eta
        # default formula below: 0.5 + 0.25·(S−1))
        if eta is None:
            _eta_last = 0.5 + 0.25 * (int(num_scales) - 1)
        else:
            _eta_last = float(eta[-1])
        self.extra_eta = float(extra_eta) if extra_eta is not None else _eta_last
        # Class weights are normalized to mean 1 and registered as a buffer (moves with the module
        # across devices)
        if class_weights is not None:
            cw = torch.tensor([float(v) for v in class_weights], dtype=torch.float32)
            cw = cw * (cw.numel() / cw.sum().clamp_min(1e-8))   # normalize to mean 1
            self.register_buffer("class_weights", cw)
        else:
            self.class_weights = None
        self.num_scales = int(num_scales)
        self.lambda_energy = float(lambda_energy)
        self.num_moments = int(num_moments)
        self.lambda_moment = float(lambda_moment)
        self.detach_teacher = bool(detach_teacher)

        # Per-scale weights η_b; default 0.5 + 0.25·b (b = 0..S−1), coarser scales get larger
        # weights
        if eta is None:
            eta = [0.5 + 0.25 * b for b in range(self.num_scales)]
        assert len(eta) == self.num_scales, (
            f"len(eta)={len(eta)} must equal num_scales={self.num_scales}"
        )
        self.register_buffer("eta", torch.tensor(eta, dtype=torch.float32))

        # ECC parameters: weight, super-level-set thresholds (buffer), sigmoid steepness, block
        # size, input type
        self.lambda_ecc = float(lambda_ecc)
        if ecc_taus is None:
            ecc_taus = [0.2, 0.35, 0.5, 0.65, 0.8]
        self.register_buffer("ecc_taus", torch.tensor(ecc_taus, dtype=torch.float32))
        self.ecc_kappa = float(ecc_kappa)
        self.ecc_block = tuple(int(v) for v in ecc_block)
        assert ecc_input in ("prob", "logit")
        self.ecc_input = ecc_input

        # Gating and profile parameters
        self.gate_hf = bool(gate_hf)
        # lambda_detail only multiplies the gated detail term; with gate_hf=False that term does not
        # exist, so a value other than 1 raises an error directly
        if self.lambda_detail != 1.0 and not self.gate_hf:
            raise ValueError("lambda_detail only acts on the gated match; with gate_hf=False this term does not exist, and a value other than 1 would be silently ignored")
        self.gate_kappa = float(gate_kappa)
        self.lambda_suppress = float(lambda_suppress)
        self.lambda_profile = float(lambda_profile)
        self.lambda_flat = float(lambda_flat)
        self.legacy_lk_moments = bool(legacy_lk_moments)

    # ---------------------- Teacher-gated high-frequency alignment (grid version, not the final method) ----------------------

    def _gated_hf_terms(self, hs: torch.Tensor, ht: torch.Tensor):
        """
        Grid version of the gated high-frequency terms. Called by _hf_scale_terms /
        _hf_extra_scale_terms when hf_operator = gate_domain = "grid"; the final method takes the
        Ω_T version _gated_hf_terms_m and does not call this function.

        Input: hs/ht = student/teacher high-pass residuals at some scale [B,C,H,W,Z].
        Gate: w = h_t²/(h_t² + τ), τ = gate_kappa·mean(h_t²) (averaged over the whole grid per
        (B,C); the teacher side is detached and the gate does not back-propagate gradients). The
        gate is always per channel and taken from the teacher logit residual.
        detail = Σ w·(n_s − n_t)² / Σ w: matches the shape of the normalized residuals where the
        teacher has structure;
        suppress = Σ (1−w)·n_s² / Σ (1−w): suppresses the student's (relative) energy where the
        teacher is smooth.
        n_s/n_t are the residuals normalized by each side's own gate-weighted energy, which makes
        both terms invariant to logit temperature scaling of each network. Because the student's denominator depends on hs, soft-threshold optimality or a
        spectral-entropy upper bound cannot be asserted from the unnormalized quadratic
        objective.
        Returns two scalars (l_detail, l_suppress).
        """
        # Gate: uses only the teacher residual (detached); the noise floor τ is the spatial mean of
        # h_t² per (B,C); eps prevents 0/0 when everything is zero
        ht_d = ht.detach()
        ht2 = ht_d * ht_d
        tau = self.gate_kappa * ht2.mean(dim=self._spatial_dims(ht_d), keepdim=True) + self.eps
        w = ht2 / (ht2 + tau)                                   # ∈ (0,1), [B,C,H,W,Z]

        # Residual normalization: per sample, computed over channels and voxels together
        sample_dims = tuple(range(1, hs.dim()))
        if self.gate_norm == "mean_rms":
            # RMS normalization: Σw·ns² = Σw. Dividing by Σw afterwards gives a weighted mean, so
            # detail/suppress are resolution-independent O(1) quantities.
            sw = torch.sum(w, dim=sample_dims, keepdim=True).clamp_min(self.eps)
            ns = hs / torch.sqrt(torch.sum(w * hs * hs, dim=sample_dims, keepdim=True) / sw + self.eps)
            nt = ht / torch.sqrt(torch.sum(w * ht * ht, dim=sample_dims, keepdim=True) / sw + self.eps)
        else:
            # "sum" normalization (default value, not the final method): Σw·ns² = 1; after dividing
            # by Σw below both terms are ∝ 1/N (N is the number of elements), close to 0 at the
            # actual resolution; not used for training, kept only for comparison
            ns = hs / torch.sqrt(torch.sum(w * hs * hs, dim=sample_dims, keepdim=True) + self.eps)
            nt = ht / torch.sqrt(torch.sum(w * ht * ht, dim=sample_dims, keepdim=True) + self.eps)

        # Per-class channel weighting: detail weight = cw·w, suppress weight = cw·(1−w). The
        # denominators are weighted the same way → overall magnitude unchanged, weight only
        # redistributed among channels; the normalization factors ns/nt are unaffected, so
        # temperature invariance is preserved. With cw=None all classes are equally weighted.
        if self.class_weights is not None:
            cw = self.class_weights.to(dtype=w.dtype).view(1, -1, *([1] * (hs.dim() - 2)))   # [1,C,1,1,1]
            w_det = cw * w
            w_sup = cw * (1.0 - w)
        else:
            w_det = w
            w_sup = 1.0 - w

        # Both terms are weighted means; eps prevents division by zero when all weights are zero
        l_detail = torch.sum(w_det * (ns - nt) ** 2) / (torch.sum(w_det) + self.eps)
        l_suppress = torch.sum(w_sup * ns * ns) / (torch.sum(w_sup) + self.eps)
        return l_detail, l_suppress

    # ---------------------- Gating and operators on the Ω_T subgraph (final method) ----------------------------
    # When either gate_domain or hf_operator is "support", _match_view dispatches to
    # _match_view_support and uses the methods below; when both are "grid", the grid path above
    # is taken. Convention: m is the indicator of the teacher support Ω_T [B,1,H,W,Z], and deg is
    # the number of 6-neighbors within Ω_T of each Ω_T voxel (see _support_geometry).

    # Floor coefficient of the region indicator in the RMS weights under the gate_rms
    # normalization (hf_norm="gate_rms" and gate_norm="mean_rms") (see _gated_hf_terms_m); not
    # used by the final method, which uses hf_norm="teacher"
    rms_floor = 1e-3        # class attribute, not part of the state_dict

    # Whether the Ω_T path is taken (_match_view uses this to dispatch to _match_view_support).
    # Both switches are "support" in the final method, so this is always True.
    @property
    def _support_mode(self) -> bool:
        return self.gate_domain == "support" or self.hf_operator == "support"

    def _nbr_sum(self, x: torch.Tensor) -> torch.Tensor:
        """Sum over the 3D 6-neighbors Σ_neighbors x; neighbors outside the grid count as 0.
        x: [B,C,H,W,Z] → same shape.
        Implemented with a depthwise conv3d whose cross-shaped kernel weights are exactly 1 (no
        float32 rounding of −1/6 is introduced).
        The kernel is cached per (dtype, device, C) in a plain attribute, not a buffer, and is not
        part of the state_dict.
        In L_Ω, deg·x − Σ_neighbors x gives the same result with zero padding and replicate
        padding (ghost neighbors cancel between the two terms); zero padding additionally makes
        deg count only real Ω_T neighbors, so isolated voxels are identified exactly. Called by
        _support_geometry and _lap_op."""
        C = x.shape[1]
        # Write directly into __dict__: bypasses nn.Module.__setattr__; the cache is just a plain
        # dict
        cache = self.__dict__.setdefault("_nbr_kernels", {})
        key = (x.dtype, x.device, C)
        k = cache.get(key)
        if k is None:
            # Build the [C,1,3,3,3] cross-shaped kernel: center 0, the 6 face neighbors 1
            k = torch.zeros(C, 1, 3, 3, 3, dtype=x.dtype, device=x.device)
            for idx in ((0, 1, 1), (2, 1, 1), (1, 0, 1), (1, 2, 1), (1, 1, 0), (1, 1, 2)):
                k[(slice(None), 0) + idx] = 1.0
            cache[key] = k
        # groups=C: each channel convolved independently; padding=1 pads 0 outside the grid
        return F.conv3d(x, k, bias=None, stride=1, padding=1, groups=C)

    def _support_geometry(self, t: torch.Tensor):
        """Support geometry from the teacher field → (m, deg), both [B,1,H,W,Z], same dtype as t,
        no gradient.
        m: indicator of Ω_T; a voxel belongs to Ω_T if any channel of the teacher field is
        non-zero (same definition as the support set used by VanillaKD.loss_logit).
        deg: actual number of 6-neighbors within Ω_T of each Ω_T voxel (integer; neighbors outside
        the grid and outside Ω_T are not counted; 0 outside Ω_T).
        Called once by _match_view_support; the result is passed as a constant to the segment
        functions of each scale."""
        m = t.detach().ne(0).any(dim=1, keepdim=True).to(t.dtype)   # [B,1,H,W,Z]
        deg = self._nbr_sum(m) * m                                   # number of neighbors in Ω_T, restricted to Ω_T
        return m, deg

    def _lap_op(self, x: torch.Tensor, m: torch.Tensor, deg: torch.Tensor) -> torch.Tensor:
        """Laplacian operator of the HF branch, x: [B,C,H,W,Z] → same shape.
        hf_operator="support" (final method): L_Ω x = m ⊙ (deg ⊙ x − Σ_neighbors x)/6, i.e.
        (1/6)·the combinatorial Laplacian of the Ω_T-induced subgraph; missing neighbors at both
        the support boundary and the grid boundary are folded back onto the diagonal (Neumann
        boundary).
        Requires x to be 0 outside Ω_T (guaranteed by the caller). Symmetric, positive
        semi-definite, spectrum ⊂ [0,2), and the constant field on each connected component lies
        in the kernel.
        This operator does not read pad_mode; when Ω_T is the whole grid it equals the grid
        operator with pad_mode="replicate" (with pad_mode="constant" the grid operator has a
        Dirichlet boundary and the two are not equal).
        hf_operator="grid" (not the final method): the base class's whole-grid 6-neighbor
        Laplacian _apply_laplacian."""
        if self.hf_operator == "grid":
            return self._apply_laplacian(x)
        # x is 0 outside Ω_T, so Σ_neighbors x naturally contains only neighbors inside Ω_T;
        # multiplying by m restricts the result to Ω_T
        return m * (deg * x - self._nbr_sum(x)) / 6.0

    # One diffusion step (low-pass) D x = x − α·L x, with L = _lap_op (L_Ω in support mode);
    # spectral response ρ(λ) = 1 − αλ, non-negative and monotone on [0,2] for α ≤ 1/2. In support
    # mode it preserves "0 outside Ω_T". The multi-scale low-pass g_b = D^b x is obtained by
    # repeated calls, and the residual is h_b = x − g_b. In the final method it is called by the
    # per-scale segment functions, the contracted moments and _teacher_gate_b1.
    def _diffuse_op(self, x, m, deg):
        return x - self.alpha * self._lap_op(x, m, deg)

    # Per-channel Dirichlet energy xᵀ L x → [B,C]; in support mode it equals
    # (1/6)·Σ_{edges inside Ω_T} (x_u − x_v)² ≥ 0.
    # In the final method it is called by the energy-distribution term L_eng of
    # _hf_scale_terms_m.
    def _dirichlet_energy_op(self, x, m, deg):
        return torch.sum(x * self._lap_op(x, m, deg), dim=self._spatial_dims(x))

    def _gate_region(self, m: torch.Tensor, deg: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Region R used for the gate statistics and the detail/suppress integrals, [B,1,H,W,Z].
        gate_domain="support" and hf_operator="support" (final method): R = Ω_T with isolated
            voxels (deg = 0) removed. Under the subgraph operator the residual of an isolated voxel
            is identically 0 at every scale and has no gradient; keeping them would only dilute τ
            and the suppress denominator.
        gate_domain="support" with the grid operator (or no deg given): R = Ω_T.
        gate_domain="grid": R = the whole grid (all ones)."""
        if self.gate_domain == "support":
            if self.hf_operator == "support" and deg is not None:
                return m * (deg > 0).to(m.dtype)
            return m
        return torch.ones_like(m)

    def _centered_rms(self, x: torch.Tensor, m: torch.Tensor):
        """The field with the channel mean removed, and its per-sample RMS over Ω_T.
        x: [B,C,H,W,Z], m: [B,1,H,W,Z].
        softmax is invariant to adding the same constant to all channels, so removing the channel
        mean only removes the gauge component that softmax cannot see.
        Returns (xc: [B,C,H,W,Z], 0 outside Ω_T; rms: [B,1,1,1,1], averaged over channels and Ω_T
        voxels together).
        In the final method it is used for the normalization of L_V, and for the logit amplitudes
        [a_s, a_t] in _match_view_support."""
        xc = (x - x.mean(dim=1, keepdim=True)) * m
        dims = tuple(range(1, x.dim()))
        n = (m.sum(dim=dims, keepdim=True) * x.shape[1]).clamp_min(1.0)   # number of elements |Ω_T|·C; 1 for an empty support to prevent division by zero
        return xc, torch.sqrt(((xc * xc).sum(dim=dims, keepdim=True) / n).clamp_min(self.eps))   # a floor instead of adding eps: when a non-zero field is scaled, rms scales exactly proportionally and the normalized result is exactly unchanged

    # Relative floor of the teacher residual RMS (multiplied by the teacher logit amplitude a_t),
    # used with hf_norm="field"/"teacher" (used by the final method)
    field_floor = 1e-2      # class attribute, not part of the state_dict

    def _field_norm(self, hs, ht, R, scl):
        """Normalization of the residuals used by detail/suppress under hf_norm="field"/"teacher";
        called by _gated_hf_terms_m and _ungated_hf_match_m (the final method uses
        hf_norm="teacher").
        Inputs: hs/ht are the student/teacher high-pass residuals at some scale [B,C,H,W,Z]; R is
        the gated region [B,1,H,W,Z];
        scl = [a_s, a_t] ([B,2,1,1,1]), the RMS over Ω_T of the teacher/student logits after
        removing the channel mean, both with gradients stopped.
        The residuals first have their channel mean removed (the gauge component that softmax
        cannot see, consistent with the definition of a).
        Teacher scale r_t = √max(mean_R h̃_t², (field_floor·a_t)²) (teacher side, no gradient);
        the floor prevents r_t → 0 when the teacher has almost no high frequency on R, which would
        blow up the residuals.
          "teacher" (final method): ns = h̃_s / r_t, nt = h̃_t / r_t.
          "field": ns = h̃_s·(a_t/a_s) / r_t, nt = h̃_t / r_t, i.e. the student residual is first
          rescaled to the teacher's logit amplitude.
        a_s has its gradient stopped: otherwise the student could lower the loss by amplifying
        class contrast along directions invisible to HF (constants on each connected component,
        isolated voxels) while the residuals and predictions stay the same; the shortcut of
        amplifying high frequencies where the gate is open does not exist either.
        Returns (ns, nt), same shapes as hs/ht."""
        a_s, a_t = scl[:, 0:1], scl[:, 1:2]                     # each [B,1,1,1,1]
        # Remove the channel mean; stop gradients on the teacher side
        hs = hs - hs.mean(dim=1, keepdim=True)
        ht = (ht - ht.mean(dim=1, keepdim=True)).detach()
        # Mean square of the teacher residual over R (over channels and voxels together) → r_t,
        # with the relative floor field_floor·a_t
        dims = tuple(range(1, hs.dim()))
        n_el = (R.sum(dim=dims, keepdim=True) * hs.shape[1]).clamp_min(1.0)   # number of elements |R|·C
        ms_t = (R * ht * ht).sum(dim=dims, keepdim=True) / n_el
        r_t = torch.sqrt(torch.maximum(ms_t, (self.field_floor * a_t) ** 2).clamp_min(self.eps))
        if self.hf_norm == "teacher":
            # The student residual is also measured in units of the teacher residual scale: there
            # is no denominator on the student side, so the gradient is the true gradient of the
            # forward loss, and neither shortcut ("amplify high frequencies where the gate is
            # open" or "amplify contrast invisible to HF") exists; the price is that invariance to
            # the student's temperature scaling no longer holds.
            return hs / r_t, ht / r_t
        # "field": the student residual is first multiplied by a_t/a_s to move to the teacher
        # amplitude, then measured in units of r_t; invariant to the student's temperature scaling
        return hs * (a_t / a_s) / r_t, ht / r_t

    def _value_match(self, s, t, m):
        """Value-matching term L_V, called by _match_view_support when lambda_value > 0 (the final
        method uses lambda_value=20).
        L_V = Σ_{Ω_T, c} cw_c·(ŝ − t̂)² / Σ_{Ω_T, c} cw_c, ŝ = s̃/rms_Ω(s̃), t̂ = t̃/rms_Ω(t̃),
        where s̃/t̃ are the logits with the channel mean removed.
        Invariant to temperature scaling of each of teacher and student; every Ω_T voxel is
        weighted equally (including isolated voxels), independent of the support geometry.
        s/t: [B,C,H,W,Z]; m: [B,1,H,W,Z]. Returns a scalar; the student-side rms carries gradient,
        the teacher side has none."""
        sc, a_s = self._centered_rms(s, m)                      # student: centered field and RMS over Ω_T (with gradient)
        tc, a_t = self._centered_rms(t.detach(), m)             # teacher: same, without gradient
        d2 = (sc / a_s - tc / a_t) ** 2                         # per-voxel per-channel squared difference [B,C,H,W,Z]
        # Class weights [1,C,1,1,1]; all-ones [B,C,1,1,1] when not set
        cw = (self.class_weights.to(dtype=s.dtype).view(1, -1, *([1] * (s.dim() - 2)))
              if self.class_weights is not None else torch.ones_like(s[:, :, :1, :1, :1]))
        w = cw * m                                              # statistics only inside Ω_T
        return (w * d2).sum() / (w.expand_as(d2).sum() + self.eps)   # weighted mean; eps prevents division by zero for an empty support

    @torch.no_grad()
    def _teacher_gate_b1(self, t, m, deg):
        """Teacher gate at b = 1 (the finest scale) → [B,1,H,W,Z], used by _continuity; no
        gradient.
        Same definition as the HF gate (gate_source / gate_share / gate_kappa / region R): e is
        the square of the teacher logit residual or of the teacher probability residual
        (gate_source="prob", final method), τ = κ·mean_R(e), w = e/(e+τ); then the maximum over
        channels is taken (with gate_share="voxel" all channels are identical, so the maximum is
        that gate), restricted to R (0 outside R)."""
        t = t.detach()
        ht = t - self._diffuse_op(t, m, deg)                    # b=1 residual of the teacher logits h_t = t − D t
        hp = None
        if self.gate_source == "prob":
            # Teacher probabilities (softmax computed in float32 and cast back to the original
            # dtype), restricted to Ω_T; b=1 residual h_p = p − D p
            p = torch.softmax(t.float(), dim=1).to(t.dtype) * m
            hp = p - self._diffuse_op(p, m, deg)
        R = self._gate_region(m, deg)
        e = self._gate_energy(ht, hp)                           # gate energy [B,C,H,W,Z]
        sp = self._spatial_dims(e)
        # τ = κ·(mean of e over R), per (B,C); the denominator is 1 when R is empty to prevent
        # division by zero; eps prevents τ = 0
        tau = self.gate_kappa * (e * R).sum(dim=sp, keepdim=True) / R.sum(dim=sp, keepdim=True).clamp_min(1.0) + self.eps
        return (e / (e + tau)).amax(dim=1, keepdim=True) * R   # w = e/(e+τ), max over channels, then restricted to R

    def _continuity(self, s, t, m, deg):
        """Class-continuity term L_C, called by _match_view_support when lambda_cont > 0 (the final
        method uses lambda_cont=20).
        L_C = Σ_{(u,v)∈E} c_uv·‖p_s,u − p_s,v‖² / Σ c_uv,
        where E is the set of voxel pairs adjacent along the H/W/Z axes of the grid (each edge
        counted once); c_uv = (1−w_u)(1−w_v)·1[u,v∈R], w is the teacher gate at b=1 (teacher side,
        no gradient); p_s is the student softmax probability.
        I.e. the weighted Dirichlet energy of the student class-probability field on the "teacher
        non-boundary subgraph": where the teacher sees no boundary, the student's class
        probabilities should be continuous. s/t: [B,C,H,W,Z]. Returns a scalar, with gradient only
        w.r.t. s."""
        w = self._teacher_gate_b1(t, m, deg)                     # teacher gate [B,1,H,W,Z], no gradient
        R = self._gate_region(m, deg)
        closed = (1.0 - w) * R                                   # degree to which the gate is closed, 0 outside R; [B,1,H,W,Z]
        p = torch.softmax(s, dim=1) * m                          # student class probabilities, restricted to Ω_T
        num = s.new_tensor(0.0)
        den = s.new_tensor(0.0)
        # Take adjacent voxel pairs (u, v) = (i, i+1) along each of the three spatial axes
        for dim in (2, 3, 4):
            n = p.shape[dim]
            pa, pb = p.narrow(dim, 0, n - 1), p.narrow(dim, 1, n - 1)       # first/second voxel of each pair
            c = closed.narrow(dim, 0, n - 1) * closed.narrow(dim, 1, n - 1)   # edge weight c_uv = closed_u·closed_v
            num = num + (c * (pa - pb).pow(2).sum(dim=1, keepdim=True)).sum()   # Σ c_uv·‖p_u − p_v‖² (summed over classes)
            den = den + c.sum()
        return num / (den + self.eps)                            # weighted mean; eps prevents division by zero when there is no valid edge

    def _gate_energy(self, ht: torch.Tensor, hp: Optional[torch.Tensor]) -> torch.Tensor:
        """Gate energy e (no gradient), the input of the gate w = e/(e+τ), same shape as ht
        [B,C,H,W,Z].
        ht: teacher logit residual h_t = t − D^b t; hp: teacher probability residual
        h_p = p_t − D^b p_t, or None.
        With gate_source="prob" (final method) and hp given, e = h_p², otherwise e = h_t².
        With gate_share="voxel" (final method), summing over channels gives the per-voxel energy
        e_v = Σ_c e_c, which is broadcast back to all channels, so all channels of the same voxel
        share one gate; with "channel" (default, not the final method) each channel has its own
        gate.
        Called by _gated_hf_terms_m and _teacher_gate_b1."""
        # Gate criterion: the teacher probability residual in prob mode, otherwise the teacher
        # logit residual
        src = hp if (self.gate_source == "prob" and hp is not None) else ht
        e = src.detach() ** 2                                    # the gate is determined only by the teacher; detach ensures it does not back-propagate
        if self.gate_share == "voxel":
            e = e.sum(dim=1, keepdim=True).expand_as(ht)        # [B,1,...] per-voxel energy → broadcast view [B,C,...]
        return e

    def _gated_hf_terms_m(self, hs, ht, m, deg=None, hp=None, scl=None):
        """Ω_T version of the gated high-frequency terms; returns two scalars (l_detail,
        l_suppress).
        hs/ht: student/teacher high-pass residuals at the current scale h_b = x − D^b x,
        [B,C,H,W,Z];
        m/deg: Ω_T indicator [B,1,H,W,Z] and number of 6-neighbors within Ω_T (see
        _support_geometry);
        hp: teacher probability residual (used as the gate criterion when gate_source="prob");
        scl: amplitudes [a_s, a_t] [B,2,1,1,1] (used with hf_norm="field"/"teacher", see
        _field_norm).
        The formulas are the same as in the grid version _gated_hf_terms, but τ, the weights and
        denominators of both terms, and the normalization are computed only over the gated region
        R (see _gate_region; in the final method, Ω_T with isolated voxels removed):
            τ = κ·mean_R(e), w = e/(e+τ), w_det = cw·w·1_R, w_sup = cw·(1−w)·1_R.
        detail: weighted mean square of the difference between the normalized teacher and student
        residuals where the gate is open (the teacher has structure), aligning with the teacher's
        high-frequency shape;
        suppress: weighted mean of the student's normalized residual energy where the gate is
        closed (the teacher is smooth), suppressing the student's spurious high frequencies.
        Final method: gate_source="prob", gate_share="voxel", hf_norm="teacher",
        suppress_target="teacher", κ = gate_kappa = 0.5, class_weights=[1,1,1,1,1,2,2,1]. Called
        by _hf_step_m."""
        R = self._gate_region(m, deg)                            # gated statistics region [B,1,H,W,Z]
        e = self._gate_energy(ht, hp)                            # gate energy (no gradient) [B,C,H,W,Z]
        sp = self._spatial_dims(e)
        # Noise floor τ = κ·mean_R(e), per (B,C): the mean is taken only inside R, so it is not
        # diluted by the structural zeros outside Ω_T
        r_cnt = R.sum(dim=sp, keepdim=True).clamp_min(1.0)       # number of voxels in R, floor 1 to prevent division by zero for an empty support
        tau = self.gate_kappa * (e * R).sum(dim=sp, keepdim=True) / r_cnt + self.eps   # eps guarantees τ>0
        w = e / (e + tau)                                        # soft gate ∈ [0,1): tends to 1 where the teacher residual is far above the noise floor
        wr = w * R                                               # gate restricted to R; weight 0 outside R

        sample_dims = tuple(range(1, hs.dim()))                  # per-sample sum over (C,H,W,Z)
        # Residual normalization, three variants
        if self.hf_norm in ("field", "teacher"):
            # Final method hf_norm="teacher": teacher and student residuals, after removing the
            # channel mean, are both divided by the RMS r_t of the teacher residual over R (floor
            # field_floor·a_t). There is no self-dependent denominator on the student side, so the
            # gradient is the true gradient of the forward loss.
            # With "field" (not the final method) the student residual is additionally multiplied
            # by a_t/a_s. See _field_norm for details.
            ns, nt = self._field_norm(hs, ht, R, scl)
        elif self.gate_norm == "mean_rms":
            # hf_norm="gate_rms" (default, not the final method): teacher and student are each
            # normalized by their gate-weighted RMS, Σ wn·n² = Σ wn, so both terms become weighted
            # means, independent of resolution.
            # RMS weights = gate weights + rms_floor·region indicator: when the gate mass is much
            # larger than rms_floor·|R| this is almost identical to pure gate weighting; when the
            # teacher has no variation at all over R (gate mass → 0) it falls back to the plain RMS
            # over R, avoiding the magnitude blow-up ns = hs/√eps.
            # Both terms scale by the same c², so invariance to the logit temperature is preserved.
            wn = wr + self.rms_floor * R.expand_as(wr)
            sw = torch.sum(wn, dim=sample_dims, keepdim=True).clamp_min(self.eps)
            ns = hs / torch.sqrt(torch.sum(wn * hs * hs, dim=sample_dims, keepdim=True) / sw + self.eps)
            nt = ht / torch.sqrt(torch.sum(wn * ht * ht, dim=sample_dims, keepdim=True) / sw + self.eps)
        else:
            # gate_norm="sum" (not the final method; default value / ablation only): normalize by
            # the gate-weighted total energy (Σ w·n² ≈ 1); dividing by Σw afterwards makes both
            # terms shrink as 1/N with the number of elements N
            ns = hs / torch.sqrt(torch.sum(wr * hs * hs, dim=sample_dims, keepdim=True) + self.eps)
            nt = ht / torch.sqrt(torch.sum(wr * ht * ht, dim=sample_dims, keepdim=True) + self.eps)

        # Per-class channel weighting: cw is already normalized to mean 1 (see __init__); numerator
        # and denominator are both multiplied by cw, which only redistributes weight among classes
        # without changing the overall magnitude; the normalization factors ns/nt are unaffected
        if self.class_weights is not None:
            cw = self.class_weights.to(dtype=w.dtype).view(1, -1, *([1] * (hs.dim() - 2)))   # [1,C,1,1,1]
            w_det = cw * wr                                      # detail weights: where the gate is open
            w_sup = cw * (1.0 - w) * R                           # suppress weights: where the gate is closed, likewise restricted to R
        else:
            w_det = wr
            w_sup = (1.0 - w) * R

        # detail: weighted mean squared difference of the normalized teacher and student residuals
        # where the gate is open
        l_detail = torch.sum(w_det * (ns - nt) ** 2) / (torch.sum(w_det) + self.eps)
        # Integrand of suppress, two targets
        if self.suppress_target == "teacher":
            # Final method: penalize only the part of the student residual energy exceeding the
            # teacher, relu(n_s² − n_t²); the teacher itself is the optimum of this term;
            # the teacher side is detached, so the gradient flows only to the student
            diff = ns * ns - nt.detach() * nt.detach()
            sup_num = torch.where(diff > 0, diff, torch.zeros_like(diff))   # penalize only the part where the student exceeds the teacher (gradient 0 at equality)
        else:
            # "zero" (default, not the final method): push the student residual energy where the
            # gate is closed towards 0 as a whole
            sup_num = ns * ns
        # suppress: weighted mean of the integrand where the gate is closed
        l_suppress = torch.sum(w_sup * sup_num) / (torch.sum(w_sup) + self.eps)
        return l_detail, l_suppress

    def _ungated_hf_match_m(self, hs, ht, m, deg=None, scl=None):
        """Ω_T version of the no-gate ablation (gate_hf=False: equivalent to a gate that is always
        1, with no suppress term); returns a scalar.
        Means and integrals are taken only over the gated region R, with the same normalization as
        the gated version _gated_hf_terms_m:
            loss = Σ_R cw·(n_s − n_t)² / Σ_R cw.
        Not the final method: only called by _hf_step_m in the "no gating" ablation arm."""
        R = self._gate_region(m, deg)
        dims = tuple(range(1, hs.dim()))
        n_el = (R.sum(dim=dims, keepdim=True) * hs.shape[1]).clamp_min(1.0)   # number of elements in R |R|·C, floor 1
        if self.hf_norm in ("field", "teacher"):
            ns, nt = self._field_norm(hs, ht, R, scl)            # same field/teacher normalization as the gated version
        elif self.gate_norm == "mean_rms":
            # Each divided by its RMS over R
            ns = hs / ((R * hs.square()).sum(dim=dims, keepdim=True) / n_el + self.eps).sqrt()
            nt = ht / ((R * ht.square()).sum(dim=dims, keepdim=True) / n_el + self.eps).sqrt()
        else:
            # gate_norm="sum": each divided by the square root of its total energy over R
            ns = hs / torch.sqrt(torch.sum(R * hs * hs, dim=dims, keepdim=True) + self.eps)
            nt = ht / torch.sqrt(torch.sum(R * ht * ht, dim=dims, keepdim=True) + self.eps)
        diff = (ns - nt).square() * R                            # element-wise squared difference, kept only inside R
        if self.class_weights is None:
            return diff.sum() / (R.expand_as(diff).sum() + self.eps)          # mean over R
        # Class-weighted mean
        cw = self.class_weights.to(device=hs.device, dtype=hs.dtype)
        cw = cw.view(1, -1, *([1] * (hs.dim() - 2)))
        return (cw * diff).sum() / ((cw * R).expand_as(diff).sum() + self.eps)

    # High-frequency alignment term of a single scale (Ω_T version); returns a scalar:
    #   gate_hf=True (final method): λ_detail·l_detail + λ_suppress·l_suppress (both 1.0 in the
    #   final method);
    #   gate_hf=False (no-gate ablation): ungated in-region matching _ungated_hf_match_m.
    # Called by _hf_scale_terms_m and _hf_extra_scale_terms_m.
    def _hf_step_m(self, hs, ht, m, deg, hp=None, scl=None):
        if self.gate_hf:
            l_detail, l_suppress = self._gated_hf_terms_m(hs, ht, m, deg, hp, scl)
            # lambda_detail=0 is the "suppress only" ablation arm; with the default 1.0 the
            # multiplication is skipped, so behavior is the same as without the switch
            if self.lambda_detail != 1.0:
                l_detail = self.lambda_detail * l_detail
            # lambda_suppress=0 is the "match only" ablation arm
            return l_detail + self.lambda_suppress * l_suppress
        return self._ungated_hf_match_m(hs, ht, m, deg, scl)

    def _hf_scale_terms_m(self, s, t, gs, gt, m, deg, p, gp, scl):
        """Ω_T version of the single-scale high-frequency terms, used as the segment function of
        segment recomputation (pure function; m/deg/p/gp/scl are gradient-free constant inputs).
        s/t: student/teacher fields [B,C,H,W,Z]; gs/gt: diffusion result of the previous scale
        D^(b−1) x (s/t themselves for the first scale);
        p/gp: teacher probabilities and their diffusion chain (advanced one step in sync with
        gs/gt when gate_source="prob"; otherwise passed through unchanged and not used);
        scl: see _gated_hf_terms_m.
        First each is diffused one step to obtain D^b x, the residual h_b = x − D^b x is taken,
        and this scale's loss is
            step = gated detail/suppress (_hf_step_m) + λ_energy·MSE(Ê_s, Ê_t),
        where Ê is the vector of per-channel Dirichlet energies h_bᵀ L_Ω h_b, L2-normalized along
        channels.
        Returns (unweighted loss of this scale, ‖h_s‖² [B], ‖h_t‖² [B], D^b s, D^b t, D^b p); the
        last three are the starting point of the next scale.
        In the final method it is called once per base scale by _match_view_support
        (λ_energy=2)."""
        # Advance the diffusion chain one step: D x = x − α·L x (with hf_operator="support", L is
        # the subgraph Laplacian L_Ω)
        gs = self._diffuse_op(gs, m, deg)
        gt = self._diffuse_op(gt, m, deg)
        # High-pass residual h_b = x − D^b x
        hs = s - gs
        ht = t - gt
        hp = None
        if self.gate_source == "prob":
            # Advance the teacher probability chain in sync to obtain the probability residual
            # h_p, used as the gate criterion
            gp = self._diffuse_op(gp, m, deg)
            hp = p - gp
        step = self._hf_step_m(hs, ht, m, deg, hp, scl)          # gated detail + suppress
        # Dirichlet-energy alignment: per-channel hᵀL h → [B,C], compared after L2 normalization
        # along channels; invariant to the overall per-sample amplitude, it compares only how the
        # high-frequency energy is distributed across the class channels
        Es = F.normalize(self._dirichlet_energy_op(hs, m, deg), p=2, dim=1)
        Et = F.normalize(self._dirichlet_energy_op(ht, m, deg), p=2, dim=1)
        l_eng = F.mse_loss(Es, Et)
        step = step + self.lambda_energy * l_eng
        # Per-sample residual energy ‖h_b‖² [B], used for the scale-energy profile
        return step, self._sample_energy(hs), self._sample_energy(ht), gs, gt, gp

    # High-frequency terms of the extra deep scales (extra_scales, [16,32] in the final method),
    # Ω_T version, used as the segment function of segment recomputation.
    # Starting from the current diffusion-chain boundaries gs/gt/gp, diffuse n_steps more steps to
    # reach the cumulative total number of steps M (16 or 32), and compute only the gated
    # detail/suppress (_hf_step_m) on the residual h = x − D^M x, without
    # energy/moments/profile/flatness/ECC.
    # Returns (unweighted loss, gs, gt, gp), for the next extra scale to continue diffusing. Called
    # by _match_view_support.
    def _hf_extra_scale_terms_m(self, s, t, gs, gt, m, deg, p, gp, scl, n_steps: int):
        # Diffuse n_steps consecutive steps; the probability chain advances only when
        # gate_source="prob"
        for _ in range(int(n_steps)):
            gs = self._diffuse_op(gs, m, deg)
            gt = self._diffuse_op(gt, m, deg)
            if self.gate_source == "prob":
                gp = self._diffuse_op(gp, m, deg)
        # Deep-scale residuals
        hs = s - gs
        ht = t - gt
        hp = (p - gp) if self.gate_source == "prob" else None   # gate criterion: teacher probability residual
        return self._hf_step_m(hs, ht, m, deg, hp, scl), gs, gt, gp

    # Ω_T version of contracted spectral-moment alignment (same formula as the base class's
    # _contracted_moment_loss, except that diffusion uses _diffuse_op):
    # e_k = ‖D^k x‖² (per channel, [B,C]), k = 1..num_moments, then normalized and compared by
    # _moment_mse (final method moment_norm="joint": normalized once over the joint channel ×
    # order axis).
    # The spectral response ρ(λ)^k of D^k decreases monotonically in λ, so with increasing order
    # the contracted moments suppress rather than amplify noise at the high-frequency end.
    # The inputs xs/xt are the residuals of the final base scale h = x − D^S x; called by
    # _match_view_support as one recomputation segment.
    def _contracted_moment_loss_m(self, xs, xt, m, deg, num_moments: int):
        es: List[torch.Tensor] = []                              # student moments of each order, each item [B,C]
        et: List[torch.Tensor] = []                              # teacher moments of each order
        gs, gt = xs, xt
        for _ in range(int(num_moments)):
            # Diffuse one more step to obtain D^k x and record its per-channel energy
            gs = self._diffuse_op(gs, m, deg)
            gt = self._diffuse_op(gt, m, deg)
            es.append(self._channel_energy(gs))
            et.append(self._channel_energy(gt))
        return self._moment_mse(es, et)

    # ---------------------- ECC: local soft Euler-characteristic curve (topological regularization) ---------------------

    # Pad [B,C,H,W,Z] at the end of each of the three spatial axes with the constant pad_value up
    # to an integer multiple of the block size block=(bh,bw,bz).
    # Returns (padded tensor, (Hp,Wp,Zp)). ECC pads with 0: padded cells are treated as unoccupied,
    # and any product term containing them is 0.
    def _pad_to_multiple_3d(self, x, block, pad_value=0.0):
        assert x.dim() == 5
        B, C, H, W, Z = x.shape
        bh, bw, bz = block
        # Round each axis up to an integer multiple of the block size
        Hp = ((H + bh - 1) // bh) * bh
        Wp = ((W + bw - 1) // bw) * bw
        Zp = ((Z + bz - 1) // bz) * bz
        # F.pad arguments go from the last dim backwards: (Z before, Z after, W before, W after,
        # H before, H after); padding only at the end
        x_pad = F.pad(x, (0, Zp - Z, 0, Wp - W, 0, Hp - H), mode="constant", value=pad_value)
        return x_pad, (Hp, Wp, Zp)

    # Cut the padded [B,C,Hp,Wp,Zp] into non-overlapping bh×bw×bz blocks and rearrange it into
    # [B,C,Nh,Nw,Nz,bh,bw,bz]: dims 2–4 index the blocks, dims 5–7 are the in-block coordinates.
    def _block_view_3d(self, x, block):
        B, C, Hp, Wp, Zp = x.shape
        bh, bw, bz = block
        Nh, Nw, Nz = Hp // bh, Wp // bw, Zp // bz                # number of blocks per axis
        x = x.view(B, C, Nh, bh, Nw, bw, Nz, bz)
        return x.permute(0, 1, 2, 4, 6, 3, 5, 7).contiguous()   # [B,C,Nh,Nw,Nz,bh,bw,bz]

    def soft_euler_char_3d_local(self, occ: torch.Tensor, block=None):
        """
        Block-local soft Euler characteristic. occ: soft occupancy [B,C,H,W,Z] with values in
        [0,1]; block defaults to self.ecc_block.
        Returns (mean of χ over blocks [B,C], per-block χ [B,C,Nh,Nw,Nz]).
        χ = #cubes − #faces + #edges − #vertices: cubes are occupied voxels, and faces / edges /
        vertices are the faces / edges / vertices shared by 2 / 4 / 8 occupied voxels (equivalent
        to the Euler characteristic of the cubical complex with voxels as vertices and
        6-neighborhood as edges). Replacing "all occupied" with the product of the voxel occupancy
        values gives a multilinear softening, which is exact when occ is 0/1 (κ→∞).
        Block localization discards cross-block adjacency and is therefore biased; but the teacher
        and student branches share the same blocking, so the bias cancels in the alignment.
        It is intended as a topological regularizer, not an exact topological invariant.
        """
        block = self.ecc_block if block is None else block
        bh, bw, bz = block
        assert bh >= 2 and bw >= 2 and bz >= 2                   # at least 2 cells per axis in a block, so that adjacent pairs, 2×2 and 2×2×2 configurations exist
        occ_pad, _ = self._pad_to_multiple_3d(occ, block, pad_value=0.0)   # padded cells treated as unoccupied
        x = self._block_view_3d(occ_pad, block)                  # [B,C,Nh,Nw,Nz,bh,bw,bz], in-block axes 5/6/7

        # Number of voxels: sum of the soft occupancy within the block
        n_cubes = x.sum(dim=(5, 6, 7))
        # Shared faces: voxel pairs adjacent along the h / w / z directions; the product is the
        # soft count of both being occupied
        n_faces = (
            (x[..., :-1, :, :] * x[..., 1:, :, :]).sum(dim=(5, 6, 7))
            + (x[..., :, :-1, :] * x[..., :, 1:, :]).sum(dim=(5, 6, 7))
            + (x[..., :, :, :-1] * x[..., :, :, 1:]).sum(dim=(5, 6, 7))
        )
        # Shared edges: the 4 voxels of a 2×2 all occupied. e_h are edges parallel to the h axis
        # (the 2×2 lies in the w-z plane); likewise for e_w and e_z
        e_h = (x[..., :, :-1, :-1] * x[..., :, 1:, :-1]
               * x[..., :, :-1, 1:] * x[..., :, 1:, 1:]).sum(dim=(5, 6, 7))
        e_w = (x[..., :-1, :, :-1] * x[..., 1:, :, :-1]
               * x[..., :-1, :, 1:] * x[..., 1:, :, 1:]).sum(dim=(5, 6, 7))
        e_z = (x[..., :-1, :-1, :] * x[..., 1:, :-1, :]
               * x[..., :-1, 1:, :] * x[..., 1:, 1:, :]).sum(dim=(5, 6, 7))
        n_edges = e_h + e_w + e_z
        # Shared vertices: the 8 voxels of a 2×2×2 all occupied
        n_vertices = (
            x[..., :-1, :-1, :-1] * x[..., 1:, :-1, :-1]
            * x[..., :-1, 1:, :-1] * x[..., 1:, 1:, :-1]
            * x[..., :-1, :-1, 1:] * x[..., 1:, :-1, 1:]
            * x[..., :-1, 1:, 1:] * x[..., 1:, 1:, 1:]
        ).sum(dim=(5, 6, 7))

        chi_blocks = n_cubes - n_faces + n_edges - n_vertices    # [B,C,Nh,Nw,Nz]
        return chi_blocks.mean(dim=(2, 3, 4)), chi_blocks       # block mean [B,C] and per-block χ

    def _ecc_curve(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Per-class ECC curve [B,C,M], where M is the number of thresholds (ecc_taus, default
        [0.2,0.35,0.5,0.65,0.8]).
        logits: [B,C,H,W,Z], or [C,H,W,Z] (a batch dim is added automatically).
        ecc_input="prob" (default, final method) takes soft super-level sets of the softmax class
        probabilities, consistent with the softmax of the segmentation head;
        "logit" (not the final method) applies a per-class sigmoid to the logits, treating the
        mutually exclusive classes as independent Bernoulli fields.
        For each threshold τ: soft occupancy occ = σ(κ·(p − τ)) (κ = ecc_kappa, i.e. a softened
        {p ≥ τ}), then the block mean of the block-local soft Euler characteristic gives [B,C].
        When lambda_ecc>0 it is called once each for teacher and student by _match_view /
        _match_view_support (the final method uses lambda_ecc=4.0).
        """
        if logits.dim() == 4:
            logits = logits.unsqueeze(0)
        assert logits.dim() == 5
        # Turn each class channel into a field with values in [0,1]
        if self.ecc_input == "prob":
            p = torch.softmax(logits, dim=1)                     # softmax along the class axis (final method)
        else:
            p = torch.sigmoid(logits)                            # independent per-class sigmoid (not the final method)
        taus = [float(v) for v in self.ecc_taus.detach().cpu().tolist()]   # thresholds converted to Python floats
        ecc_list = []
        for tau in taus:
            # One recomputation segment per threshold: the 4-/8-way products of the soft Euler
            # count keep a large number of full-size partial products (on the order of GB per
            # threshold at [1,8,480,360,32]), while the segment output is only [B,C].
            # _tau=tau binds the current threshold at definition time via a default argument,
            # avoiding late binding of the closure.
            def _chi(p_in, _tau=tau):
                occ = torch.sigmoid(self.ecc_kappa * (p_in - p_in.new_tensor(_tau)))   # soft super-level set {p ≥ τ}
                chi_mean, _ = self.soft_euler_char_3d_local(occ)
                return chi_mean
            ecc_list.append(self._ckpt(_chi, p))
        return torch.stack(ecc_list, dim=-1)                     # [B,C,M]

    # ---------------------- Main loss --------------------------------------------

    def _ungated_hf_match(self, hs: torch.Tensor, ht: torch.Tensor):
        """Grid version of the ungated matching term (gate_hf=False: gate always 1, no
        complementary suppress); returns a scalar.
        Keeps the gate_norm normalization and the class weights.
        Not the final method: only called by _hf_scale_terms / _hf_extra_scale_terms on the grid
        path (gate_domain=hf_operator="grid")."""
        dims = tuple(range(1, hs.dim()))
        if self.gate_norm == "mean_rms":
            # Each divided by its RMS over the whole sample
            ns = hs / (hs.square().mean(dim=dims, keepdim=True) + self.eps).sqrt()
            nt = ht / (ht.square().mean(dim=dims, keepdim=True) + self.eps).sqrt()
        else:
            # gate_norm="sum": each normalized to unit total energy
            ns = self._normalize_response(hs)
            nt = self._normalize_response(ht)
        diff = (ns - nt).square()
        if self.class_weights is None:
            return diff.sum() / (diff.numel() + self.eps)        # mean over all elements
        # Class-weighted mean
        cw = self.class_weights.to(device=hs.device, dtype=hs.dtype)
        cw = cw.view(1, -1, *([1] * (hs.dim() - 2)))
        return (cw * diff).sum() / (cw.expand_as(diff).sum() + self.eps)

    def _hf_scale_terms(self, s: torch.Tensor, t: torch.Tensor,
                        gs: torch.Tensor, gt: torch.Tensor):
        """Grid version of the single-scale high-frequency terms (segment function of segment
        recomputation): diffuse one step to obtain D^b x, compute the gated detail/suppress on the
        residual h_b = x − D^b x (the ungated match when gate_hf=False), then add
        λ_energy·Dirichlet-energy alignment.
        Returns (unweighted loss, prof_s_b [B], prof_t_b [B], gs_next, gt_next).
        Not the final method: only called by _match_view on the grid path; the final method uses
        _hf_scale_terms_m."""
        # One diffusion step on the whole grid, then take the high-pass residual
        gs = self._diffuse(gs)
        gt = self._diffuse(gt)
        hs = s - gs
        ht = t - gt

        if self.gate_hf:
            l_detail, l_suppress = self._gated_hf_terms(hs, ht)          # grid-version gated detail / suppress
            step = l_detail + self.lambda_suppress * l_suppress
        else:
            step = self._ungated_hf_match(hs, ht)

        # Per-channel Dirichlet energy aligned after L2 normalization along channels
        Es = F.normalize(self._dirichlet_energy(hs), p=2, dim=1)
        Et = F.normalize(self._dirichlet_energy(ht), p=2, dim=1)
        l_eng = F.mse_loss(Es, Et)
        step = step + self.lambda_energy * l_eng
        return step, self._sample_energy(hs), self._sample_energy(ht), gs, gt

    def _hf_extra_scale_terms(self, s: torch.Tensor, t: torch.Tensor,
                              gs: torch.Tensor, gt: torch.Tensor, n_steps: int):
        """Grid version of the extra deep-scale terms: from the current diffusion-chain boundary,
        continue accumulating n_steps diffusion steps to a total of m steps, and compute only the
        gated detail/suppress on the residual h = x − D^m x (without
        energy/moments/profile/flatness/ECC).
        Returns (unweighted loss, gs, gt).
        Not the final method: only called by _match_view on the grid path; the final method uses
        _hf_extra_scale_terms_m."""
        for _ in range(int(n_steps)):
            gs = self._diffuse(gs)
            gt = self._diffuse(gt)
        hs = s - gs
        ht = t - gt
        if self.gate_hf:
            l_detail, l_suppress = self._gated_hf_terms(hs, ht)
            step = l_detail + self.lambda_suppress * l_suppress
        else:
            step = self._ungated_hf_match(hs, ht)
        return step, gs, gt

    # Complete HF loss of a single view. s/t: student/teacher voxel logits [B,C,H,W,Z] (teacher
    # already detached); returns a scalar.
    # When either gate_domain or hf_operator is "support", it dispatches to _match_view_support
    # (the path taken by the final method); otherwise the whole-grid path below is taken (not the
    # final method, i.e. the default value "grid" of both switches). The grid path has the same
    # structure as _match_view_support, but the gate statistics and the diffusion are computed on
    # the whole grid, and there are no L_V and L_C terms.
    def _match_view(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        assert s.dim() == 5 and t.dim() == 5 and s.shape == t.shape, (
            f"Expect matching [B,C,H,W,Z], got {tuple(s.shape)} vs {tuple(t.shape)}"
        )
        if self._support_mode:
            return self._match_view_support(s, t)

        # (A) Multi-scale gated high-frequency alignment (per scale: diffuse one step first, then
        # take the residual h_b = x − D^b x).
        # One recomputation segment per scale: full-size intermediate activations of the voxel
        # logits are not kept, only the segment boundaries gs/gt
        # (a single [1,8,480,360,32] fp32 tensor is about 177MB).
        gs, gt = s, t
        loss_hf = s.new_tensor(0.0)
        prof_s: List[torch.Tensor] = []   # residual scale-energy profile [B] per scale
        prof_t: List[torch.Tensor] = []

        for b in range(self.num_scales):
            step, ps, pt, gs, gt = self._ckpt(self._hf_scale_terms, s, t, gs, gt)
            # The teacher boundary carries requires_grad after the recomputation-segment output;
            # cut it explicitly.
            gt = gt.detach()
            eta_b = self.eta[b].to(step.dtype)                   # scale weight η_b (default 0.5 + 0.25·b)
            loss_hf = loss_hf + eta_b * step
            prof_s.append(ps)
            prof_t.append(pt)

        eta_norm = self.eta.sum().to(loss_hf.dtype)               # Σ η, used to turn the multi-scale loss into a weighted average
        if self.extra_scales:
            # After the base scales, continue accumulating diffusion to a total of m steps (e.g.
            # 16, 32), reusing the gated detail/suppress.
            # Profile/moments/flatness/ECC still look only at the base scales (the gs/gt used by
            # h_final are not advanced).
            gs_e, gt_e = gs, gt
            eta_x = loss_hf.new_tensor(self.extra_eta)           # weight shared by the extra scales (defaults to the last η)
            prev = self.num_scales                               # number of diffusion steps accumulated so far
            for m in self.extra_scales:
                # _n=(m − prev) is the number of diffusion steps still needed in this segment,
                # bound at definition time via a default argument
                fn = (lambda a, b_, c, d, _n=(m - prev):
                      self._hf_extra_scale_terms(a, b_, c, d, _n))
                step, gs_e, gt_e = self._ckpt(fn, s, t, gs_e, gt_e)
                gt_e = gt_e.detach()
                loss_hf = loss_hf + eta_x * step
                prev = m
            eta_norm = eta_norm + float(len(self.extra_scales)) * eta_x.detach()
        loss_hf = loss_hf / (eta_norm + self.eps)                 # η-weighted average over all scales

        # (B) Residual scale-energy profile alignment + flatness hinge: the profile q_b = ‖h_b‖² is
        #     a statistic along the diffusion scale axis (not the graph-Fourier spectrum); it can
        #     be compared directly with evaluation numbers only when the evaluation uses the same
        #     scale axis and the same operator.
        qs = torch.stack(prof_s, dim=-1)                                  # [B,S]
        qt = torch.stack(prof_t, dim=-1)
        if self.profile_metric == "l1":
            l_prof = self._profile_l1(qs, qt)                             # first-order total-variation distance
        else:
            l_prof = self._js_divergence(qs, qt)                          # JS divergence (default)
        flat_s = self._profile_flatness(qs)
        flat_t = self._profile_flatness(qt)
        # One-sided barrier: penalize only "flatter than the teacher" (excess spurious high
        # frequency along the diffusion scale axis; flatter ⇔ E_b saturates earlier ⇔ larger
        # high-frequency share).
        # l_flat being 0 with zero gradient means the constraint is satisfied, not that the term
        # is dead: the derivative of relu on the negative side is exactly 0 by construction.
        # Once triggered, the magnitude of the hinge subgradient does not depend on the excess,
        # so the term staying close to 0 in loss value for long periods does not mean it has no
        # effect on the optimization.
        l_flat = F.relu(flat_s - flat_t).mean()

        # (C) Contracted spectral-moment alignment of the high-pass residual at the final
        # (strongest) base scale h = x − D^S x.
        # The whole thing is one recomputation segment (an internal diffusion chain of
        # num_moments steps, scalar output).
        h_final_s = s - gs
        h_final_t = t - gt
        if self.legacy_lk_moments:
            # Raw L^k moments (for ablation, not the final method; their sensitivity to
            # high-frequency noise grows exponentially with k)
            l_mom = self._ckpt(
                lambda a, b: self._legacy_lk_moment_loss(a, b, self.num_moments),
                h_final_s, h_final_t)
        else:
            # Contracted moments e_k = ‖D^k h‖² (default)
            l_mom = self._ckpt(
                lambda a, b: self._contracted_moment_loss(a, b, self.num_moments),
                h_final_s, h_final_t)

        # Combine: multi-scale HF + moments + profile + flatness
        loss = (loss_hf
                + self.lambda_moment * l_mom
                + self.lambda_profile * l_prof
                + self.lambda_flat * l_flat)
        # (D) Local ECC topological regularization (softmax-probability super-level sets): the
        # per-class teacher and student ECC curves are L2-normalized along the threshold axis and
        # their mean absolute difference is taken
        if self.lambda_ecc > 0:
            ecc_s = F.normalize(self._ecc_curve(s), p=2, dim=-1)
            ecc_t = F.normalize(self._ecc_curve(t), p=2, dim=-1)
            l_ecc = torch.mean(torch.abs(ecc_s - ecc_t))
            loss = loss + self.lambda_ecc * l_ecc
        return loss

    def _match_view_support(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Ω_T version of the single-view HF loss (used by the final method); returns a scalar.
        s/t: student/teacher voxel logits [B,C,H,W,Z] (teacher already detached), both
        common-support fields that are non-zero only on the teacher support Ω_T.
        The structure is the same as the grid version _match_view, with these differences: the
        gate statistics are computed only over the gated region R (gate_domain="support"); with
        hf_operator="support" the diffusion, Dirichlet energy and contracted moments of the HF
        branch all use the Neumann Laplacian L_Ω on the Ω_T-induced subgraph; and there are two
        additional terms, value matching L_V and continuity L_C. ECC is the same as in the grid
        version.
        Total loss = (η-weighted average of the multi-scale HF) + λ_moment·L_mom
                     + λ_profile·L_prof + λ_flat·L_flat + λ_value·L_V + λ_cont·L_C + λ_ecc·L_ECC.
        Final method: 8 base scales + extra_scales=[16,32]; λ_energy=2 (inside each per-scale
        term), λ_moment=2, λ_profile=5 (profile_metric="l1"), λ_flat=1, λ_value=20, λ_cont=20,
        λ_ecc=4."""
        m, deg = self._support_geometry(t)                       # Ω_T indicator [B,1,H,W,Z] and number of 6-neighbors within Ω_T (no gradient)
        if self.hf_operator == "support":
            # The Neumann subgraph operator requires the field to be 0 outside Ω_T; a
            # common-support field should already satisfy this, but it is enforced explicitly here.
            s = s * m
            t = t * m
        # Teacher class probabilities p_t (gate criterion, no gradient): softmax computed in
        # float32 and cast back to the original dtype (AMP safe), restricted to Ω_T; with
        # gate_source="logit" it is only a placeholder and is not used in the segment functions
        p = (torch.softmax(t.detach().float(), dim=1).to(t.dtype) * m) if self.gate_source == "prob" else t.detach()
        gp = p                                                   # starting point of the probability diffusion chain
        if self.hf_norm in ("field", "teacher"):
            # scl = [a_s, a_t] [B,2,1,1,1]: RMS over Ω_T of the teacher/student logits after
            # removing the channel mean, both with gradients stopped
            # (see _field_norm; the final method hf_norm="teacher" uses only a_t as the floor of
            # the teacher residual scale)
            scl = torch.cat([self._centered_rms(s.detach(), m)[1], self._centered_rms(t.detach(), m)[1]], dim=1)
        else:
            scl = torch.ones_like(m[:, :, :1, :1, :1]).expand(-1, 2, -1, -1, -1).contiguous()   # placeholder, not used in the computation

        # (A) Multi-scale gated high-frequency alignment: one recomputation segment per scale; only
        # the diffusion-chain boundaries gs/gt/gp are passed between segments
        gs, gt = s, t
        loss_hf = s.new_tensor(0.0)
        prof_s: List[torch.Tensor] = []                          # student residual energy per scale ‖h_b‖² [B]
        prof_t: List[torch.Tensor] = []                          # teacher residual energy per scale
        for b in range(self.num_scales):
            step, ps, pt, gs, gt, gp = self._ckpt(self._hf_scale_terms_m, s, t, gs, gt, m, deg, p, gp, scl)
            # Teacher-side boundaries carry requires_grad after the recomputation-segment output;
            # cut them explicitly
            gt = gt.detach()
            gp = gp.detach()
            loss_hf = loss_hf + self.eta[b].to(step.dtype) * step   # scale weight η_b (default 0.5 + 0.25·b)
            prof_s.append(ps)
            prof_t.append(pt)

        eta_norm = self.eta.sum().to(loss_hf.dtype)               # Σ η, used to turn the multi-scale loss into a weighted average
        if self.extra_scales:
            # Extra deep scales ([16,32] in the final method): continue diffusing from the
            # base-scale boundaries to cumulative totals of 16 and 32 steps, adding only the gated
            # detail/suppress; profile/moments/flatness still look only at the base scales (the
            # gs/gt used by h_final are not advanced)
            gs_e, gt_e, gp_e = gs, gt, gp
            eta_x = loss_hf.new_tensor(self.extra_eta)           # weight shared by the extra scales (defaults to the last η)
            prev = self.num_scales                               # number of diffusion steps accumulated so far
            for mm in self.extra_scales:
                # _n=(mm − prev) is the number of diffusion steps still needed in this segment,
                # bound at definition time via a default argument (avoiding late binding of the
                # closure)
                fn = (lambda a, b_, c, d, e, f, g, h, i, _n=(mm - prev):
                      self._hf_extra_scale_terms_m(a, b_, c, d, e, f, g, h, i, _n))
                step, gs_e, gt_e, gp_e = self._ckpt(fn, s, t, gs_e, gt_e, m, deg, p, gp_e, scl)
                gt_e = gt_e.detach()
                gp_e = gp_e.detach()
                loss_hf = loss_hf + eta_x * step
                prev = mm
            eta_norm = eta_norm + float(len(self.extra_scales)) * eta_x.detach()
        loss_hf = loss_hf / (eta_norm + self.eps)                 # η-weighted average over all scales

        # (B) Residual scale-energy profile alignment ([B,S], compared after row normalization,
        #     invariant to the overall magnitude)
        #     + one-sided flatness hinge (penalizes only a student profile flatter than the
        #     teacher's; see the explanation in _match_view)
        qs = torch.stack(prof_s, dim=-1)
        qt = torch.stack(prof_t, dim=-1)
        if self.profile_metric == "l1":
            l_prof = self._profile_l1(qs, qt)                    # final method: first-order total-variation distance
        else:
            l_prof = self._js_divergence(qs, qt)                 # default "js" (not the final method)
        l_flat = F.relu(self._profile_flatness(qs) - self._profile_flatness(qt)).mean()

        # (C) Contracted spectral-moment alignment of the final base-scale residual
        # h = x − D^S x, as one recomputation segment
        h_final_s = s - gs
        h_final_t = t - gt
        if self.legacy_lk_moments:
            # Raw L^k moments (for ablation, not the final method; only allowed together with
            # hf_operator="grid" at construction)
            l_mom = self._ckpt(
                lambda a, b: self._legacy_lk_moment_loss(a, b, self.num_moments),
                h_final_s, h_final_t)
        else:
            # Final method: Ω_T version of the contracted moments
            l_mom = self._ckpt(
                lambda a, b, c, d: self._contracted_moment_loss_m(a, b, c, d, self.num_moments),
                h_final_s, h_final_t, m, deg)

        # Combine: multi-scale HF + moments + profile + flatness
        loss = (loss_hf
                + self.lambda_moment * l_mom
                + self.lambda_profile * l_prof
                + self.lambda_flat * l_flat)
        # Value matching L_V: teacher and student logits, with the channel mean removed and
        # normalized by the RMS over Ω_T, compared voxel by voxel inside Ω_T (see _value_match)
        if self.lambda_value > 0:
            loss = loss + self.lambda_value * self._value_match(s, t, m)
        # Continuity L_C: on adjacent voxel pairs whose both ends lie where the teacher gate is
        # closed, penalize the difference of the student class probabilities (see _continuity);
        # the no-gate ablation arm uses lambda_cont=0
        if self.lambda_cont > 0:
            loss = loss + self.lambda_cont * self._continuity(s, t, m, deg)
        # (D) Local ECC topological regularization: the per-class teacher and student ECC curves
        # are L2-normalized along the threshold axis and their mean absolute difference is taken
        if self.lambda_ecc > 0:
            ecc_s = F.normalize(self._ecc_curve(s), p=2, dim=-1)
            ecc_t = F.normalize(self._ecc_curve(t), p=2, dim=-1)
            l_ecc = torch.mean(torch.abs(ecc_s - ecc_t))
            loss = loss + self.lambda_ecc * l_ecc
        return loss

    def forward(self, logits_s: torch.Tensor, logits_t: torch.Tensor) -> torch.Tensor:
        """HF distillation entry point; returns a scalar loss.
        logits_s/t: student/teacher per-class voxel logits [B,C,H,W,Z] (a batch dim is added
        automatically for [C,H,W,Z]);
        in the final method these are 8-class common-support fields that are non-zero only on the
        teacher support Ω_T.
        Called as self.spectral_distiller(s_field, logits_vox_t) in tpv_aggregator."""
        # Add the batch dim
        if logits_s.dim() == 4:
            logits_s = logits_s.unsqueeze(0)
        if logits_t.dim() == 4:
            logits_t = logits_t.unsqueeze(0)
        # Cut the teacher gradient when detach_teacher=True (default)
        if self.detach_teacher:
            logits_t = logits_t.detach()
        return self._match_view(logits_s, logits_t)