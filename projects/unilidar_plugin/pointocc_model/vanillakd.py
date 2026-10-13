"""VanillaKD: temperature-softened logit KL (dark knowledge) and pointwise feature MSE.

Role in LiEqualizer training: this module implements the dark-knowledge KL term. With the final
configuration (distill_mode='spectral', hybrid_dark_kd=True), tpv_aggregator imports VanillaKD and
uses only its loss_logit (temperature KL), in parallel with the spectral distiller; the feature MSE
does not take part in that mode. The module can also be used on its own as plain (vanilla) KD via
distill_mode='vanilla_kd'.

It shares the whole signal path with the spectral distiller (Spectraldistiller.py) and only replaces
the loss function:
  - feature level: pointwise MSE on the three TPV planes (same tap point as the LF branch);
  - response level: temperature-softened KL on the voxel distillation field over the teacher's
    support set (exactly the same _query_distill_field path as the HF branch).

Deliberately not done (part of the definition of this loss; do not "optimize it in passing"):
  - no diffusion/smoothing, no spectral statistics, no gating, no Laplacian;
  - no comparison after normalization (L2 normalization ≈ cosine, forbidden); mse_loss uses the default mean;
  - fixed KL direction with the teacher as the target: kl_div(log_softmax(student), softmax(teacher)).

Config switch for the standalone use: KDbaseline (bool) -> distill_mode='vanilla_kd', mutually exclusive
with 'spectral'.
Loss naming convention (it determines the weighting path; do not rename):
  - loss_kd_feat / loss_kd_logit: added to the total loss after static λ weighting. They contain no
    point/voxel/sensor substring -> they do not fall into the three GradNorm groups. loss_kd_logit is
    additionally placed under the rho gradient-ratio controller only when the config's rho contains
    'kd' (see pointtpv_seg.py); otherwise it keeps its static weight.
  - kd_raw_feat / kd_raw_logit: unweighted raw values without "loss" in the name -> logging only.

Optional variant placeholders (not implemented by default): adapter=True (a 1x1 conv per view applied
to f_s before the MSE), init_from_teacher=True (student initialized from the teacher weights). If any
variant is enabled, it must be enabled uniformly for all three experiment rows to keep a single variable.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaKD(nn.Module):
    """Plain distillation: pointwise feature MSE + temperature-softened logit KL.

    No smoothing, no normalization statistics, no gating, no Laplacian.
    The module holds no model references; all tensors are passed in by the caller.
    """

    def __init__(self,
                 T: float = 2.0,            # KL temperature
                 lambda_kd: float = 1.0,    # logit KL weight (set to 0 -> Feature-mimicking only)
                 lambda_feat: float = 0.5,  # feature MSE weight (set to 0 -> Soft-label-KL only)
                 detach_teacher: bool = True):
        super().__init__()
        assert T > 0
        self.T = float(T)
        self.lambda_kd = float(lambda_kd)
        self.lambda_feat = float(lambda_feat)
        self.detach_teacher = detach_teacher

    # ---- Response level: temperature KL on the voxel logits over the teacher's support set ----
    def loss_logit(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """z_s/z_t: [N_vox, C], or a coarse-grid distillation field [B, C, gw, gh, gz] (teacher and
        student share the same support set, produced by _query_distill_field from the same dense points).

        A 5D field is first flattened to [N, C] with the teacher support-set mask before the KL: voxels
        outside the support set are structural zeros produced by scatter (constant logits -> constant KL
        term, uninformative gradient); mixing them in produces a fake distillation where "kd_raw_logit is
        stuck at a constant floor" -- the same symptom the spectral distiller had before its fix.
        """
        if self.detach_teacher:
            z_t = z_t.detach()
        if z_s.dim() == 5:
            sup = z_t.detach().ne(0).any(dim=1)          # [B, gw, gh, gz]
            if not bool(sup.any()):
                # teacher field is all zeros (should not happen in theory): return a 0 that keeps a gradient path, to keep DDP aligned
                return z_s.sum() * 0.0
            z_s = z_s.permute(0, 2, 3, 4, 1)[sup]         # [N, C]
            z_t = z_t.permute(0, 2, 3, 4, 1)[sup]
        log_p_s = F.log_softmax(z_s / self.T, dim=-1)
        p_t = F.softmax(z_t / self.T, dim=-1)
        # batchmean averages over dim 0 (= number of voxels); the gradient magnitude is compensated by T^2
        return F.kl_div(log_p_s, p_t, reduction='batchmean') * (self.T ** 2)

    # ---- Feature level: pointwise MSE on the three TPV planes ----
    def loss_feat(self, feats_s, feats_t) -> torch.Tensor:
        # feats_s/feats_t: list of 3 tensors, each [B, C, H, W], same shape for teacher and student
        loss = z = 0.
        for f_s, f_t in zip(feats_s, feats_t):
            if self.detach_teacher:
                f_t = f_t.detach()
            loss = loss + F.mse_loss(f_s, f_t)            # mean over all elements
            z += 1
        return loss / max(z, 1)

    def forward(self, z_s=None, z_t=None, feats_s=None, feats_t=None):
        out = {}
        total = 0.
        if self.lambda_kd > 0 and z_s is not None:
            raw = self.loss_logit(z_s, z_t)
            out['kd_raw_logit'] = raw.detach()
            total = total + self.lambda_kd * raw
        if self.lambda_feat > 0 and feats_s is not None:
            raw = self.loss_feat(feats_s, feats_t)
            out['kd_raw_feat'] = raw.detach()
            total = total + self.lambda_feat * raw
        out['loss_kd_total'] = total
        return out
