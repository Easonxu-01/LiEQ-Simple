import torch, torch.nn as nn, torch.nn.functional as F
from mmdet3d.models.builder import VOXEL_ENCODERS
from mmengine.model import BaseModule
from copy import deepcopy
# from utils.lovasz_losses import lovasz_softmax
# from utils.sem_geo_loss import geo_scal_loss, sem_scal_loss
from projects.unilidar_plugin.utils.semkitti import geo_scal_loss, sem_scal_loss, CE_ssc_loss
from projects.unilidar_plugin.utils.lovasz_softmax import lovasz_softmax
import numpy as np
from projects.unilidar_plugin.utils.nusc_param import nusc_class_frequencies, nusc_class_names
from projects.unilidar_plugin.utils.semkitti import semantic_kitti_class_frequencies, kitti_class_names
import torch.distributed as dist
from .Spectraldistiller import TPVHighFreqSpectralDistiller
import torch.nn.functional as F

CLASS_NAMES_DG = ['Other', 'Driveable Ground', 'Structure', 'Vehicle', 'Nature', 'Living Being', 'Objects', 'Other Ground']


def _cb_adc_consistency_loss(
    logits_p,
    logits_q,
    voxel_label,
    num_classes=8,
    T=1.5,

    # Label / empty setting
    ignore_index=0,
    empty_class=0,

    # Additive structural terms
    boundary_weight=1.0,
    minority_classes=(5, 6),
    minority_weight=5.0,
    continuity_weight=0.001,

    # Asymmetric empty-pull term
    # Trigger: GT non-empty valid voxel, exactly one branch predicts empty_class.
    # Direction: non-empty branch.detach() -> empty branch.
    empty_pull_weight=0.03,
    teacher_conf_threshold=0.0,

    # Require both views to actually cover a voxel before enforcing consistency.
    # Voxel logits come from scatter_nd over each view's own points: voxels the
    # view never touched are exact all-zero vectors (softmax=uniform, and with
    # NO gradient path into that view). Without this gate, the symmetric JS drags
    # the covered view's confident prediction toward uniform on every voxel the
    # other (differently-configured) view missed - ring/sector shaped bands of
    # ground get pushed toward low confidence, and the minority extra weight
    # amplifies exactly the instances a sparse view misses entirely.
    require_dual_coverage=True,

    eps=1e-8,
    return_dict=False,
):
    """
    Final compact CBA-CL loss with asymmetric empty-pull.

    Components:
        1. L_base:
            Original entropy-weighted symmetric JS consistency.

        2. L_boundary:
            Extra JS consistency on one-voxel-thin GT semantic boundary voxels.

        3. L_minority:
            Extra JS consistency on GT minority-class voxels.

        4. L_continuity:
            Single-view same-category continuity.
            For adjacent valid voxel pairs with the same GT label:
                TV(p_i, p_j) + TV(q_i, q_j)

        5. L_empty_asym:
            If one branch predicts empty_class and the other predicts non-empty,
            the non-empty branch is detached as teacher and pulls the empty
            branch through KL. This prevents the non-empty prediction from being
            dragged toward empty by symmetric JS.

    Total:
        L = L_base
          + boundary_weight   * L_boundary
          + minority_weight   * L_minority
          + continuity_weight * L_continuity
          + empty_pull_weight * L_empty_asym

        L = L * T^2
        (T² is the gradient-scale compensation of temperature distillation, a principled term rather than a tuning knob. The overall
          magnitude is applied uniformly by the caller: consistency_gain (config) × hook warmup schedule;
          there is no longer an in-function outer_scale / consistency_loss_weight multiplier.)

    Notes:
        - No boundary maxpool / dilation.
        - No multiplicative modulator.
        - Confidence weight in JS terms is not detached, matching the original
          optimizable baseline.
        - Empty-pull uses prediction-based pseudo routing because no explicit
          per-view occupancy is available.
    """

    # ============================================================
    # 0) Checks
    # ============================================================
    if logits_p.shape != logits_q.shape:
        raise ValueError(
            f"logits_p and logits_q must have the same shape, "
            f"got {logits_p.shape} and {logits_q.shape}"
        )

    if logits_p.dim() != 5:
        raise ValueError(
            f"expected logits shape [B, C, D, H, W], got {logits_p.shape}"
        )

    if voxel_label is None:
        raise ValueError("voxel_label is required for this supervised CBA-CL loss.")

    B, C, D, H, W = logits_p.shape

    if C != num_classes:
        raise ValueError(
            f"num_classes={num_classes}, but logits channel dim is {C}"
        )

    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")

    if empty_class is None:
        empty_pull_weight = 0.0
    else:
        if not (0 <= empty_class < C):
            raise ValueError(
                f"empty_class={empty_class} is out of range [0, {C})"
            )

    device = logits_p.device
    dtype = logits_p.dtype

    if voxel_label.dim() == 3:
        label = voxel_label.unsqueeze(0)
    else:
        label = voxel_label

    if label.shape != (B, D, H, W):
        raise ValueError(
            f"voxel_label shape should be [B,D,H,W] or [D,H,W], "
            f"got {voxel_label.shape}, expected {(B, D, H, W)}"
        )

    label = label.to(device=device)
    valid = (label != ignore_index).to(dtype=dtype)

    if require_dual_coverage:
        with torch.no_grad():
            # Voxels not written by scatter_nd hold exactly all-zero logit vectors (the probability that the network outputs exactly
            # all zeros is 0), which is used to derive each view's coverage mask. Once valid is tightened to dual coverage,
            # all consistency terms (base/boundary/minority/continuity/empty-pull) inherit it automatically.
            # ne(0).any() is used instead of abs().sum(): a boolean intermediate (~44MB) is cheaper than a full-size float one (~177MB).
            cov_p = logits_p.detach().ne(0).any(dim=1).to(dtype)
            cov_q = logits_q.detach().ne(0).any(dim=1).to(dtype)
        valid = valid * cov_p * cov_q

    valid_count = valid.sum()
    if valid_count.item() == 0:
        zero = logits_p.sum() * 0.0
        if return_dict:
            return {
                "loss_total": zero,
                "L_base_raw": zero.detach(),
                "L_boundary_raw": zero.detach(),
                "L_minority_raw": zero.detach(),
                "L_continuity_raw": zero.detach(),
                "L_empty_asym_raw": zero.detach(),
                "valid_count": valid_count.detach(),
                "valid_ratio": valid.detach().mean(),
            }
        return zero

    denom_valid = valid_count.clamp_min(1.0)

    # ============================================================
    # 1) Temperature-scaled distributions
    # ============================================================
    log_p = F.log_softmax(logits_p / T, dim=1)
    log_q = F.log_softmax(logits_q / T, dim=1)

    p = torch.exp(log_p)
    q = torch.exp(log_q)

    # ============================================================
    # 2) Symmetric JS divergence
    #    Keep the same style as the original working baseline.
    # ============================================================
    m = 0.5 * (p + q)
    log_m = torch.log(m + eps)

    kl_pm = F.kl_div(log_m, p, reduction="none").sum(dim=1)
    kl_qm = F.kl_div(log_m, q, reduction="none").sum(dim=1)

    js_div = 0.5 * (kl_pm + kl_qm)
    js_div = js_div.clamp_min(0.0)

    # ============================================================
    # 3) Entropy confidence weight
    #    Detached: the confidence gate should only modulate the loss, not be optimized by it.
    #    Without detach, d(js*w)/dH = -js*exp(-H) < 0, so the model could lower the consistency loss by actively raising
    #    its own entropy where the two views disagree -- the opposite of "improving consistency".
    # ============================================================
    entropy_p = -(p * log_p).sum(dim=1)
    entropy_q = -(q * log_q).sum(dim=1)

    mean_entropy = 0.5 * (entropy_p + entropy_q)
    weight_conf = torch.exp(-mean_entropy).detach()

    weighted_js = js_div * weight_conf

    # ============================================================
    # 4) GT boundary and minority masks
    #    Boundary is one-voxel-thin. No maxpooling / dilation.
    # ============================================================
    with torch.no_grad():
        is_boundary = torch.zeros_like(valid)

        # D-axis boundary
        if D > 1:
            diff = (
                (label[:, 1:, :, :] != label[:, :-1, :, :]).to(dtype)
                * valid[:, 1:, :, :]
                * valid[:, :-1, :, :]
            )
            is_boundary[:, 1:, :, :] = torch.maximum(
                is_boundary[:, 1:, :, :], diff
            )
            is_boundary[:, :-1, :, :] = torch.maximum(
                is_boundary[:, :-1, :, :], diff
            )

        # H-axis boundary
        if H > 1:
            diff = (
                (label[:, :, 1:, :] != label[:, :, :-1, :]).to(dtype)
                * valid[:, :, 1:, :]
                * valid[:, :, :-1, :]
            )
            is_boundary[:, :, 1:, :] = torch.maximum(
                is_boundary[:, :, 1:, :], diff
            )
            is_boundary[:, :, :-1, :] = torch.maximum(
                is_boundary[:, :, :-1, :], diff
            )

        # W-axis boundary
        if W > 1:
            diff = (
                (label[:, :, :, 1:] != label[:, :, :, :-1]).to(dtype)
                * valid[:, :, :, 1:]
                * valid[:, :, :, :-1]
            )
            is_boundary[:, :, :, 1:] = torch.maximum(
                is_boundary[:, :, :, 1:], diff
            )
            is_boundary[:, :, :, :-1] = torch.maximum(
                is_boundary[:, :, :, :-1], diff
            )

        is_boundary = is_boundary * valid

        is_minority = torch.zeros_like(valid)
        if minority_classes is not None and len(minority_classes) > 0:
            for cls in minority_classes:
                if cls < 0 or cls >= C:
                    raise ValueError(
                        f"minority class {cls} out of range [0, {C})"
                    )
                is_minority = torch.maximum(
                    is_minority,
                    (label == cls).to(dtype),
                )

            is_minority = is_minority * valid

    # ============================================================
    # 5) Base JS consistency
    # ============================================================
    L_base = (weighted_js * valid).sum() / denom_valid

    # ============================================================
    # 6) Boundary JS extra
    #    Normalize by valid_count to keep it as a mild additive term.
    # ============================================================
    if boundary_weight is not None and boundary_weight > 0.0:
        L_boundary = (weighted_js * is_boundary).sum() / denom_valid
    else:
        L_boundary = logits_p.sum() * 0.0

    # ============================================================
    # 7) Minority JS extra
    # ============================================================
    if minority_weight is not None and minority_weight > 0.0:
        L_minority = (weighted_js * is_minority).sum() / denom_valid
    else:
        L_minority = logits_p.sum() * 0.0

    # ============================================================
    # 8) Same-category single-view continuity
    # ============================================================
    def tv_pair(a, b):
        return 0.5 * (a - b).abs().sum(dim=1)

    L_continuity = logits_p.sum() * 0.0
    continuity_pair_count = logits_p.new_tensor(0.0)

    if continuity_weight is not None and continuity_weight > 0.0:
        cont_sum = logits_p.sum() * 0.0

        # D-axis same-category pairs
        if D > 1:
            with torch.no_grad():
                same_d = (
                    (label[:, 1:, :, :] == label[:, :-1, :, :]).to(dtype)
                    * valid[:, 1:, :, :]
                    * valid[:, :-1, :, :]
                )

            tv_p_d = tv_pair(p[:, :, 1:, :, :], p[:, :, :-1, :, :])
            tv_q_d = tv_pair(q[:, :, 1:, :, :], q[:, :, :-1, :, :])

            cont_sum = cont_sum + (0.5 * (tv_p_d + tv_q_d) * same_d).sum()
            continuity_pair_count = continuity_pair_count + same_d.sum()

        # H-axis same-category pairs
        if H > 1:
            with torch.no_grad():
                same_h = (
                    (label[:, :, 1:, :] == label[:, :, :-1, :]).to(dtype)
                    * valid[:, :, 1:, :]
                    * valid[:, :, :-1, :]
                )

            tv_p_h = tv_pair(p[:, :, :, 1:, :], p[:, :, :, :-1, :])
            tv_q_h = tv_pair(q[:, :, :, 1:, :], q[:, :, :, :-1, :])

            cont_sum = cont_sum + (0.5 * (tv_p_h + tv_q_h) * same_h).sum()
            continuity_pair_count = continuity_pair_count + same_h.sum()

        # W-axis same-category pairs
        if W > 1:
            with torch.no_grad():
                same_w = (
                    (label[:, :, :, 1:] == label[:, :, :, :-1]).to(dtype)
                    * valid[:, :, :, 1:]
                    * valid[:, :, :, :-1]
                )

            tv_p_w = tv_pair(p[:, :, :, :, 1:], p[:, :, :, :, :-1])
            tv_q_w = tv_pair(q[:, :, :, :, 1:], q[:, :, :, :, :-1])

            cont_sum = cont_sum + (0.5 * (tv_p_w + tv_q_w) * same_w).sum()
            continuity_pair_count = continuity_pair_count + same_w.sum()

        # Normalize by valid_count, not pair_count.
        L_continuity = cont_sum / denom_valid

    # ============================================================
    # 9) Asymmetric empty-pull
    #
    # Since true per-view occupancy is unavailable, we use prediction-based
    # pseudo routing:
    #   q predicts empty, p predicts non-empty -> p.detach() teaches q
    #   p predicts empty, q predicts non-empty -> q.detach() teaches p
    #
    # This term is additive. It does not replace the symmetric JS base.
    # ============================================================
    L_empty_asym = logits_p.sum() * 0.0

    with torch.no_grad():
        pull_q_raw = torch.zeros_like(valid)
        pull_p_raw = torch.zeros_like(valid)
        pull_q_mask = torch.zeros_like(valid)
        pull_p_mask = torch.zeros_like(valid)

    if empty_pull_weight is not None and empty_pull_weight > 0.0:
        with torch.no_grad():
            pred_p = p.argmax(dim=1)
            pred_q = q.argmax(dim=1)

            p_empty = (pred_p == empty_class).to(dtype)
            q_empty = (pred_q == empty_class).to(dtype)

            p_non_empty = 1.0 - p_empty
            q_non_empty = 1.0 - q_empty

            # Trigger only on GT-valid non-empty voxels.
            # q empty, p non-empty: p teaches q.
            pull_q_raw = q_empty * p_non_empty * valid

            # p empty, q non-empty: q teaches p.
            pull_p_raw = p_empty * q_non_empty * valid

            # Optional teacher confidence gate.
            p_teacher_conf = p.detach().max(dim=1).values
            q_teacher_conf = q.detach().max(dim=1).values

            if teacher_conf_threshold is not None and teacher_conf_threshold > 0.0:
                pull_q_raw = pull_q_raw * (
                    p_teacher_conf >= teacher_conf_threshold
                ).to(dtype)
                pull_p_raw = pull_p_raw * (
                    q_teacher_conf >= teacher_conf_threshold
                ).to(dtype)

            # Reliability weight for teacher side.
            # Detached by construction: this term should route gradients only
            # into the empty student branch.
            reliability_p_teacher = torch.exp(-entropy_p.detach())
            reliability_q_teacher = torch.exp(-entropy_q.detach())

            pull_q_mask = pull_q_raw * reliability_p_teacher
            pull_p_mask = pull_p_raw * reliability_q_teacher

        # p.detach() -> q
        kl_p_to_q = (
            p.detach() * (torch.log(p.detach() + eps) - log_q)
        ).sum(dim=1)

        # q.detach() -> p
        kl_q_to_p = (
            q.detach() * (torch.log(q.detach() + eps) - log_p)
        ).sum(dim=1)

        # Normalize by valid_count, not trigger count, to keep this term mild.
        L_empty_asym = (
            (kl_p_to_q * pull_q_mask).sum()
            +
            (kl_q_to_p * pull_p_mask).sum()
        ) / denom_valid

    # ============================================================
    # 10) Total
    # ============================================================
    total_unscaled = (
        L_base
        + (boundary_weight or 0.0) * L_boundary
        + (minority_weight or 0.0) * L_minority
        + (continuity_weight or 0.0) * L_continuity
        + (empty_pull_weight or 0.0) * L_empty_asym
    )

    total = total_unscaled * (T ** 2)

    # ============================================================
    # 11) Diagnostics
    # ============================================================
    if return_dict:
        with torch.no_grad():
            scale = (T ** 2)
            valid_count_safe = valid_count.clamp_min(1.0)

            diagnostics = {
                "loss_total": total,
                "loss_unscaled_before_T": total_unscaled.detach(),

                # raw components before local weights and global scale
                "L_base_raw": L_base.detach(),
                "L_boundary_raw": L_boundary.detach(),
                "L_minority_raw": L_minority.detach(),
                "L_continuity_raw": L_continuity.detach(),
                "L_empty_asym_raw": L_empty_asym.detach(),

                # actual contributions to final scaled loss
                "L_base_contrib": L_base.detach() * scale,
                "L_boundary_contrib": (
                    L_boundary.detach() * (boundary_weight or 0.0) * scale
                ),
                "L_minority_contrib": (
                    L_minority.detach() * (minority_weight or 0.0) * scale
                ),
                "L_continuity_contrib": (
                    L_continuity.detach() * (continuity_weight or 0.0) * scale
                ),
                "L_empty_asym_contrib": (
                    L_empty_asym.detach() * (empty_pull_weight or 0.0) * scale
                ),

                # raw consistency
                "js_div_valid_mean": (
                    js_div.detach() * valid
                ).sum() / valid_count_safe,
                "weighted_js_valid_mean": (
                    weighted_js.detach() * valid
                ).sum() / valid_count_safe,

                # entropy / confidence
                "entropy_p_valid": (
                    entropy_p.detach() * valid
                ).sum() / valid_count_safe,
                "entropy_q_valid": (
                    entropy_q.detach() * valid
                ).sum() / valid_count_safe,
                "weight_conf_valid_mean": (
                    weight_conf.detach() * valid
                ).sum() / valid_count_safe,

                # mask coverage
                "boundary_count": is_boundary.detach().sum(),
                "boundary_ratio": is_boundary.detach().sum() / valid_count_safe,
                "minority_count": is_minority.detach().sum(),
                "minority_ratio": is_minority.detach().sum() / valid_count_safe,
                "continuity_pair_count": continuity_pair_count.detach(),
                "continuity_pair_per_valid": (
                    continuity_pair_count.detach() / valid_count_safe
                ),

                # empty pull diagnostics
                "pull_q_trigger_count": pull_q_raw.detach().sum(),
                "pull_p_trigger_count": pull_p_raw.detach().sum(),
                "pull_q_trigger_ratio": (
                    pull_q_raw.detach().sum() / valid_count_safe
                ),
                "pull_p_trigger_ratio": (
                    pull_p_raw.detach().sum() / valid_count_safe
                ),
                "pull_q_weighted_ratio": (
                    pull_q_mask.detach().sum() / valid_count_safe
                ),
                "pull_p_weighted_ratio": (
                    pull_p_mask.detach().sum() / valid_count_safe
                ),

                "valid_count": valid_count.detach(),
                "valid_ratio": valid.detach().mean(),
            }

        return diagnostics

    return total

def cb_adc_consistency_loss_ckpt(logits_p, logits_q, voxel_label, num_classes,
                                 T=1.5, ignore_index=0,
                                 use_checkpoint=True):
    """Checkpoint wrapper of _cb_adc_consistency_loss.

    The loss has no learnable parameters inside (a pure function of logits), so recomputation via torch.utils.checkpoint
    is safe under DDP (it does not trigger parameters marked-ready-twice). It frees ~8-10 internal dense
    [C,W,H,Z] intermediate tensors (about 1.5GB per pair in fp32), which are recomputed on demand during backward.
    Returns the loss at its "natural magnitude" (only the T² compensation included); the overall gain is applied by the caller.
    """
    if isinstance(voxel_label, torch.Tensor) and voxel_label.dim() == 3:
        voxel_label = voxel_label.unsqueeze(0)

    def _fn(lp, lq, lab):
        return _cb_adc_consistency_loss(
            lp, lq, lab, num_classes=num_classes, T=T,
            ignore_index=ignore_index)

    if use_checkpoint and torch.is_grad_enabled() and (
            logits_p.requires_grad or logits_q.requires_grad):
        return torch.utils.checkpoint.checkpoint(_fn, logits_p, logits_q, voxel_label)
    return _fn(logits_p, logits_q, voxel_label)


def scatter_nd(indices, updates, shape):
    """pytorch edition of tensorflow scatter_nd.

    this function don't contain except handle code. so use this carefully when
    indice repeats, don't support repeat add which is supported in tensorflow.
    """
    ret = torch.zeros(*shape, dtype=updates.dtype, device=updates.device)
    ndim = indices.shape[-1]
    trailing_shape = shape[ndim:]
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    ret[slices] = updates.reshape(-1, *trailing_shape)
    return ret

@VOXEL_ENCODERS.register_module()
class TPVAggregator_Occ(BaseModule):
    def __init__(
        self, tpv_h, tpv_w, tpv_z, grid_size_occ, coarse_ratio, loss_weight=[1,1,1,1],
        nbr_classes=20, in_dims=64, hidden_dims=128, out_dims=None,
        scale_h=2, scale_w=2, scale_z=2, use_checkpoint=False, dual = False,
    ):
        super().__init__()
        self.tpv_h = tpv_h
        self.tpv_w = tpv_w
        self.tpv_z = tpv_z
        self.scale_h = scale_h
        self.scale_w = scale_w
        self.scale_z = scale_z
        self.loss_weight = loss_weight
        self.grid_size_occ = np.asarray(grid_size_occ).astype(np.int32)
        self.coarse_ratio = coarse_ratio
        out_dims = in_dims if out_dims is None else out_dims
        self.dual = dual

        if self.dual:
            self.decoder_nu = nn.Sequential(
                nn.Linear(in_dims, hidden_dims),
                nn.Softplus(),
                nn.Linear(hidden_dims, out_dims),
            )
            self.decoder_sk = nn.Sequential(
                nn.Linear(in_dims, hidden_dims),
                nn.Softplus(),
                nn.Linear(hidden_dims, out_dims),
            )
            self.classifier_nu = nn.Linear(out_dims, nbr_classes[0])
            self.classes_nu = nbr_classes[0]
            self.classifier_sk = nn.Linear(out_dims, nbr_classes[1])
            self.classes_sk = nbr_classes[1]
        else:
            self.decoder = nn.Sequential(
                nn.Linear(in_dims, hidden_dims),
                nn.Softplus(),
                nn.Linear(hidden_dims, out_dims),
            )
            self.classifier = nn.Linear(out_dims, nbr_classes)
            self.classes = nbr_classes
        self.use_checkpoint = use_checkpoint

        self.ce_loss_func = CE_ssc_loss
        self.lovasz_loss_func = lovasz_softmax

        # Initialize the dynamically accumulated class_counts
        self.register_buffer(
            "class_counts", torch.ones(self.classes)  # initialized to 1 to avoid division by zero
        )

    def seesaw_loss(self, logits, labels, p=0.8, eps=1e-6, ignore_index=0):
        """
        logits: any shape, as long as dim=1 is the channel (C)
        labels: semantic labels corresponding to logits, any shape, as long as it matches the spatial dims of logits
        """
        # ---- reshape: (B, C, ...) → (N, C) ----
        B, C = logits.shape[:2]
        logits_flat = logits.permute(0, *range(2, logits.ndim), 1).reshape(-1, C)
        # logits_flat shape = (N, C)
        # ---- labels reshape: (B, ...) → (N,) ----
        labels_flat = labels.reshape(-1)
        # ---- mask ignore_index ----
        valid = labels_flat != ignore_index
        if not valid.any():
            # no valid points → return 0 loss
            return logits.sum() * 0
        logits_flat = logits_flat[valid]
        labels_flat = labels_flat[valid]
        # ---- one-hot ----
        targets = F.one_hot(labels_flat.long(), num_classes=C).float()
        def distribution_agnostic_seesaw_loss(logits, targets, p=0.8, eps=1e-6):
            # class-frequency statistics
            batch_class_count = targets.sum(dim=0)
            self.class_counts += batch_class_count
            cc = self.class_counts
            # s matrix
            conditions = cc[:, None] > cc[None, :]
            trues = (cc[None, :] / cc[:, None]).pow(p)
            falses = torch.ones_like(trues)
            s = torch.where(conditions, trues, falses)
            # overflow protection
            logits = logits - logits.max(dim=-1, keepdim=True)[0]
            numerator = torch.exp(logits)
            denominator = (
                (1 - targets)[:, None, :] * s[None, :, :] * torch.exp(logits)[:, None, :]
            ).sum(dim=-1) + torch.exp(logits)
            sigma = numerator / (denominator + eps)
            loss = (-targets * torch.log(sigma + eps)).sum(dim=-1)
            return loss.mean()
        # ---- seesaw loss ----
        return distribution_agnostic_seesaw_loss(logits_flat, targets, p=p, eps=eps)

        if not self.dual:
            if nbr_classes == 17:
                self.class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies + 0.001))
                self.class_names = nusc_class_names
            elif nbr_classes == 20:
                self.class_weights = torch.from_numpy(1 / np.log(semantic_kitti_class_frequencies + 0.001))# FIXME hardcode
                self.class_names = kitti_class_names
            else:
                self.class_weights = torch.ones(nbr_classes)/nbr_classes  # FIXME hardcode
                self.class_names = CLASS_NAMES_DG
        else:
            self.class_weights_nu = torch.from_numpy(1 / np.log(nusc_class_frequencies + 0.001))
            self.class_weights_sk = torch.from_numpy(1 / np.log(semantic_kitti_class_frequencies + 0.001))

    def forward(self, tpv_list, voxels=None, voxels_coarse=None, voxel_label=None, dataset_flag=None, return_loss=True):
        """
        x y z -> w h z
        tpv_list[0]: bs, c, w, h
        tpv_list[1]: bs, c, h, z
        tpv_list[2]: bs, c, z, w
        """
        if not self.dual or (isinstance(dataset_flag, list) and len(dataset_flag) == 1) or (all(flag.item() == dataset_flag[0].item() for flag in dataset_flag)):
            tpv_xy, tpv_yz, tpv_zx = tpv_list[0], tpv_list[1], tpv_list[2]
            tpv_hw = tpv_xy.permute(0, 1, 3, 2)
            tpv_wz = tpv_zx.permute(0, 1, 3, 2)
            tpv_zh = tpv_yz.permute(0, 1, 3, 2)
            bs, c, _, _ = tpv_hw.shape

            if self.scale_h != 1 or self.scale_w != 1:
                tpv_hw = F.interpolate(
                    tpv_hw,
                    size=(int(self.tpv_h*self.scale_h), int(self.tpv_w*self.scale_w)),
                    mode='bilinear'
                )
            if self.scale_z != 1 or self.scale_h != 1:
                tpv_zh = F.interpolate(
                    tpv_zh,
                    size=(int(self.tpv_z*self.scale_z), int(self.tpv_h*self.scale_h)),
                    mode='bilinear'
                )
            if self.scale_w != 1 or self.scale_z != 1:
                tpv_wz = F.interpolate(
                    tpv_wz,
                    size=(int(self.tpv_w*self.scale_w), int(self.tpv_z*self.scale_z)),
                    mode='bilinear'
                )

            # voxel_coarse: bs, (vox_w*vox_h*vox_z)/coarse_ratio**3, 3
            _, n, _ = voxels_coarse.shape
            voxels_coarse = voxels_coarse.reshape(bs, 1, n, 3)
            voxels_coarse[..., 0] = voxels_coarse[..., 0] / (self.tpv_w*self.scale_w) * 2 - 1
            voxels_coarse[..., 1] = voxels_coarse[..., 1] / (self.tpv_h*self.scale_h) * 2 - 1
            voxels_coarse[..., 2] = voxels_coarse[..., 2] / (self.tpv_z*self.scale_z) * 2 - 1

            sample_loc_vox = voxels_coarse[:, :, :, [0, 1]]
            tpv_hw_vox = F.grid_sample(tpv_hw, sample_loc_vox, padding_mode="border").squeeze(2) # bs, c, n
            sample_loc_vox = voxels_coarse[:, :, :, [1, 2]]
            tpv_zh_vox = F.grid_sample(tpv_zh, sample_loc_vox, padding_mode="border").squeeze(2)
            sample_loc_vox = voxels_coarse[:, :, :, [2, 0]]
            tpv_wz_vox = F.grid_sample(tpv_wz, sample_loc_vox, padding_mode="border").squeeze(2)
            fused = tpv_hw_vox + tpv_zh_vox + tpv_wz_vox

            fused = fused.permute(0, 2, 1)   # bs, whz, c
        else:
            tpv_xy, tpv_yz, tpv_zx = tpv_list[0], tpv_list[1], tpv_list[2]
            tpv_hw = tpv_xy.permute(0, 1, 3, 2)
            tpv_wz = tpv_zx.permute(0, 1, 3, 2)
            tpv_zh = tpv_yz.permute(0, 1, 3, 2)

            tpv_hw_nu, tpv_wz_nu, tpv_zh_nu = tpv_hw[::2], tpv_wz[::2], tpv_zh[::2]
            tpv_hw_sk, tpv_wz_sk, tpv_zh_sk = tpv_hw[1::2], tpv_wz[1::2], tpv_zh[1::2]

            bs_nu, c, _, _ = tpv_hw_nu.shape
            bs_sk, c, _, _ = tpv_hw_sk.shape

            if self.scale_h != 1 or self.scale_w != 1:
                tpv_hw_nu = F.interpolate(
                    tpv_hw_nu,
                    size=(int(self.tpv_h*self.scale_h), int(self.tpv_w*self.scale_w)),
                    mode='bilinear'
                )
                tpv_hw_sk = F.interpolate(
                    tpv_hw_sk,
                    size=(int(self.tpv_h*self.scale_h), int(self.tpv_w*self.scale_w)),
                    mode='bilinear'
                )
            if self.scale_z != 1 or self.scale_h != 1:
                tpv_zh_nu = F.interpolate(
                    tpv_zh_nu,
                    size=(int(self.tpv_z*self.scale_z), int(self.tpv_h*self.scale_h)),
                    mode='bilinear'
                )
                tpv_zh_sk = F.interpolate(
                    tpv_zh_sk,
                    size=(int(self.tpv_z*self.scale_z), int(self.tpv_h*self.scale_h)),
                    mode='bilinear'
                )
            if self.scale_w != 1 or self.scale_z != 1:
                tpv_wz_nu = F.interpolate(
                    tpv_wz_nu,
                    size=(int(self.tpv_w*self.scale_w), int(self.tpv_z*self.scale_z)),
                    mode='bilinear'
                )
                tpv_wz_sk = F.interpolate(
                    tpv_wz_sk,
                    size=(int(self.tpv_w*self.scale_w), int(self.tpv_z*self.scale_z)),
                    mode='bilinear'
                )

            # voxel_coarse: bs, (vox_w*vox_h*vox_z)/coarse_ratio**3, 3
            if isinstance(voxels_coarse, list):
                voxels_coarse_nu = torch.cat(voxels_coarse[::2], dim=0)
                voxels_coarse_sk = torch.cat(voxels_coarse[1::2], dim=0)
                _, n_nu, _ = voxels_coarse_nu.shape
                _, n_sk, _ = voxels_coarse_sk.shape
            else:
                voxels_coarse_nu = voxels_coarse[::2]
                voxels_coarse_sk = voxels_coarse[1::2]
                _, n_nu, _ = voxels_coarse_nu.shape
                _, n_sk, _ = voxels_coarse_sk.shape

            voxels_coarse_nu = voxels_coarse_nu.reshape(bs_nu, 1, n_nu, 3)
            voxels_coarse_nu[..., 0] = voxels_coarse_nu[..., 0] / (self.tpv_w*self.scale_w) * 2 - 1
            voxels_coarse_nu[..., 1] = voxels_coarse_nu[..., 1] / (self.tpv_h*self.scale_h) * 2 - 1
            voxels_coarse_nu[..., 2] = voxels_coarse_nu[..., 2] / (self.tpv_z*self.scale_z) * 2 - 1

            voxels_coarse_sk = voxels_coarse_sk.reshape(bs_sk, 1, n_sk, 3)
            voxels_coarse_sk[..., 0] = voxels_coarse_sk[..., 0] / (self.tpv_w*self.scale_w) * 2 - 1
            voxels_coarse_sk[..., 1] = voxels_coarse_sk[..., 1] / (self.tpv_h*self.scale_h) * 2 - 1
            voxels_coarse_sk[..., 2] = voxels_coarse_sk[..., 2] / (self.tpv_z*self.scale_z) * 2 - 1

            sample_loc_vox_nu = voxels_coarse_nu[:, :, :, [0, 1]]
            tpv_hw_vox_nu = F.grid_sample(tpv_hw_nu, sample_loc_vox_nu, padding_mode="border").squeeze(2) # bs, c, n
            sample_loc_vox_nu = voxels_coarse_nu[:, :, :, [1, 2]]
            tpv_zh_vox_nu = F.grid_sample(tpv_zh_nu, sample_loc_vox_nu, padding_mode="border").squeeze(2)
            sample_loc_vox_nu = voxels_coarse_nu[:, :, :, [2, 0]]
            tpv_wz_vox_nu = F.grid_sample(tpv_wz_nu, sample_loc_vox_nu, padding_mode="border").squeeze(2)
            fused_nu_input = tpv_hw_vox_nu + tpv_zh_vox_nu + tpv_wz_vox_nu
            fused_nu_input = fused_nu_input.permute(0, 2, 1)   # bs_nu, whz, c

            sample_loc_vox_sk = voxels_coarse_sk[:, :, :, [0, 1]]
            tpv_hw_vox_sk = F.grid_sample(tpv_hw_sk, sample_loc_vox_sk, padding_mode="border").squeeze(2) # bs, c, n
            sample_loc_vox_sk = voxels_coarse_sk[:, :, :, [1, 2]]
            tpv_zh_vox_sk = F.grid_sample(tpv_zh_sk, sample_loc_vox_sk, padding_mode="border").squeeze(2)
            sample_loc_vox_sk = voxels_coarse_sk[:, :, :, [2, 0]]
            tpv_wz_vox_sk = F.grid_sample(tpv_wz_sk, sample_loc_vox_sk, padding_mode="border").squeeze(2)
            fused_sk_input = tpv_hw_vox_sk + tpv_zh_vox_sk + tpv_wz_vox_sk
            fused_sk_input = fused_sk_input.permute(0, 2, 1)   # bs_sk, whz, c

        if self.use_checkpoint:
            if not self.dual:
                fused = torch.utils.checkpoint.checkpoint(self.decoder, fused)
                logits = torch.utils.checkpoint.checkpoint(self.classifier, fused)
            elif (isinstance(dataset_flag, list) and len(dataset_flag) == 1) or (all(flag.item() == dataset_flag[0].item() for flag in dataset_flag)):
                flag = dataset_flag[0]
                if flag.item() == 1:
                    fused_nu = torch.utils.checkpoint.checkpoint(self.decoder_nu, fused)
                    logits_nu = torch.utils.checkpoint.checkpoint(self.classifier_nu, fused_nu)
                if flag.item() == 2:
                    fused_sk = torch.utils.checkpoint.checkpoint(self.decoder_sk, fused)
                    logits_sk = torch.utils.checkpoint.checkpoint(self.classifier_sk, fused_sk)
            else:
                fused_nu = torch.utils.checkpoint.checkpoint(self.decoder_nu, fused_nu_input)
                fused_sk = torch.utils.checkpoint.checkpoint(self.decoder_sk, fused_sk_input)
                logits_nu = torch.utils.checkpoint.checkpoint(self.classifier_nu, fused_nu)
                logits_sk = torch.utils.checkpoint.checkpoint(self.classifier_sk, fused_sk)
        else:
            if not self.dual:
                fused = self.decoder(fused)
                logits = self.classifier(fused)
            elif (isinstance(dataset_flag, list) and len(dataset_flag) == 1) or (all(flag.item() == dataset_flag[0].item() for flag in dataset_flag)):
                flag = dataset_flag[0]
                if flag.item() == 1:
                    fused_nu = self.decoder_nu(fused)
                    logits_nu = self.classifier_nu(fused_nu)
                if flag.item() == 2:
                    fused_sk = self.decoder_sk(fused)
                    logits_sk = self.classifier_sk(fused_sk)
            else:
                fused_nu = self.decoder_nu(fused_nu_input)
                fused_sk = self.decoder_sk(fused_sk_input)
                logits_nu = self.classifier_nu(fused_nu)
                logits_sk = self.classifier_sk(fused_sk)

        if not self.dual:
            W, H, D = int(self.grid_size_occ[0]/self.coarse_ratio), int(self.grid_size_occ[1]/self.coarse_ratio), int(self.grid_size_occ[2]/self.coarse_ratio)
            logits = logits.permute(0, 2, 1)
            B, C, N = logits.shape
            logits = logits.reshape(B, C, W, H ,D)
        elif (isinstance(dataset_flag, list) and len(dataset_flag) == 1) or (all(flag.item() == dataset_flag[0].item() for flag in dataset_flag)):
            if flag.item() == 1:
                W1_, H1_, D1_ = int(self.grid_size_occ[0][0]/self.coarse_ratio), int(self.grid_size_occ[0][1]/self.coarse_ratio), int(self.grid_size_occ[0][2]/self.coarse_ratio)
                logits_nu = logits_nu.permute(0, 2, 1)
                B1_, C1_, N1_ = logits_nu.shape
                logits_nu = logits_nu.reshape(B1_, C1_, W1_, H1_, D1_)
            if flag.item() == 2:
                W2_, H2_, D2_ = int(self.grid_size_occ[1][0]/self.coarse_ratio), int(self.grid_size_occ[1][1]/self.coarse_ratio), int(self.grid_size_occ[1][2]/self.coarse_ratio)
                logits_sk = logits_sk.permute(0, 2, 1)
                B2_, C2_, N2_ = logits_sk.shape
                logits_sk = logits_sk.reshape(B2_, C2_, W2_, H2_, D2_)
        else:
            W1_, H1_, D1_ = int(self.grid_size_occ[0][0]/self.coarse_ratio), int(self.grid_size_occ[0][1]/self.coarse_ratio), int(self.grid_size_occ[0][2]/self.coarse_ratio)
            logits_nu = logits_nu.permute(0, 2, 1)
            B1_, C1_, N1_ = logits_nu.shape
            logits_nu = logits_nu.reshape(B1_, C1_, W1_, H1_, D1_)
            W2_, H2_, D2_ = int(self.grid_size_occ[1][0]/self.coarse_ratio), int(self.grid_size_occ[1][1]/self.coarse_ratio), int(self.grid_size_occ[1][2]/self.coarse_ratio)
            logits_sk = logits_sk.permute(0, 2, 1)
            B2_, C2_, N2_ = logits_sk.shape
            logits_sk = logits_sk.reshape(B2_, C2_, W2_, H2_, D2_)

        if return_loss:
            if not self.dual:
                # resize gt
                ratio = voxel_label.shape[2] // H
                if ratio != 1:
                    voxel_label_coarse = voxel_label.reshape(B, W, ratio, H, ratio, D, ratio).permute(0,1,3,5,2,4,6).reshape(B, W, H, D, ratio**3)
                    empty_mask = voxel_label_coarse.sum(-1) == 0
                    voxel_label_coarse = voxel_label_coarse.to(torch.int64)
                    occ_space = voxel_label_coarse[~empty_mask]
                    occ_space[occ_space==0] = -torch.arange(len(occ_space[occ_space==0])).to(occ_space.device) - 1
                    voxel_label_coarse[~empty_mask] = occ_space
                    voxel_label_coarse = torch.mode(voxel_label_coarse, dim=-1)[0]
                    voxel_label_coarse[voxel_label_coarse<0] = 255
                    voxel_label_coarse = voxel_label_coarse.long()
                    with torch.cuda.amp.autocast(enabled=False):
                        logits = logits.float()
                        loss_dict = {}

                        # igore 255 = ignore noise. we keep the loss bascward for the label=0 (free voxels)
                        loss_dict['loss_voxel_ce'] = self.loss_weight[0] * self.ce_loss_func(logits, voxel_label_coarse, self.class_weights.type_as(logits), ignore_index=255)
                        loss_dict['loss_voxel_sem_scal'] = self.loss_weight[2] * sem_scal_loss(logits, voxel_label_coarse, ignore_index=255)
                        loss_dict['loss_voxel_geo_scal'] = self.loss_weight[3] * geo_scal_loss(logits, voxel_label_coarse, ignore_index=255, non_empty_idx=0)
                        loss_dict['loss_voxel_lovasz'] = self.loss_weight[1] * self.lovasz_loss_func(torch.softmax(logits, dim=1), voxel_label_coarse, ignore=255)

                else:
                    with torch.cuda.amp.autocast(enabled=False):
                        logits = logits.float()
                        loss_dict = {}

                        # igore 255 = ignore noise. we keep the loss bascward for the label=0 (free voxels)
                        loss_dict['loss_voxel_ce'] = self.loss_weight[0] * self.ce_loss_func(logits, voxel_label_coarse, self.class_weights.type_as(logits), ignore_index=255)
                        loss_dict['loss_voxel_sem_scal'] = self.loss_weight[2] * sem_scal_loss(logits, voxel_label, ignore_index=255)
                        loss_dict['loss_voxel_geo_scal'] = self.loss_weight[3] * geo_scal_loss(logits, voxel_label, ignore_index=255, non_empty_idx=0)
                        loss_dict['loss_voxel_lovasz'] = self.loss_weight[1] * self.lovasz_loss_func(torch.softmax(logits, dim=1), voxel_label, ignore=255)

            elif (isinstance(dataset_flag, list) and len(dataset_flag) == 1) or (all(flag.item() == dataset_flag[0].item() for flag in dataset_flag)):
                if flag.item() == 1:
                    ratio = voxel_label.shape[2] // H1_
                elif flag.item() == 2:
                    ratio = voxel_label.shape[2] // H2_

                if ratio != 1:
                    if flag.item() == 1:
                        voxel_label_coarse = voxel_label.reshape(B1_, W1_, ratio, H1_, ratio, D1_, ratio).permute(0,1,3,5,2,4,6).reshape(B1_, W1_, H1_, D1_, ratio**3)
                    if flag.item() == 2:
                        voxel_label_coarse = voxel_label.reshape(B2_, W2_, ratio, H2_, ratio, D2_, ratio).permute(0,1,3,5,2,4,6).reshape(B2_, W2_, H2_, D2_, ratio**3)
                    empty_mask = voxel_label_coarse.sum(-1) == 0
                    voxel_label_coarse = voxel_label_coarse.to(torch.int64)
                    occ_space = voxel_label_coarse[~empty_mask]
                    occ_space[occ_space==0] = -torch.arange(len(occ_space[occ_space==0])).to(occ_space.device) - 1
                    voxel_label_coarse[~empty_mask] = occ_space
                    voxel_label_coarse = torch.mode(voxel_label_coarse, dim=-1)[0]
                    voxel_label_coarse[voxel_label_coarse<0] = 255
                    voxel_label_coarse = voxel_label_coarse.long()
                    with torch.cuda.amp.autocast(enabled=False):
                        if flag.item() == 1:
                            logits = logits_nu.float()
                            loss_dict = {}

                            # igore 255 = ignore noise. we keep the loss bascward for the label=0 (free voxels)
                            loss_dict['loss_voxel_ce_1'] = self.loss_weight[0] * self.ce_loss_func(logits, voxel_label_coarse, self.class_weights_nu.type_as(logits), ignore_index=255)
                            loss_dict['loss_voxel_sem_scal_1'] = self.loss_weight[2] * sem_scal_loss(logits, voxel_label_coarse, ignore_index=255)
                            loss_dict['loss_voxel_geo_scal_1'] = self.loss_weight[3] * geo_scal_loss(logits, voxel_label_coarse, ignore_index=255, non_empty_idx=0)
                            loss_dict['loss_voxel_lovasz_1'] = self.loss_weight[1] * self.lovasz_loss_func(torch.softmax(logits, dim=1), voxel_label_coarse, ignore=255)
                        if flag.item() == 2:
                            logits = logits_sk.float()
                            loss_dict = {}

                            # igore 255 = ignore noise. we keep the loss bascward for the label=0 (free voxels)
                            loss_dict['loss_voxel_ce_2'] = self.loss_weight[0] * self.ce_loss_func(logits, voxel_label_coarse, self.class_weights_sk.type_as(logits), ignore_index=255)
                            loss_dict['loss_voxel_sem_scal_2'] = self.loss_weight[2] * sem_scal_loss(logits, voxel_label_coarse, ignore_index=255)
                            loss_dict['loss_voxel_geo_scal_2'] = self.loss_weight[3] * geo_scal_loss(logits, voxel_label_coarse, ignore_index=255, non_empty_idx=0)
                            loss_dict['loss_voxel_lovasz_2'] = self.loss_weight[1] * self.lovasz_loss_func(torch.softmax(logits, dim=1), voxel_label_coarse, ignore=255)
                else:
                    with torch.cuda.amp.autocast(enabled=False):
                        if flag.item() == 1:
                            logits = logits_nu.float()
                            loss_dict = {}
                            # igore 255 = ignore noise. we keep the loss bascward for the label=0 (free voxels)
                            loss_dict['loss_voxel_ce_1'] = self.loss_weight[0] * self.ce_loss_func(logits, voxel_label, self.class_weights_nu.type_as(logits), ignore_index=255)
                            loss_dict['loss_voxel_sem_scal_1'] = self.loss_weight[2] * sem_scal_loss(logits, voxel_label, ignore_index=255)
                            loss_dict['loss_voxel_geo_scal_1'] = self.loss_weight[3] * geo_scal_loss(logits, voxel_label, ignore_index=255, non_empty_idx=0)
                            loss_dict['loss_voxel_lovasz_1'] = self.loss_weight[1] * self.lovasz_loss_func(torch.softmax(logits, dim=1), voxel_label, ignore=255)
                        if flag.item() == 2:
                            logits = logits_sk.float()
                            loss_dict = {}
                            # igore 255 = ignore noise. we keep the loss bascward for the label=0 (free voxels)
                            loss_dict['loss_voxel_ce_2'] = self.loss_weight[0] * self.ce_loss_func(logits, voxel_label, self.class_weights_sk.type_as(logits), ignore_index=255)
                            loss_dict['loss_voxel_sem_scal_2'] = self.loss_weight[2] * sem_scal_loss(logits, voxel_label, ignore_index=255)
                            loss_dict['loss_voxel_geo_scal_2'] = self.loss_weight[3] * geo_scal_loss(logits, voxel_label, ignore_index=255, non_empty_idx=0)
                            loss_dict['loss_voxel_lovasz_2'] = self.loss_weight[1] * self.lovasz_loss_func(torch.softmax(logits, dim=1), voxel_label, ignore=255)
            else:
                if isinstance(voxel_label, list):
                    voxel_label_nu = torch.cat(voxel_label[::2], dim=0)
                    voxel_label_sk = torch.cat(voxel_label[1::2], dim=0)
                else:
                    voxel_label_nu = voxel_label[::2]
                    voxel_label_sk = voxel_label[1::2]
                ratio_nu = voxel_label_nu.shape[2] // H1_
                ratio_sk = voxel_label_sk.shape[2] // H2_
                loss_dict = {}
                if ratio_nu != 1:
                    voxel_label_coarse_nu = voxel_label_nu.reshape(B1_, W1_, ratio_nu, H1_, ratio_nu, D1_, ratio_nu).permute(0,1,3,5,2,4,6).reshape(B1_, W1_, H1_, D1_, ratio_nu**3)
                    empty_mask_nu = voxel_label_coarse_nu.sum(-1) == 0
                    voxel_label_coarse_nu = voxel_label_coarse_nu.to(torch.int64)
                    occ_space_nu = voxel_label_coarse_nu[~empty_mask_nu]
                    occ_space_nu[occ_space_nu==0] = -torch.arange(len(occ_space_nu[occ_space_nu==0])).to(occ_space_nu.device) - 1
                    voxel_label_coarse_nu[~empty_mask_nu] = occ_space_nu
                    voxel_label_coarse_nu = torch.mode(voxel_label_coarse_nu, dim=-1)[0]
                    voxel_label_coarse_nu[voxel_label_coarse_nu<0] = 255
                    voxel_label_coarse_nu = voxel_label_coarse_nu.long()
                    with torch.cuda.amp.autocast(enabled=False):
                        logits_nu = logits_nu.float()

                        # igore 255 = ignore noise. we keep the loss bascward for the label=0 (free voxels)
                        loss_dict['loss_voxel_ce_1'] = self.loss_weight[0] * self.ce_loss_func(logits_nu, voxel_label_coarse_nu, self.class_weights_nu.type_as(logits_nu), ignore_index=255)
                        loss_dict['loss_voxel_sem_scal_1'] = self.loss_weight[2] * sem_scal_loss(logits_nu, voxel_label_coarse_nu, ignore_index=255)
                        loss_dict['loss_voxel_geo_scal_1'] = self.loss_weight[3] * geo_scal_loss(logits_nu, voxel_label_coarse_nu, ignore_index=255, non_empty_idx=0)
                        loss_dict['loss_voxel_lovasz_1'] = self.loss_weight[1] * self.lovasz_loss_func(torch.softmax(logits_nu, dim=1), voxel_label_coarse_nu, ignore=255)
                else:
                    with torch.cuda.amp.autocast(enabled=False):
                        logits_nu = logits_nu.float()
                        # igore 255 = ignore noise. we keep the loss bascward for the label=0 (free voxels)
                        loss_dict['loss_voxel_ce_1'] = self.loss_weight[0] * self.ce_loss_func(logits_nu, voxel_label_nu, self.class_weights_nu.type_as(logits_nu), ignore_index=255)
                        loss_dict['loss_voxel_sem_scal_1'] = self.loss_weight[2] * sem_scal_loss(logits_nu, voxel_label_nu, ignore_index=255)
                        loss_dict['loss_voxel_geo_scal_1'] = self.loss_weight[3] * geo_scal_loss(logits_nu, voxel_label_nu, ignore_index=255, non_empty_idx=0)
                        loss_dict['loss_voxel_lovasz_1'] = self.loss_weight[1] * self.lovasz_loss_func(torch.softmax(logits_nu, dim=1), voxel_label_nu, ignore=255)

                if ratio_sk != 1:
                    voxel_label_coarse_sk = voxel_label_sk.reshape(B2_, W2_, ratio_sk, H2_, ratio_sk, D2_, ratio_sk).permute(0,1,3,5,2,4,6).reshape(B2_, W2_, H2_, D2_, ratio_sk**3)
                    empty_mask_sk = voxel_label_coarse_sk.sum(-1) == 0
                    voxel_label_coarse_sk = voxel_label_coarse_sk.to(torch.int64)
                    occ_space_sk = voxel_label_coarse_sk[~empty_mask_sk]
                    occ_space_sk[occ_space_sk==0] = -torch.arange(len(occ_space_sk[occ_space_sk==0])).to(occ_space_sk.device) - 1
                    voxel_label_coarse_sk[~empty_mask_sk] = occ_space_sk
                    voxel_label_coarse_sk = torch.mode(voxel_label_coarse_sk, dim=-1)[0]
                    voxel_label_coarse_sk[voxel_label_coarse_sk<0] = 255
                    voxel_label_coarse_sk = voxel_label_coarse_sk.long()
                    with torch.cuda.amp.autocast(enabled=False):
                        logits_sk = logits_sk.float()

                        # igore 255 = ignore noise. we keep the loss bascward for the label=0 (free voxels)
                        loss_dict['loss_voxel_ce_2'] = self.loss_weight[0] * self.ce_loss_func(logits_sk, voxel_label_coarse_sk, self.class_weights_sk.type_as(logits_sk), ignore_index=255)
                        loss_dict['loss_voxel_sem_scal_2'] = self.loss_weight[2] * sem_scal_loss(logits_sk, voxel_label_coarse_sk, ignore_index=255)
                        loss_dict['loss_voxel_geo_scal_2'] = self.loss_weight[3] * geo_scal_loss(logits_sk, voxel_label_coarse_sk, ignore_index=255, non_empty_idx=0)
                        loss_dict['loss_voxel_lovasz_2'] = self.loss_weight[1] * self.lovasz_loss_func(torch.softmax(logits_sk, dim=1), voxel_label_coarse_sk, ignore=255)
                else:
                    with torch.cuda.amp.autocast(enabled=False):
                        logits_sk = logits_sk.float()

                        # igore 255 = ignore noise. we keep the loss bascward for the label=0 (free voxels)
                        loss_dict['loss_voxel_ce_2'] = self.loss_weight[0] * self.ce_loss_func(logits_sk, voxel_label_sk, self.class_weights_sk.type_as(logits_sk), ignore_index=255)
                        loss_dict['loss_voxel_sem_scal_2'] = self.loss_weight[2] * sem_scal_loss(logits_sk, voxel_label_sk, ignore_index=255)
                        loss_dict['loss_voxel_geo_scal_2'] = self.loss_weight[3] * geo_scal_loss(logits_sk, voxel_label_sk, ignore_index=255, non_empty_idx=0)
                        loss_dict['loss_voxel_lovasz_2'] = self.loss_weight[1] * self.lovasz_loss_func(torch.softmax(logits_sk, dim=1), voxel_label_sk, ignore=255)
            return loss_dict
        else:
            if not self.dual:
                B_, W_, H_, D_ = voxel_label.shape
                pred = F.interpolate(logits, size=[W_, H_, D_], mode='trilinear', align_corners=False).contiguous()
                res = {
                    'output_voxels': pred,
                }
            elif (isinstance(dataset_flag, list) and len(dataset_flag) == 1) or (all(flag.item() == dataset_flag[0].item() for flag in dataset_flag)):
                flag = dataset_flag[0]
                if flag.item() == 1:
                    B_, W_, H_, D_ = voxel_label.shape
                    pred_nu = F.interpolate(logits_nu, size=[W_, H_, D_], mode='trilinear', align_corners=False).contiguous()
                    res = {
                        'output_voxels_1': pred_nu,
                    }
                if flag.item() == 2:
                    B_, W_, H_, D_ = voxel_label.shape
                    pred_sk = F.interpolate(logits_sk, size=[W_, H_, D_], mode='trilinear', align_corners=False).contiguous()
                    res = {
                        'output_voxels_2': pred_sk,
                    }
            else:
                voxel_label_nu = voxel_label[::2]
                voxel_label_sk = voxel_label[1::2]
                B1_, W1_, H1_, D1_ = voxel_label_nu.shape
                pred_nu = F.interpolate(logits_nu, size=[W1_, H1_, D1_], mode='trilinear', align_corners=False).contiguous()
                res = {
                    'output_voxels_1': pred_nu,
                }
                B2_, W2_, H2_, D2_ = voxel_label_sk.shape
                pred_sk = F.interpolate(logits_sk, size=[W2_, H2_, D2_], mode='trilinear', align_corners=False).contiguous()
                res = {
                    'output_voxels_2': pred_sk,
                }
            return res

class SensorEncoder(nn.Module):
    """Encode the raw sensor parameters into an embedding vector."""
    def __init__(self, in_dim=5, emb_dim=128, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, emb_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, sensor_params):
        # sensor_params: [B, in_dim]
        return self.mlp(sensor_params)  # [B, emb_dim]

class SensorHead(nn.Module):
    """Predict the sensor parameters from the global feature."""
    def __init__(self, feat_dim, out_dim=5, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, global_feat):
        # global_feat: [B, feat_dim]
        return self.mlp(global_feat)  # [B, out_dim]

class FiLMLayer(nn.Module):
    """
    FiLM: given a sensor embedding e, generate gamma, beta,
    and apply a channel-wise affine transform to the feature map:
        F' = (1 + gamma(e)) * F + beta(e)
    Supports x of shape [B, C, L] or [B, C, H, W].
    """
    def __init__(self, sensor_emb_dim, num_channels, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(sensor_emb_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * num_channels),
        )
        self.num_channels = num_channels

    def forward(self, x, sensor_emb):
        """
        x: [B, C, L] or [B, C, H, W]
        sensor_emb: [B, sensor_emb_dim]
        """
        B, C = x.shape[0], x.shape[1]
        assert C == self.num_channels, "FiLM channel count must match"

        gamma_beta = self.fc(sensor_emb)           # [B, 2C]
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)  # each [B, C]

        if x.dim() == 3:
            # [B, C, L]
            gamma = gamma.unsqueeze(-1)  # [B, C, 1]
            beta = beta.unsqueeze(-1)
        elif x.dim() == 4:
            # [B, C, H, W]
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
            beta = beta.unsqueeze(-1).unsqueeze(-1)
        else:
            raise NotImplementedError(f"Unsupported x dims: {x.shape}")

        return x * (1 + gamma) + beta

@VOXEL_ENCODERS.register_module()
class TPVAggregator_Seg(BaseModule):
    def __init__(
        self, tpv_h, tpv_w, tpv_z, nbr_classes=20,
        in_dims=64, hidden_dims=128, out_dims=None,
        scale_h=2, scale_w=2, scale_z=2, use_checkpoint=False,
        loss_weight=[1,1], dual = False,
        # Named static loss weights (the single entry point). When None, equivalent values are derived from the legacy
        # loss_weight / sensor_loss_weight, so the behaviour of old configs is unchanged.
        loss_weights=None,
        # Explicit gain of the consistency loss (not part of GradNorm):
        # effective scale = consistency_gain × T² (inside the function) × hook warmup schedule.
        # The default 40.0 exactly matches the effective magnitude of the historical outer_scale=40.
        consistency_gain=40.0,
        # NEW: sensor-related hyperparameters
        sensor_in_dim=5,
        sensor_emb_dim=256,
        sensor_loss_weight=0.6,
        consistency = False,
        consistency_loss_weight = 0.0,
        dense_train = False,
        d2skd = False,
        outside_d2skd = False,
        # Signal-field resolution of the high-frequency spectral distillation: the voxel logits are bilinearly grid_sampled from
        # the TPV planes, so on the fine grid (scale_w×) the octave above the TPV Nyquist is 100% interpolation artifact and
        # carries no network information. By default the distillation field is built at the native TPV resolution (ds=scale_w), which
        # cuts compute by scale^3=8x (measured: high-frequency distillation 896→132 ms/scene). Set 1 to fall back to the fine grid (ablation).
        distill_res_scale = None,
        # High-frequency distiller hyperparameters (None = use the class defaults). If ecc_block is not given, it is scaled
        # proportionally with distill_res_scale so that the **physical** extent of a block matches (32,32,16) on the fine grid.
        highfreq_distiller_cfg = None,
        # [Ablation switch] Master switch of the HF branch (spectral distillation of high-frequency voxel logits).
        # Two things must be known, otherwise the ablation runs the wrong thing:
        #  (1) setting highfreq_distiller_cfg to None does **not** turn this branch off -- the construction site is
        #      `dict(highfreq_distiller_cfg or {})`, so None degenerates to construction with all default parameters.
        #  (2) dark knowledge (temperature KL) is **not** inside the HF branch; it is an independent term in parallel
        #      (loss_kd_logit; static λ unless the config's rho contains 'kd' -- as in the released configs, where it is
        #      governed by the gradient-ratio controller, see pointtpv_seg.py). So "turning HF off" does not automatically
        #      turn dark knowledge off; it must be bound explicitly -- when this switch is False it also forces
        #      hybrid_dark_kd=False, so that the LF-only arm does not accidentally carry dark knowledge.
        # Definition of the two ablation arms:
        #   LF-only = LF only (no HF, no dark knowledge) → enable_hf_distill=False
        #   HF-only = HF + dark knowledge                 → enable_hf_distill=True, enable_lf_distill=False
        enable_hf_distill = True,
        # Choice of the distillation loss (the signal path is fully shared, only the loss changes; see the vanillakd.py module docstring):
        #  'spectral'   high-frequency spectral distillation (existing path, default)
        #  'vanilla_kd' plain-KD baseline of the paper: temperature KL on the same _query_distill_field distillation field
        # Derived from the config's KDbaseline (bool); the two are mutually exclusive.
        distill_mode = 'spectral',
        kd_cfg = None,          # dict(T=, lambda_kd=, lambda_feat=), read by vanilla_kd / hybrid
        pvkd_cfg = None,        # dict(...), passed to PVKD when distill_mode='pvkd'; ignored in the other modes
        # Hybrid dark knowledge: in spectral mode, a temperature-KL soft-label distillation additionally runs in parallel (exactly the
        # same as vanilla_kd: same support set, λ=kd_cfg['lambda_kd']; static λ unless the config's rho contains 'kd' -- as in the
        # released configs, where it is governed by the gradient-ratio controller, see pointtpv_seg.py). The spectral term only
        # transfers the frequency structure of the logit map and discards the per-voxel class probabilities; with a strong teacher,
        # dark knowledge is a gold mine for rare classes. The previous version EMA-normalized the dark knowledge and squeezed it into the
        # spectral rho budget (magnitude 0.03 vs 0.8 for vanilla, 30x weaker, measured to be ineffective); this version computes it
        # independently at its natural magnitude (when rho contains 'kd', its weight comes from its own rho_kd group, not from the
        # spectral budget). When True: spectral → spectral + dark knowledge (vox path); the inter path stays purely spectral.
        hybrid_dark_kd = False,
        sensor_film = False,
        # Gradient/memory mode of the two CBA-CL views:
        #  'alternate': in each iter one view carries gradients and is supervised (student), the other view
        #    provides the consistency target under no_grad (teacher); they swap every iter. Memory ≈ a single view.
        #  'both': gradients flow back through both graphs (old behaviour), memory ≈ 2x.
        consistency_grad_mode='alternate',
        # In 'both' mode, whether the auxiliary view also computes the supervised loss (alternate mode naturally supervises both across iterations)
        supervise_aux_views=False,
        # The consistency loss has no parameters inside; checkpoint recomputation saves ~1.5GB of dense intermediate activations
        checkpoint_consistency_loss=True,
    ):
        super().__init__()
        self.tpv_h = tpv_h
        self.tpv_w = tpv_w
        self.tpv_z = tpv_z
        self.scale_h = scale_h
        self.scale_w = scale_w
        self.scale_z = scale_z
        self.loss_weight = loss_weight
        self.dual = dual
        self.consistency = consistency
        self.dense_train = dense_train
        self.d2skd = d2skd
        self.outside_d2skd = outside_d2skd
        self.consistency_loss_weight = consistency_loss_weight
        self.consistency_grad_mode = consistency_grad_mode
        self.supervise_aux_views = bool(supervise_aux_views)
        self.checkpoint_consistency_loss = bool(checkpoint_consistency_loss)
        out_dims = in_dims if out_dims is None else out_dims
        self.distill_res_scale = int(distill_res_scale) if distill_res_scale else int(scale_w)
        self.distill_mode = distill_mode
        assert distill_mode in ('spectral', 'vanilla_kd', 'cwd', 'pvkd'), distill_mode
        # Hybrid only makes sense in spectral mode (vanilla_kd mode already runs the logit KL)
        self.enable_hf_distill = bool(enable_hf_distill)
        # Dark knowledge is explicitly bound to the HF master switch: it is an independent parallel term and cannot be turned off without this binding (see the comment above).
        self.hybrid_dark_kd = (bool(hybrid_dark_kd) and distill_mode == 'spectral'
                               and self.enable_hf_distill)
        if self.outside_d2skd:
            if distill_mode == 'spectral' and not self.enable_hf_distill:
                # Ablation: HF and dark knowledge are both off; neither of the keys loss_distill_vox / loss_kd_logit
                # appears. Whether a key exists depends only on the config (identical on all ranks), so DDP stays aligned.
                self.spectral_distiller = None
            elif distill_mode == 'spectral':
                hf_cfg = dict(highfreq_distiller_cfg or {})
                if 'ecc_block' not in hf_cfg:
                    ds = max(1, self.distill_res_scale)
                    hf_cfg['ecc_block'] = (max(2, 32 // ds), max(2, 32 // ds), max(2, 16 // ds))
                self.spectral_distiller = TPVHighFreqSpectralDistiller(**hf_cfg)
                if self.hybrid_dark_kd:
                    from .vanillakd import VanillaKD
                    # Same instance semantics as the vanilla_kd baseline; only its loss_logit (dark knowledge) is used, the feature MSE does not take part
                    self.vanilla_kd = VanillaKD(**(kd_cfg or {}))
            elif distill_mode == 'vanilla_kd':
                from .vanillakd import VanillaKD
                self.vanilla_kd = VanillaKD(**(kd_cfg or {}))
            elif distill_mode == 'pvkd':
                # PVKD voxel side: voxel-level temperature KL + inter-voxel SP affinity (see pvkd.py)
                from .pvkd import PVKD
                self.pvkd = PVKD(**(pvkd_cfg or {}))
            # distill_mode == 'cwd': by design, CWD acts only on the TPV feature planes;
            # there is no component and no loss term on the voxel side, so nothing is instantiated here.
        self.decoder = nn.Sequential(
            nn.Linear(in_dims, hidden_dims),
            nn.Softplus(),
            nn.Linear(hidden_dims, out_dims)
        )
        self.classifier = nn.Linear(out_dims, nbr_classes)
        self.classes = nbr_classes
        self.use_checkpoint = use_checkpoint
        self.ce_loss_func = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.lovasz_loss_func = lovasz_softmax

        # Initialize the dynamically accumulated class_counts
        self.register_buffer("class_counts",
            torch.ones(self.classes)  # initialized to 1 to avoid division by zero
        )
        # NEW: sensor encoder / prediction head / FiLM layer
        self.sensor_in_dim = sensor_in_dim
        self.sensor_emb_dim = sensor_emb_dim
        self.sensor_loss_weight = sensor_loss_weight
        # Expansion of the named static weights (when loss_weights is absent, the legacy values are mapped equivalently:
        # voxel_lovasz = loss_weight[1]×0.5 corresponds to the 0.5 hidden in the old 'loss_voxel_ce')
        _lw = loss_weights if isinstance(loss_weights, dict) else {}
        self.w_point_lovasz = float(_lw.get('point_lovasz', self.loss_weight[0]))
        self.w_point_seesaw = float(_lw.get('point_seesaw', self.loss_weight[0]))
        self.w_voxel_lovasz = float(_lw.get('voxel_lovasz', self.loss_weight[1] * 0.5))
        self.w_voxel_seesaw = float(_lw.get('voxel_seesaw', self.loss_weight[1]))
        self.w_sensor = float(_lw.get('sensor', sensor_loss_weight))
        self.consistency_gain = float(_lw.get('consistency_gain', consistency_gain))
        self.sensor_film = sensor_film
        # sensor_head extracts the global feature with "multi-statistic pooling" (mean⊕std⊕max) rather than a single average pooling:
        # sensor properties such as beam count / resolution show up in the distribution (density) of points/voxels along the L dim, which
        # average pooling smooths out, while std/max retain the information sensitive to sampling density. Hence global_feat has _sensor_feat_mult*in_dims dims.
        self._sensor_feat_mult = 3
        if not self.dense_train and self.sensor_film:
            # encodes the raw sensor vector into an embedding
            self.sensor_encoder = SensorEncoder(
                in_dim=sensor_in_dim,
                emb_dim=sensor_emb_dim,
                hidden_dim=sensor_emb_dim
            )
            # predicts the sensor parameters from the global feature (global_feat comes from multi-statistic pooling of fused)
            self.sensor_head = SensorHead(
                feat_dim=in_dims * self._sensor_feat_mult,
                out_dim=sensor_in_dim,
                hidden_dim=sensor_emb_dim
            )
            # FiLM modulation of the fused feature [B, C, L]
            self.film_layer = FiLMLayer(
                sensor_emb_dim=sensor_emb_dim,
                num_channels=in_dims,
                hidden_dim=sensor_emb_dim
            )
        # NEW: FiLM mixing coefficient alpha; 0 means pure GT, 1 means pure Pred.
        # Stored as a buffer (read/written through the property below), so resuming from a checkpoint no longer silently resets it to 0;
        # old checkpoints lacking this key keep 0 under non-strict loading, and the hook quickly pulls it back.
        self.register_buffer('_film_alpha_buf', torch.zeros(1))
        # The initial values match the first curriculum phase (0-8); the hook updates them per schedule in before_train_epoch
        # phase0: beams(32,64), height(1.7,1.9), H_res(900,2400), theta_low(-25,-20), theta_up(0,10)
        # mean=(a+b)/2, std=(b-a)/√12
        self.register_buffer('sensor_mean', torch.tensor([48.0, 1.8, 1650.0, -22.5, 5.0], dtype=torch.float32))
        self.register_buffer('sensor_std', torch.tensor([9.24, 0.058, 433.0, 1.44, 2.89], dtype=torch.float32))

    # NEW: provides an interface to adjust the FiLM alpha during training,
    # e.g. a hook sets alpha from 0->1 over different epochs
    @property
    def film_alpha(self):
        return float(self._film_alpha_buf.item())

    @film_alpha.setter
    def film_alpha(self, value):
        self._film_alpha_buf.fill_(float(value))

    def set_film_alpha(self, alpha: float):
        self.film_alpha = float(alpha)

    def update_sensor_stats(self, mean, std):
        self.sensor_mean = mean.to(self.sensor_mean.device)
        self.sensor_std = std.to(self.sensor_std.device)

    def _sensor_global_feat(self, fused):
        """Global feature for sensor prediction: concatenation of three statistics, mean ⊕ std ⊕ max.

        A single average pooling smooths out the distribution of points/voxels along the L dim (which is exactly what carries
        sensor properties such as beam count / resolution); std/max are more sensitive to sampling density and retain this information.
        fused: [B, C, L] -> [B, _sensor_feat_mult*C] (3C here).
        """
        mean_feat = fused.mean(dim=-1)
        std_feat = fused.std(dim=-1, unbiased=False)   # biased std avoids nan when L=1
        max_feat = fused.amax(dim=-1)
        return torch.cat([mean_feat, std_feat, max_feat], dim=1)

    def seesaw_loss(self, logits, labels, p=0.8, eps=1e-6, ignore_index=0):
        """
        logits: any shape, as long as dim=1 is the channel (C)
        labels: semantic labels corresponding to logits, any shape, as long as it matches the spatial dims of logits
        """
        # ---- reshape: (B, C, ...) → (N, C) ----
        B, C = logits.shape[:2]
        logits_flat = logits.permute(0, *range(2, logits.ndim), 1).reshape(-1, C)
        # logits_flat shape = (N, C)
        # ---- labels reshape: (B, ...) → (N,) ----
        labels_flat = labels.reshape(-1)
        # ---- mask ignore_index ----
        valid = labels_flat != ignore_index
        if not valid.any():
            # no valid points → return 0 loss
            return logits.sum() * 0
        logits_flat = logits_flat[valid]
        labels_flat = labels_flat[valid]
        # ---- one-hot ----
        targets = F.one_hot(labels_flat.long(), num_classes=C).float()
        def distribution_agnostic_seesaw_loss(logits, targets, p=0.8, eps=1e-6):
            # class-frequency statistics
            batch_class_count = targets.sum(dim=0)
            self.class_counts += batch_class_count
            cc = self.class_counts
            # s matrix
            conditions = cc[:, None] > cc[None, :]
            trues = (cc[None, :] / cc[:, None]).pow(p)
            falses = torch.ones_like(trues)
            s = torch.where(conditions, trues, falses)
            # overflow protection
            logits = logits - logits.max(dim=-1, keepdim=True)[0]
            numerator = torch.exp(logits)
            denominator = (
                (1 - targets)[:, None, :] * s[None, :, :] * torch.exp(logits)[:, None, :]
            ).sum(dim=-1) + torch.exp(logits)
            sigma = numerator / (denominator + eps)
            loss = (-targets * torch.log(sigma + eps)).sum(dim=-1)
            return loss.mean()
        # ---- seesaw loss ----
        return distribution_agnostic_seesaw_loss(logits_flat, targets, p=p, eps=eps)

    def forward(self, tpv_list, points=None, voxel_label=None, point_labels=None, dataset_flag=None, return_loss=True, sensor_vec=None, logits_vox_t=None, logits_pts_t=None, skip_consistency=False, return_logits_for_consistency=False, distill_points=None, return_distill_field=False):
        """
        x y z -> w h z
        tpv_list[0]: bs, c, w, h
        tpv_list[1]: bs, c, h, z
        tpv_list[2]: bs, c, z, w

        Args:
            logits_vox_t: voxel predictions of the teacher model, used for knowledge distillation
            logits_pts_t: point predictions of the teacher model, used for knowledge distillation
            skip_consistency: when True, skip the internal consistency branch (computed group by group in sequence by the caller)
            distill_points: the teacher's dense points, used to query the student on the teacher's voxel support set
                to obtain a spectral distillation field with the same support set (see the _forward_single docstring)
        """
        # When d2skd=True, force return_loss=False (only output predictions)
        if self.d2skd:
            return_loss = False

        # Run the consistency logic only during training (sensor_vec is not None) and when not skipped
        if self.consistency and sensor_vec is not None and not skip_consistency:
            if isinstance(points, list):
                points_list = points
            else:
                # If points is a tensor, check its batch size;
                # if the batch size > 1, it must be split into a list so that each element corresponds to one sample
                if isinstance(points, torch.Tensor) and points.dim() >= 2 and points.size(0) > 1:
                    # split the tensor into a list along the batch dim
                    points_list = [points[i:i+1] for i in range(points.size(0))]
                else:
                    points_list = [points]
            num_samples = len(points_list)

            # Run sample by sample and collect voxel logits for consistency
            losses, voxel_logits, point_logits = [], [], []
            sensor_outs = []
            voxel_labels_for_consistency = []

            # helper that extracts an element from a container
            def _get(container, idx):
                if container is None:
                    return None
                if isinstance(container, list):
                    return container[idx]
                if isinstance(container, torch.Tensor) and container.dim() > 0 and container.size(0) > idx:
                    return container[idx:idx+1]
                return container

            for i in range(num_samples):
                # Convention: samples come in pairs (0,1), (2,3), (4,5), ...
                # The first sample of each pair is the main sample and takes part in the supervised loss and backprop;
                # the second sample of the pair is only used for consistency regularization (inference mode, no backprop).
                is_main_view = (i % 2 == 0)

                tpv_xy_i, tpv_yz_i, tpv_zx_i = [
                    feat.index_select(0, torch.tensor([i], device=feat.device)) for feat in tpv_list
                ]
                pts_i = points_list[i]
                # Make sure pts_i has shape [1, n, 3]. Note that 2-D [n,3] is "a single frame with n points",
                # so a batch dim must be added as a whole; the old code checked size(0)>1 first and mistook it for
                # "batch>1", truncating it to a single point [1,3], which made the point-level loss degenerate/crash.
                if isinstance(pts_i, torch.Tensor) and pts_i.dim() == 2:
                    pts_i = pts_i.unsqueeze(0)
                elif isinstance(pts_i, torch.Tensor) and pts_i.dim() >= 3 and pts_i.size(0) > 1:
                    # genuine batch>1: a per-view call only takes the first sample corresponding to this view
                    pts_i = pts_i[0:1]
                voxel_label_i = _get(voxel_label, i)
                point_labels_i = _get(point_labels, i)
                sensor_vec_i = _get(sensor_vec, i)
                dataset_flag_i = _get(dataset_flag, i)

                # extract the corresponding teacher predictions (used only for the main sample)
                logits_vox_t_i = None
                logits_pts_t_i = None
                if is_main_view:
                    if logits_vox_t is not None:
                        if isinstance(logits_vox_t, list):
                            logits_vox_t_i = logits_vox_t[i] if i < len(logits_vox_t) else None
                        else:
                            logits_vox_t_i = logits_vox_t[i:i+1] if logits_vox_t is not None else None
                    if logits_pts_t is not None:
                        if isinstance(logits_pts_t, list):
                            logits_pts_t_i = logits_pts_t[i] if i < len(logits_pts_t) else None
                        else:
                            logits_pts_t_i = logits_pts_t[i:i+1] if logits_pts_t is not None else None

                if is_main_view:
                    # main sample: compute the full supervised loss and take part in backprop
                    out_i = self._forward_single(
                        [tpv_xy_i, tpv_yz_i, tpv_zx_i],
                        points=pts_i,
                        voxel_label=voxel_label_i,
                        point_labels=point_labels_i,
                        dataset_flag=dataset_flag_i,
                        return_loss=return_loss,
                        sensor_vec=sensor_vec_i,
                        return_logits=True,
                        logits_vox_t=logits_vox_t_i,
                        logits_pts_t=logits_pts_t_i,
                    )

                    if return_loss:
                        loss_i, vox_i, pts_out_i, sensor_out_i = out_i
                        losses.append(loss_i)
                        voxel_logits.append(vox_i)
                        point_logits.append(pts_out_i)
                        sensor_outs.append(sensor_out_i)
                        voxel_labels_for_consistency.append(voxel_label_i)
                    else:
                        vox_i, pts_out_i, sensor_out_i = out_i
                        voxel_logits.append(vox_i)
                        point_logits.append(pts_out_i)
                        sensor_outs.append(sensor_out_i)
                elif self.supervise_aux_views and return_loss:
                    # Auxiliary samples also compute the supervised loss: views pulled towards extreme configs by maximin
                    # are no longer shaped by the consistency loss alone. Note that voxel_label / point_labels /
                    # sensor are all per view and can be used directly.
                    out_i = self._forward_single(
                        [tpv_xy_i, tpv_yz_i, tpv_zx_i],
                        points=pts_i,
                        voxel_label=voxel_label_i,
                        point_labels=point_labels_i,
                        dataset_flag=dataset_flag_i,
                        return_loss=True,
                        sensor_vec=sensor_vec_i,
                        return_logits=True,
                        logits_vox_t=None,
                        logits_pts_t=None,
                    )
                    loss_i, vox_i, pts_out_i, sensor_out_i = out_i
                    losses.append(loss_i)
                    voxel_logits.append(vox_i)
                    point_logits.append(pts_out_i)
                    sensor_outs.append(sensor_out_i)
                else:
                    # auxiliary sample: no supervised loss, but it takes part in the backprop of the consistency loss
                    # (in CBA-CL 'both' mode both views carry gradients, so torch.no_grad is not used here)
                    out_i = self._forward_single(
                        [tpv_xy_i, tpv_yz_i, tpv_zx_i],
                        points=pts_i,
                        voxel_label=None,          # no supervised voxel loss
                        point_labels=None,         # no supervised point loss
                        dataset_flag=dataset_flag_i,
                        return_loss=False,         # logits only
                        sensor_vec=sensor_vec_i,
                        return_logits=True,
                        logits_vox_t=None,
                        logits_pts_t=None,
                    )
                    vox_i, pts_out_i, sensor_out_i = out_i
                    voxel_logits.append(vox_i)
                    point_logits.append(pts_out_i)
                    sensor_outs.append(sensor_out_i)

            if not return_loss:
                return voxel_logits, point_logits, sensor_outs

            if len(losses) == 0:
                agg_loss = {}
                # defensive: if there is no loss, a device is still needed
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                # collect all possible keys
                all_keys = set()
                for ld in losses:
                    all_keys.update(ld.keys())
                all_keys.add('loss_consistency')

                # [FIX 1] Key change: sort the keys so that the dict insertion order is the same on all ranks;
                # otherwise the order of dict.values() may differ between ranks and break all_reduce.
                # Note: during training the loss keys are usually identical on all ranks, so only sorting is done here;
                # if keys really need to be synchronized, a simpler method can be used (e.g. a file, or assuming consistency).
                sorted_keys = sorted(list(all_keys))

                # Optional: a barrier can be used to make sure all ranks have the same keys;
                # but keys should normally be identical during training, so the sorted keys are used directly here
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()  # make sure all ranks reach this point

                # initialize agg_loss
                agg_loss = {}
                device = losses[0][list(losses[0].keys())[0]].device

                # [FIX 2] iterate over the sorted sorted_keys
                for k in sorted_keys:
                    if k == 'loss_consistency':
                        # placeholder, overwritten later. requires_grad=True keeps DDP checks from complaining (even though it is overwritten)
                        agg_loss[k] = torch.tensor(0.0, device=device, requires_grad=True)
                    else:
                        sample_tensor = None
                        for ld in losses:
                            if k in ld:
                                sample_tensor = ld[k]
                                break
                        if sample_tensor is not None:
                            agg_loss[k] = sample_tensor.new_zeros(())
                        else:
                            agg_loss[k] = torch.zeros((), device=device)

                # aggregate the losses
                for ld in losses:
                    for k in sorted_keys: # also use sorted_keys; not strictly necessary but keeps the habit consistent
                        if k != 'loss_consistency' and k in ld:
                            agg_loss[k] = agg_loss[k] + ld[k]

                # Average: normalize by the "number of main samples (i.e. the number of losses)",
                # not by the number of all views, to avoid the overall loss being wrongly scaled down with repeated sampling.
                for k in agg_loss:
                    if k != 'loss_consistency':
                        agg_loss[k] = agg_loss[k] / (len(losses) + 1e-8)  # avoid division by zero
            # T: temperature.
            # T > 1 smooths the distribution and focuses on consistency of the overall distribution (Dark Knowledge).
            # T = 1 is the most standard setting. Start with 1.0 and increase it if a stronger smoothing constraint is desired. The released configs use T = 1.5 (below).
            T = 1.5
            # initialize the consistency loss so that all ranks have this key
            if len(voxel_logits) > 0:
                device = voxel_logits[0].device
                loss_consistency_val = voxel_logits[0].sum() * 0.0
            else:
                # if there are no samples, create a 0 tensor with grad to avoid DDP unused-parameter errors
                if 'device' not in locals():
                     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                loss_consistency_val = torch.tensor(0.0, device=device, requires_grad=True)
            # compute only when there are enough samples or samples need to be duplicated
            if num_samples == 1 and len(voxel_logits) > 0:
                voxel_logits.append(voxel_logits[0].detach())
                voxel_labels_for_consistency.append(voxel_labels_for_consistency[0])
                num_samples = 2 # update the local count
            if len(voxel_logits) > 1:
                # Same implementation as the external _cb_adc_consistency_loss; both views take part in backprop.
                cl_weight = float(self.consistency_loss_weight)
                cl_total = loss_consistency_val.new_zeros(())
                pairs = 0
                # computed only within each pair: (0,1), (2,3), (4,5)...; no cross-group comparison
                for i in range(0, len(voxel_logits) - 1, 2):
                    j = i + 1
                    # the main sample has voxel_label, the auxiliary sample does not; the group shares the main sample's label
                    group_idx = i // 2
                    vi = voxel_labels_for_consistency[group_idx] if group_idx < len(voxel_labels_for_consistency) else None
                    if vi is None:
                        continue
                    # effective scale = consistency_gain (static) × cl_weight (hook warmup) × raw (T² included)
                    cl_total = cl_total + self.consistency_gain * cl_weight * cb_adc_consistency_loss_ckpt(
                        voxel_logits[i],
                        voxel_logits[j],
                        vi,
                        num_classes=self.classes,
                        T=T,
                        ignore_index=0,
                        use_checkpoint=self.checkpoint_consistency_loss,
                    )
                    pairs += 1

                if pairs > 0:
                    loss_consistency_val = loss_consistency_val + cl_total / pairs

            # final assignment
            agg_loss['loss_consistency'] = loss_consistency_val
            return agg_loss

        # defensive check: make sure points is a tensor rather than a list
        if isinstance(points, list):
            if len(points) == 0:
                raise ValueError("points list is empty")
            if all(pl.shape == points[0].shape for pl in points):
                points = torch.cat([pl[None, ...] if pl.dim() == 2 else pl for pl in points], dim=0)
            else:
                points = points[0].unsqueeze(0) if points[0].dim() == 2 else points[0]

        return self._forward_single(
            tpv_list, points, voxel_label, point_labels, dataset_flag, return_loss, sensor_vec,
            return_logits=return_logits_for_consistency, logits_vox_t=logits_vox_t, logits_pts_t=logits_pts_t,
            distill_points=distill_points, return_distill_field=return_distill_field
        )

    def _query_distill_field(self, tpv_hw, tpv_zh, tpv_wz, distill_points, bs, sensor_emb=None):
        """Query TPV features on the teacher's voxel support set and produce a coarse-grid distillation field [bs,C,gw,gh,gz].

        Built at the **native TPV resolution** rather than on the fine grid used for supervision: the voxel logits are bilinearly
        grid_sampled from the TPV planes, so the octave above the TPV Nyquist on the fine grid is 100% interpolation
        artifact and contains no network information -- spectral distillation (Laplacian/diffusion/ECC) there would amount to aligning
        interpolation artifacts while paying scale^3=8x the compute for it (measured: high-frequency distillation 896 → 132 ms/scene).
        distill_res_scale=1 falls back to the fine grid (for ablation).

        Not merged into `fused`: sensor_head infers the sensor parameters from the "sampling density distribution" via the
        multi-statistic pooling (mean⊕std⊕max) of fused, and mixing in dense voxel columns would contaminate these statistics. Only the
        same TPV features are borrowed here, going through a separate FiLM (same condition vector) + decoder + classifier.

        Teacher and student call the same method with the same distill_points, so the grids and support sets of the two spectral fields
        agree voxel by voxel (this is the "same frequency basis for teacher and student" precondition required by the SI-STJD transport model M).
        """
        ds = int(getattr(self, 'distill_res_scale', 0)) or self.scale_w
        gw = int(self.tpv_w * self.scale_w // ds)
        gh = int(self.tpv_h * self.scale_h // ds)
        gz = int(self.tpv_z * self.scale_z // ds)

        dp = distill_points
        if dp.dim() == 2:
            dp = dp.unsqueeze(0)
        if dp.shape[0] > bs:
            dp = dp[0:1]
        elif dp.shape[0] < bs and dp.shape[0] == 1:
            dp = dp.expand(bs, -1, -1)
        dp = dp.to(dtype=tpv_hw.dtype).unsqueeze(1)                 # bs,1,Nd,3
        voxels_d = torch.unique(torch.floor(dp / ds), dim=2)
        voxels_d = torch.stack([
            voxels_d[..., 0].clamp(0, gw - 1),
            voxels_d[..., 1].clamp(0, gh - 1),
            voxels_d[..., 2].clamp(0, gz - 1)], dim=-1)
        voxels_d_ind = voxels_d.clone().type(torch.long).squeeze(1)
        # normalized coordinate of the coarse voxel center (j+0.5)/g*2-1, equivalent to the fine-grid center 2j+1
        voxels_d = voxels_d + 0.5
        voxels_d[..., 0] = voxels_d[..., 0] / gw * 2 - 1
        voxels_d[..., 1] = voxels_d[..., 1] / gh * 2 - 1
        voxels_d[..., 2] = voxels_d[..., 2] / gz * 2 - 1
        sl = voxels_d[:, :, :, [0, 1]]
        fd = F.grid_sample(tpv_hw, sl, padding_mode="border").squeeze(2)
        sl = voxels_d[:, :, :, [1, 2]]
        fd = fd + F.grid_sample(tpv_zh, sl, padding_mode="border").squeeze(2)
        sl = voxels_d[:, :, :, [2, 0]]
        fd = fd + F.grid_sample(tpv_wz, sl, padding_mode="border").squeeze(2)   # bs,C,Nd

        if sensor_emb is not None:
            fd = self.film_layer(fd, sensor_emb)
        fd = fd.permute(0, 2, 1)
        if self.use_checkpoint:
            fd = torch.utils.checkpoint.checkpoint(self.decoder, fd)
            lg_d = torch.utils.checkpoint.checkpoint(self.classifier, fd)
        else:
            fd = self.decoder(fd)
            lg_d = self.classifier(fd)                              # bs,Nd,classes
        out_shape = [gw, gh, gz, lg_d.shape[-1]]
        return torch.stack([
            scatter_nd(indices=voxels_d_ind[b], updates=lg_d[b],
                       shape=out_shape).permute(3, 0, 1, 2)
            for b in range(bs)], dim=0)                             # bs,C,gw,gh,gz

    def _forward_single(self, tpv_list, points, voxel_label, point_labels, dataset_flag, return_loss, sensor_vec, return_logits=False, logits_vox_t=None, logits_pts_t=None, distill_points=None, return_distill_field=False):
        """
        distill_points: the teacher's dense points (voxel grid coordinates). When given, the student's TPV features are
        additionally queried on **the teacher's voxel support set**, producing logits_vox_s_dense with the same support
        set as the teacher, for use by the high-frequency spectral distillation.

        Why this is necessary: logits_vox is written by scatter_nd into `unique(floor(points))`, so its
        support set is determined by the student's own sparse points; voxels that are not hit are structural zeros with **no
        gradient path** (see the same comment at the voxel supervision loss below). If it were spectrally aligned directly with the
        teacher's dense field, the two fields would be defined on different support sets: the neighbourhood coupling of the Laplacian would
        treat the artificial "covered/uncovered" cliff as high-frequency structure, comparing coverage differences rather than semantic structure,
        and the student could not learn at unobserved locations at all.
        The student's TPV features are dense convolutional feature maps that grid_sample can query at any location --
        switching the query indices to the teacher's support set makes the two spectral fields isomorphic, and the student gets a gradient
        at every voxel observed by the teacher.
        """
        tpv_xy, tpv_yz, tpv_zx = tpv_list[0], tpv_list[1], tpv_list[2]
        tpv_hw = tpv_xy.permute(0, 1, 3, 2)
        tpv_wz = tpv_zx.permute(0, 1, 3, 2)
        tpv_zh = tpv_yz.permute(0, 1, 3, 2)
        bs, c, _, _ = tpv_hw.shape

        # Align the batch dim of the labels, to prevent labels with batch=1 from mismatching features with a larger batch at data construction time.
        # The train_pts_label produced by the pipeline is 1-D [N] (PointoccMapping squeeze(-1)); when the consistency branch
        # passes it per view it has no batch dim, so add one first, otherwise the per-batch check below would always raise a false alarm
        if point_labels is not None and point_labels.dim() == 1:
            point_labels = point_labels.unsqueeze(0)
        if voxel_label is not None and voxel_label.dim() == 3:
            voxel_label = voxel_label.unsqueeze(0)
        if voxel_label is not None and voxel_label.shape[0] != bs:
            if voxel_label.shape[0] == 1:
                voxel_label = voxel_label.expand(bs, *voxel_label.shape[1:])
            else:
                raise ValueError(f"voxel_label batch ({voxel_label.shape[0]}) != feature batch ({bs})")
        if point_labels is not None and point_labels.shape[0] != bs:
            if point_labels.shape[0] == 1:
                point_labels = point_labels.expand(bs, *point_labels.shape[1:])
            else:
                raise ValueError(f"point_labels batch ({point_labels.shape[0]}) != feature batch ({bs})")

        if self.scale_h != 1 or self.scale_w != 1:
            tpv_hw = F.interpolate(
                tpv_hw,
                size=(int(self.tpv_h*self.scale_h), int(self.tpv_w*self.scale_w)),
                mode='bilinear'
            )
        if self.scale_z != 1 or self.scale_h != 1:
            tpv_zh = F.interpolate(
                tpv_zh,
                size=(int(self.tpv_z*self.scale_z), int(self.tpv_h*self.scale_h)),
                mode='bilinear'
            )
        if self.scale_w != 1 or self.scale_z != 1:
            tpv_wz = F.interpolate(
                tpv_wz,
                size=(int(self.tpv_w*self.scale_w), int(self.tpv_z*self.scale_z)),
                mode='bilinear'
            )

        # Fast path for teacher distillation: only the coarse-grid distillation field is needed, so the whole fine-grid computation is skipped.
        # Otherwise the teacher would run unique + grid_sample + decoder on ~560k dense points plus
        # a 177MB scatter to produce logits_vox / logits_pts -- while the caller only takes
        # the 4th return value (the distillation field) and discards the first three.
        if return_distill_field and distill_points is not None and not return_loss \
                and getattr(self, 'distill_field_only', True):
            return (None, None, None,
                    self._query_distill_field(tpv_hw, tpv_zh, tpv_wz,
                                              distill_points, bs, sensor_emb=None))

        # points: bs, n, 3  (a single sample n,3 automatically gets a batch dim)
        if points.dim() == 2:
            points = points.unsqueeze(0)
        points_bs, n, _ = points.shape

        # make sure the batch size of points matches that of tpv_hw;
        # if the batch size of points is larger than that of tpv_hw, take only the first sample
        if points_bs > bs:
            points = points[0:1]  # take only the first sample
            points_bs = 1
        elif points_bs < bs:
            # if the batch size of points is smaller than that of tpv_hw, expand points
            if points_bs == 1:
                points = points.expand(bs, -1, -1)
                points_bs = bs
            else:
                raise ValueError(f"points batch ({points_bs}) < tpv_hw batch ({bs}) and cannot be expanded")

        # The clone is required: `points[..., 0] = ...` below writes in place into a **view** produced by unsqueeze,
        # which would normalize the caller's point cloud to [-1,1] in place. On the d2skd path the same dense points
        # are first given to the teacher as points and then to the student as distill_points; without the clone the student
        # would receive already normalized coordinates, and after floor() the support set would collapse to 0/-1.
        points = points.unsqueeze(1).clone()  # bs,1,n,3
        voxels = torch.unique(torch.floor(points), dim=2)
        # Keep batch dimension; remove only the singleton channel dimension
        voxels_ind = deepcopy(voxels).type(torch.long).squeeze(1)

        voxels += 0.5
        points[..., 0] = points[..., 0] / (self.tpv_w*self.scale_w) * 2 - 1
        points[..., 1] = points[..., 1] / (self.tpv_h*self.scale_h) * 2 - 1
        points[..., 2] = points[..., 2] / (self.tpv_z*self.scale_z) * 2 - 1
        sample_loc = points[:, :, :, [0, 1]]
        tpv_hw_pts = F.grid_sample(tpv_hw, sample_loc, padding_mode="border").squeeze(2) # bs, c, n
        sample_loc = points[:, :, :, [1, 2]]
        tpv_zh_pts = F.grid_sample(tpv_zh, sample_loc, padding_mode="border").squeeze(2)
        sample_loc = points[:, :, :, [2, 0]]
        tpv_wz_pts = F.grid_sample(tpv_wz, sample_loc, padding_mode="border").squeeze(2)

        voxels[..., 0] = voxels[..., 0] / (self.tpv_w*self.scale_w) * 2 - 1
        voxels[..., 1] = voxels[..., 1] / (self.tpv_h*self.scale_h) * 2 - 1
        voxels[..., 2] = voxels[..., 2] / (self.tpv_z*self.scale_z) * 2 - 1

        sample_loc_vox = voxels[:, :, :, [0, 1]]
        tpv_hw_vox = F.grid_sample(tpv_hw, sample_loc_vox, padding_mode="border").squeeze(2) # bs, c, n
        sample_loc_vox = voxels[:, :, :, [1, 2]]
        tpv_zh_vox = F.grid_sample(tpv_zh, sample_loc_vox, padding_mode="border").squeeze(2)
        sample_loc_vox = voxels[:, :, :, [2, 0]]
        tpv_wz_vox = F.grid_sample(tpv_wz, sample_loc_vox, padding_mode="border").squeeze(2)
        fused_vox = tpv_hw_vox + tpv_zh_vox + tpv_wz_vox

        fused_pts = tpv_hw_pts + tpv_zh_pts + tpv_wz_pts
        fused = torch.cat([fused_vox, fused_pts], dim=-1) # bs, c, whz+n

        # The distillation-field query has been factored out into _query_distill_field (the teacher fast path reuses the same code,
        # guaranteeing that the grids and support sets of teacher and student agree voxel by voxel). This is only the student-side
        # call site, placed after sensor/FiLM (sensor_emb is needed), see below.
        # ---------------- Sensor head + FiLM branch begins ----------------
        sensor_loss = fused.new_tensor(0.0)
        pred_sensor_output = None  # used to return the predicted sensor parameters at test time
        sensor_emb = None          # the distillation support-set path reuses the same FiLM condition vector
        # error statistics for logging
        sensor_mae = fused.new_tensor(0.0)           # total MAE
        sensor_mae_norm = fused.new_tensor(0.0)      # total MAE after normalization
        sensor_mae_per_dim = fused.new_tensor([0.0, 0.0, 0.0, 0.0, 0.0])  # MAE of each dim
        # run only if the model has the relevant modules (defensive)
        # The sensor_vec of reset_random real-scan samples is a NaN sentinel (no GT sensor parameters); consistent with
        # inference, they take the "predicted sensor → FiLM" branch, so real data is trained end to end on the test-time path.
        has_sensor_gt = sensor_vec is not None
        if has_sensor_gt and torch.is_tensor(sensor_vec):
            has_sensor_gt = bool(torch.isfinite(sensor_vec).all().item())
        if hasattr(self, 'sensor_head') and hasattr(self, 'sensor_encoder') and not self.dense_train:
            # training phase: GT sensor_vec available
            if has_sensor_gt:
                # 1. normalize the GT sensor parameters
                sensor_gt = sensor_vec.to(self.sensor_mean.device).float()        # [B, D]
                if sensor_gt.dim() == 1:  # shape is [D]
                    sensor_gt = sensor_gt.unsqueeze(0)  # → [1, D]
                sensor_gt_norm = (sensor_gt - self.sensor_mean) / self.sensor_std # [B, D]
                # 2. extract the global feature from fused for predicting the sensor parameters
                # fused: [B, C, L]; multi-statistic pooling (mean⊕std⊕max) gives [B, 3C], preserving density information
                global_feat = self._sensor_global_feat(fused)  # [B, 3C]
                # 3. predict the normalized parameters
                pred_sensor_norm = self.sensor_head(global_feat)  # [B, D]
                # 4. de-normalize to the raw space and compute the regression loss
                pred_sensor_raw = pred_sensor_norm * self.sensor_std + self.sensor_mean  # [B, D]
                # --------- error statistics (for the logger) ---------
                # absolute error: [B, D]
                abs_err = (pred_sensor_raw - sensor_gt).abs()
                # scalar: mean MAE over all dims and the whole batch
                sensor_mae = abs_err.mean()
                sensor_mae_norm = (pred_sensor_norm - sensor_gt_norm).abs().mean()
                # a separate MAE for each dim (shape [D]), optional
                sensor_mae_per_dim = abs_err.mean(dim=0)  # [D]
                # compute the loss in the normalized space so that gradients across dims are more balanced; weight height(dim1)/theta_up(dim4)
                per_dim_loss = F.smooth_l1_loss(pred_sensor_norm, sensor_gt_norm, reduction='none')  # [B, D]
                # dim_weights = fused.new_tensor([2.0, 1.5, 3.0, 1.0, 1.0])   # beam, height, H_res, theta_low, theta_up
                sensor_loss = per_dim_loss.mean()
                # 5. Progressive mixing (in the normalized space)
                alpha = float(self.film_alpha)
                alpha = max(0.0, min(1.0, alpha))  # keep within bounds
                sensor_mix_norm = (1.0 - alpha) * sensor_gt_norm + alpha * pred_sensor_norm.detach()
                # 6. encode into an embedding for FiLM
                sensor_emb = self.sensor_encoder(sensor_mix_norm)  # [B, sensor_emb_dim]
                # 7. FiLM modulation of the fused feature
                # fused: [B, C, L]
                fused = self.film_layer(fused, sensor_emb)
                # the raw prediction can also be kept during training for analysis
                pred_sensor_output = pred_sensor_raw  # [B, D]
            # test / inference phase: no GT, only the predicted sensor parameters are used
            else:
                # 1. global feature (same as in training: multi-statistic pooling)
                global_feat = self._sensor_global_feat(fused)  # [B, 3C]
                # 2. predict the normalized parameters; clamp to ±3σ to prevent OOD inputs (e.g. feature extrapolation
                #    on real data) from producing FiLM condition vectors never seen during training
                pred_sensor_norm = self.sensor_head(global_feat).clamp(-3.0, 3.0)  # [B, D]
                # 3. de-normalize to get the prediction in the raw space (can be returned to the caller for analysis)
                pred_sensor_raw = pred_sensor_norm * self.sensor_std + self.sensor_mean  # [B, D]
                # 4. use the (normalized) prediction directly as the FiLM condition
                sensor_emb = self.sensor_encoder(pred_sensor_norm)  # [B, sensor_emb_dim]
                # 5. FiLM modulation
                fused = self.film_layer(fused, sensor_emb)
                pred_sensor_output = pred_sensor_raw  # [B, D]
        # ---------------- Sensor head + FiLM branch ends ----------------
        # continue with the original head pipeline
        # if sensor_vec is None, FiLM is skipped entirely, compatible with the old logic
        fused = fused.permute(0, 2, 1)
        if self.use_checkpoint:
            fused = torch.utils.checkpoint.checkpoint(self.decoder, fused)
            logits = torch.utils.checkpoint.checkpoint(self.classifier, fused)
        else:
            fused = self.decoder(fused)
            logits = self.classifier(fused)
        logits = logits.permute(0, 2, 1)

        # ---- distillation field (coarse grid, same support set as the teacher, every voxel has a gradient) ----
        logits_vox_s_dense = None
        if distill_points is not None:
            logits_vox_s_dense = self._query_distill_field(
                tpv_hw, tpv_zh, tpv_wz, distill_points, bs, sensor_emb)

        # Preserve batch dimension; shape: (bs, whz, classes)
        feats_vox = logits[:, :, :(-n)].permute(0,2,1)
        output_shape = [int(self.tpv_w*self.scale_w), int(self.tpv_h*self.scale_h), int(self.tpv_z*self.scale_z), feats_vox.shape[-1]]
        # Scatter per batch to build (bs, C, W, H, Z)
        batch_logits_vox = []
        for b in range(bs):
            scattered = scatter_nd(indices=voxels_ind[b], updates=feats_vox[b], shape=output_shape)  # (W,H,Z,C)
            batch_logits_vox.append(scattered.permute(3,0,1,2))  # (C,W,H,Z)
        logits_vox = torch.stack(batch_logits_vox, dim=0)
        # print(logits_vox.shape)
        # np.save("baseline_logits.npy", logits_vox.detach().float().cpu().numpy())
        logits_pts = logits[:, :, (-n):].reshape(bs, self.classes, n, 1, 1)

        if return_loss:
            # Always return fixed keys with Tensor values to keep DDP collectives aligned
            zero_point = logits_pts.sum() * 0
            zero_voxel = logits_vox.sum() * 0

            point_loss = zero_point
            voxel_loss = zero_voxel
            seesaw_loss_points = zero_point
            seesaw_loss_voxels = zero_voxel

            # Point-level loss
            if point_labels is not None:
                valid_mask = point_labels != 0  # Ignore label 0
                if valid_mask.any():
                    point_loss = self.lovasz_loss_func(torch.softmax(logits_pts, dim=1), point_labels, ignore=0)
                    # torch.argmax(torch.softmax(logits_pts, dim=1), dim=1).reshape(1, -1).detach().cpu().numpy().tofile("predicted_labels.bin")
                    seesaw_loss_points = self.seesaw_loss(logits_pts, point_labels, ignore_index=0)

            # Voxel-level loss
            if voxel_label is not None:
                # Supervision is only meaningful on voxels actually covered (written by scatter) by this view:
                # the logits of unwritten voxels are constantly all zero (softmax=uniform and no gradient path),
                # yet with dense GT such voxels make up 68-83% of the valid voxels, flooding Lovasz's global
                # sorting and diluting the seesaw class-frequency statistics. GT outside the coverage is set to ignore(0).
                with torch.no_grad():
                    vox_cov = logits_vox.detach().ne(0).any(dim=1)  # boolean intermediate, saves memory
                voxel_label_sup = torch.where(vox_cov, voxel_label, torch.zeros_like(voxel_label))
                valid_voxel_mask = voxel_label_sup != 0
                if valid_voxel_mask.any():
                    # use Lovasz-Softmax instead of CE as the main voxel-level loss
                    voxel_probs = torch.softmax(logits_vox, dim=1)
                    voxel_loss = self.lovasz_loss_func(voxel_probs, voxel_label_sup, ignore=0)
                    seesaw_loss_voxels = self.seesaw_loss(logits_vox, voxel_label_sup, ignore_index=0)

            # NEW: sensor regression loss
            sensor_loss_term = sensor_loss * self.w_sensor

            # NEW: high-frequency logits distillation loss (if teacher predictions are provided).
            # The raw magnitude is passed through; the weight is assigned uniformly by the gradient-ratio controller on the detector side.
            # Spectral alignment must be done on a field **with the same support set as the teacher** (see the _forward_single
            # docstring): the support set of logits_vox comes from the student's own sparse points, and uncovered locations are
            # gradient-free structural zeros, so aligning it directly would make the Laplacian compare coverage differences.
            distill_loss_vox = None
            kd_logit_loss = None
            kd_logit_raw = None
            pvkd_vox = None
            if logits_vox_t is not None and not self.dense_train:
                s_field = logits_vox_s_dense if logits_vox_s_dense is not None else logits_vox
                if self.distill_mode == 'cwd':
                    # CWD acts only on the TPV feature planes (by design);
                    # the voxel side deliberately produces no loss term.
                    pass
                elif self.distill_mode == 'pvkd':
                    # Same support set and same field as vanilla_kd, only the objective changes. Static λ;
                    # the keys are not in _distill_key_map → bypasses the rho controller.
                    _raw_lg = self.pvkd.loss_logit_vox(s_field, logits_vox_t)
                    _raw_af, _ = self.pvkd.loss_aff_vox(s_field, logits_vox_t)
                    pvkd_vox = {
                        'loss_pvkd_logit_vox': self.pvkd.alpha * _raw_lg,
                        'loss_pvkd_aff_vox': self.pvkd.beta * self.pvkd._hard_clamp(_raw_af),
                        'pvkd_raw_logit_vox': _raw_lg.detach(),
                        'pvkd_raw_aff_vox': _raw_af.detach(),
                    }
                elif self.distill_mode == 'vanilla_kd':
                    # Baseline: same support set, same field, only the loss changes (temperature KL). Static λ weighting,
                    # not passed through the rho controller unless the config's rho contains 'kd' (the released configs drop
                    # 'kd' from rho when KDbaseline=True, so for the baseline the key is not in _distill_key_map).
                    kd_logit_raw = self.vanilla_kd.loss_logit(s_field, logits_vox_t)
                    kd_logit_loss = self.vanilla_kd.lambda_kd * kd_logit_raw
                elif self.enable_hf_distill:
                    distill_loss_vox = self.spectral_distiller(s_field, logits_vox_t)
                    if self.hybrid_dark_kd:
                        # Hybrid: in addition to the spectral term (loss_distill_vox, under rho), a dark-knowledge logit KL runs in parallel.
                        # Exactly the same as vanilla_kd -- static λ, natural magnitude (~0.8-1.4); unless the config's rho contains
                        # 'kd', the key is not in _distill_key_map, so it bypasses the rho controller and acts at full strength.
                        # In the released configs rho contains 'kd' (kd=1.30), so loss_kd_logit is governed by the gradient-ratio
                        # controller as its own group (see pointtpv_seg.py). This is the fix for the
                        # failure of the previous version: dark knowledge is no longer EMA-normalized/rescaled to the small spectral scale.
                        kd_logit_raw = self.vanilla_kd.loss_logit(s_field, logits_vox_t)
                        kd_logit_loss = self.vanilla_kd.lambda_kd * kd_logit_raw

            # Key names keep the historical naming (loss_voxel_ce is actually Lovasz; not renamed for log continuity)
            loss_dict = {
                'loss_point_lovasz': self.w_point_lovasz * point_loss,
                'loss_point_seesaw': self.w_point_seesaw * seesaw_loss_points,
                'loss_voxel_ce': self.w_voxel_lovasz * voxel_loss,
                'loss_voxel_seesaw': self.w_voxel_seesaw * seesaw_loss_voxels,
                'loss_sensor': sensor_loss_term,
                'sensor_mae_total': sensor_mae_norm.detach(),
            }

            # Add the distillation losses (if enabled). Whether a key exists depends only on whether
            # teacher logits are provided (identical on all ranks), not on loss values, which keeps the DDP
            # collectives aligned.
            if distill_loss_vox is not None:
                loss_dict['loss_distill_vox'] = distill_loss_vox
            if pvkd_vox is not None:
                # Same as above: the existence of these keys depends only on logits_vox_t and distill_mode
                # (identical on all ranks), not on the values, which keeps the DDP collectives aligned.
                loss_dict.update(pvkd_vox)
            if kd_logit_loss is not None:
                # Likewise, whether the key exists depends only on logits_vox_t (identical on all ranks, DDP aligned).
                # kd_raw_logit contains no "loss" → logging only (for magnitude comparison).
                loss_dict['loss_kd_logit'] = kd_logit_loss
                loss_dict['kd_raw_logit'] = kd_logit_raw.detach()
            # if you want per-dim logging
            if sensor_mae_per_dim is not None:
                # assume D == 6
                dim_names = ['beam', 'height', 'H_res', 'theta_low', 'theta_up']
                for i, name in enumerate(dim_names):
                    loss_dict[f'sensor_mae_{name}'] = sensor_mae_per_dim[i].detach()

            if return_logits:
                return loss_dict, logits_vox, logits_pts, pred_sensor_output
            return loss_dict

        # The teacher side goes here: with return_distill_field=True the coarse-grid distillation field is additionally returned,
        # for the student to use as logits_vox_t. Teacher and student run the same code with the same distill_points,
        # so the grids and support sets of the two fields are identical.
        if return_distill_field:
            return logits_vox, logits_pts, pred_sensor_output, logits_vox_s_dense
        return logits_vox, logits_pts, pred_sensor_output
