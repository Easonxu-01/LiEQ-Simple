# ---------------------------------------------
# Teacher label-protocol guard for d2skd.
#
# The dark-knowledge path of distillation assumes that the teacher is a "strictly better invariance target" (see tpv_aggregator.py).
# If the teacher was trained under a different fine->coarse mapping, it puts high-confidence probability mass on
# classes that the current GT no longer uses; the gradient of the temperature KL is (T/N)(p_s^T - p_t^T), which transfers this wrong
# class prior voxel by voxel onto the student logits. rho / temperature / EMA are all positive scalars and cannot change the direction.
# Therefore this module turns "teacher and student share the same label protocol" from an assumption in comments into a hard constraint at start-up.
# This module does not change any loss definition or optimization objective; it only decides whether a training run may start.
# Measured example: a SemanticKITTI teacher checkpoint embedded the mapping 15:3/72:4/52:2/99:6 while the
# current config uses 15:6/72:7/52:0/99:0; the mismatch fell on 11.8% of the real points; the nuScenes/Waymo teachers were affected likewise.
# ---------------------------------------------
import re

import torch


def _dg_maps_from_cfg(obj):
    """Recursively find all actually deployed 8-class (DG) seg_label_mapping dicts in the parsed cfg."""
    found = []

    def walk(o):
        if isinstance(o, dict):
            m = o.get('seg_label_mapping')
            if isinstance(m, dict) and m and max(m.values()) <= 7:
                found.append({int(k): int(v) for k, v in m.items()})
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)

    walk(obj)
    return found


def _dg_map_from_ckpt_text(cfg_text):
    """Find the same DG mapping in the config dump text embedded in the checkpoint."""
    for m in re.finditer(r"seg_label_mapping=dict\(\{(.*?)\}\)", cfg_text, re.S):
        d = {int(k): int(v) for k, v in re.findall(r'(\d+)\s*:\s*(\d+)', m.group(1))}
        if d and max(d.values()) <= 7:
            return d
    return None


def assert_teacher_label_protocol(teacher_ckpt, cfg, logger=None):
    """Refuse distillation when the teacher's training label protocol differs from the current config.

    Returns the teacher's DG mapping; if it cannot be determined, only warns and returns None (does not block, but leaves a trace).
    Also refuses directly if the config contains several mutually inconsistent DG mappings (train/val using different conventions).
    """
    try:
        meta = torch.load(teacher_ckpt, map_location='cpu').get('meta', {})
    except Exception as e:                                    # noqa: BLE001
        raise ValueError(f'cannot read teacher checkpoint {teacher_ckpt}: {e}')

    t_map = _dg_map_from_ckpt_text(meta.get('config', '') or '')
    c_maps = _dg_maps_from_cfg(cfg._cfg_dict if hasattr(cfg, '_cfg_dict') else cfg)
    c_map = c_maps[0] if c_maps else None
    if c_maps and any(m != c_map for m in c_maps[1:]):
        raise ValueError(
            'config carries more than one distinct 8-class seg_label_mapping; '
            'refusing to pick one silently — unify train/val/test mappings first')

    if t_map is None or c_map is None:
        msg = ('cannot verify d2skd teacher label protocol for '
               f'{teacher_ckpt} (teacher_map='
               f'{"missing" if t_map is None else "ok"}, config_map='
               f'{"missing" if c_map is None else "ok"}); distillation targets '
               'are UNVERIFIED — record this in the experiment appendix')
        if logger is not None:
            logger.warning(msg)
        else:
            print('WARNING:', msg)
        return None

    diff = {k: (t_map.get(k), c_map.get(k))
            for k in set(t_map) | set(c_map) if t_map.get(k) != c_map.get(k)}
    if diff:
        lines = [f'    fine {k}: teacher->{a}  config->{b}'
                 for k, (a, b) in sorted(diff.items())]
        raise ValueError(
            'd2skd teacher label protocol mismatch — refusing to distill from a '
            'teacher trained on a different class map.\n'
            f'  teacher checkpoint: {teacher_ckpt} '
            f'(epoch {meta.get("epoch")}, {meta.get("time")})\n'
            f'  {len(diff)} differing fine label(s):\n' + '\n'.join(lines) +
            '\n  Retrain the teacher under the current map, or point '
            'd2skd_load_from at a teacher that matches it. Set '
            'd2skd_skip_protocol_check=True to override deliberately '
            '(and record the reason in the experiment appendix).')

    if logger is not None:
        logger.info(f'd2skd teacher label protocol verified: {teacher_ckpt}')
    return t_map
