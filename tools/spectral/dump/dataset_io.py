"""Per-dataset input settings of the dump stage: label-map name, TF32 policy, data.test overrides, label-map parsing.
No analysis definition (bands, operators, caps, seeds, statistics) lives here.
"""
import ast
import re
from pathlib import Path

DATASETS = {
    'semantickitti': dict(map_name='SEMANTICKITTI_TO_CL', tf32_default='default'),
    'nuscenes': dict(map_name='NUSCENES_TO_CL', tf32_default='default'),
    # Waymo: every GPU process runs with TF32 off (matmul and cuDNN).
    'waymo': dict(map_name='WAYMO_TO_CL', tf32_default='off'),
}


def get(ds):
    if ds not in DATASETS:
        raise SystemExit(f'--dataset must be one of {sorted(DATASETS)}: {ds}')
    return DATASETS[ds]


def normalize_data_root(ds, root):
    """data.test.data_root override. The dataset classes join paths by string replacement, so the value ends with '/'.
    Waymo: the shared loader applies the Waymo label format (int32 (instance, semantic) pairs -> semantic column) and
    the Waymo label-0 point filter only when the label path contains the lowercase substring 'waymo'; otherwise the
    labels would be read in SemanticKITTI format. The path (or a symlink to it) must therefore contain 'waymo'."""
    if root is None:
        return None
    r = str(root)
    if not r.endswith('/'):
        r += '/'
    if ds == 'waymo' and 'waymo' not in r:
        raise SystemExit(f'--data-root {r} does not contain the lowercase substring "waymo": the shared Waymo loader '
                         'would read the labels in SemanticKITTI format and skip the label-0 filter. '
                         'Use a path (or a symlink) whose name contains "waymo".')
    return r


def data_test_overrides(ds, role, data_root=None):
    """mmcv merge_from_dict overrides for cfg.data.test, and loader-step overrides (none are needed for the inputs of
    the paper: the dense-union and raw inputs are derived from the data config by run_dump.make_ds_cfg).
    role: 'data' (the inputs, built from the data config)."""
    o = {}
    if data_root is not None:
        o['data.test.data_root'] = data_root
    loader = []
    return o, loader


def apply_data_overrides(cfg, ds, role, data_root=None):
    o, loader = data_test_overrides(ds, role, data_root)
    if o:
        cfg.merge_from_dict(o)
    for typ, key, val in loader:
        n = 0
        for step in cfg.data.test.pipeline:
            if step.get('type') == typ:
                step[key] = val
                n += 1
        if n != 1:
            raise RuntimeError(f'{ds}/{role}: expected exactly one {typ} in data.test.pipeline, got {n}')
    return dict(merge=o, pipeline=[list(x) for x in loader])


def set_tf32(mode):
    """mode: 'default' (torch default) or 'off'. Must run before any CUDA work."""
    import torch
    if mode == 'off':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    elif mode != 'default':
        raise SystemExit(f'--tf32 must be default|off: {mode}')
    return dict(mode=mode, matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32))


def resolve_tf32(ds, requested):
    d = get(ds)['tf32_default']
    if requested in (None, '', 'dataset'):
        return d
    if ds == 'waymo' and requested != 'off':
        raise SystemExit('waymo requires TF32 off')
    return requested


def _strip_comment_lines(text):
    return '\n'.join(ln for ln in text.splitlines() if not ln.lstrip().startswith('#'))


def parse_map_text(text, name):
    """The 8-class label map `name` from config text: full-line comments are dropped first, then all definitions
    found by the regex must be identical."""
    t = _strip_comment_lines(text)
    ms = re.findall(name + r'\s*=\s*(?:dict\(\s*)?(\{[^}]*\})', t)
    if not ms:
        raise RuntimeError(f'cannot find {name} in text')
    maps = [{int(k): int(v) for k, v in ast.literal_eval(m).items()} for m in ms]
    if any(mp != maps[0] for mp in maps):
        raise RuntimeError(f'{name}: several inconsistent definitions in text')
    return maps[0]


def parse_map_file(path, name):
    """Label map of a config file: the regex result must equal mmcv Config.fromfile(path)[name]."""
    from mmcv import Config
    p = Path(path)
    mp = parse_map_text(p.read_text(), name)
    c = Config.fromfile(str(p))
    ref = {int(k): int(v) for k, v in dict(c[name]).items()}
    if ref != mp:
        raise RuntimeError(f'{p}: regex {name} != Config.fromfile {name}')
    return mp
