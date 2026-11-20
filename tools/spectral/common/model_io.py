"""Repository root, plug-in import, checkpoint loading and seeding for the spectral-measurement suite.

The suite runs from a checkout of the code release. All model code (the spectral distillers in particular) is
imported from that checkout, so the operators used for the measurements are exactly the training operators.
"""
import importlib
import importlib.util
import os
import random
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = None
_SPECTRAL_MODULE = None
SPECTRAL_DISTILLER_REL = 'projects/unilidar_plugin/pointocc_model/Spectraldistiller.py'


def set_repo_root(path):
    """Set the repository root (the directory that contains projects/). Call before importing graph_spectral."""
    global _REPO_ROOT
    _REPO_ROOT = Path(path).resolve()


def repo_root():
    """Repository root: the value given to set_repo_root, else three levels above tools/spectral/common."""
    if _REPO_ROOT is not None:
        return _REPO_ROOT
    return Path(__file__).resolve().parents[3]


def load_spectral_module():
    """Spectraldistiller.py of the checkout, imported by file path (it depends on torch only)."""
    global _SPECTRAL_MODULE
    if _SPECTRAL_MODULE is None:
        p = repo_root() / SPECTRAL_DISTILLER_REL
        if not p.is_file():
            raise FileNotFoundError(f'{p} not found: pass the repository root (the directory that contains projects/)')
        spec = importlib.util.spec_from_file_location('spectral_suite_distiller', str(p))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _SPECTRAL_MODULE = m
    return _SPECTRAL_MODULE


def import_plugin(cfg, config_path):
    """Import the mmdet3d plug-in named by the config (same rule as tools/train.py)."""
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    if getattr(cfg, 'plugin', False):
        pdir = cfg.plugin_dir if hasattr(cfg, 'plugin_dir') else os.path.dirname(config_path)
        mod = '.'.join(p for p in os.path.dirname(pdir).split('/') if p)
        importlib.import_module(mod)


def load_weights_report(model, path, allow_missing, role='student'):
    """Non-strict state-dict load with a report; refuses checkpoints that miss non-distillation parameters."""
    ck = torch.load(path, map_location='cpu')
    sd = ck.get('state_dict', ck)
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    try:
        res = model.load_state_dict(sd, strict=False)
    except RuntimeError as e:
        raise RuntimeError(f'{path} does not match the structure built from its config '
                           f'(use the config the checkpoint was trained with): {str(e)[:300]}')
    benign = ('distill', 'ema_', '_balance', 'vanilla_kd', 'cwd', 'pvkd', 'num_batches_tracked')
    if role == 'teacher':
        # The serving teacher runs with dense_train=True: the sensor head and FiLM are not used (the distillation
        # fast path has sensor_emb=None); training also loads the teacher non-strictly, so these keys may be missing.
        benign = benign + ('sensor_encoder', 'sensor_head', 'film_layer', '_film_alpha', 'sensor_mean', 'sensor_std')
    crit = [k for k in res.missing_keys if not any(b in k for b in benign)]
    rep = dict(path=str(path), role=role, meta_epoch=(ck.get('meta') or {}).get('epoch'),
               meta_time=(ck.get('meta') or {}).get('time'),
               n_missing=len(res.missing_keys), n_unexpected=len(res.unexpected_keys),
               critical_missing=crit[:50], unexpected_sample=res.unexpected_keys[:20])
    if crit and not allow_missing:
        raise RuntimeError(f'{path} misses {len(crit)} non-distillation parameters (e.g. {crit[:5]}); '
                           'wrong checkpoint for this config?')
    return rep


def _first(x):
    if isinstance(x, (list, tuple)):
        return x[0]
    if torch.is_tensor(x) and x.dim() >= 3:
        return x[0]
    return x


def seed_all(s):
    random.seed(s)
    np.random.seed(s % (2 ** 32))
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
