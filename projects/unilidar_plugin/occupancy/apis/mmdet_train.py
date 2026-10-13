# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Xiaofeng Wang
# ---------------------------------------------
import inspect
import random
import warnings
import sys
import copy
import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner, get_dist_info, load_checkpoint)
from mmcv.utils import build_from_cfg

from mmdet.core import EvalHook

from mmdet.datasets import (build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_root_logger
import time
import os.path as osp
from projects.unilidar_plugin.datasets.builder import build_dataloader
from projects.unilidar_plugin.core.evaluation.eval_hooks import OccDistEvalHook, OccEvalHook
from projects.unilidar_plugin.datasets import custom_build_dataset, ConcatenatedDataset
from mmdet3d.datasets import build_dataset
sys.setrecursionlimit(1000000)

import os

os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp' 

# import os, pandarallel

#clean up the /tmp folder	
# if os.path.isdir("/tmp") : 
#     os.system('rm -R /tmp/*')

# os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'	
# pandarallel.initialize(nb_workers = int(os.cpu_count())-1, use_memory_fs = False , progress_bar=True,verbose=2 ) 

def custom_train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):
    
    logger = get_root_logger(cfg.log_level)
    
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    
    
    
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            shuffler_sampler=cfg.data.shuffler_sampler,  # dict(type='DistributedGroupSampler'),
            nonshuffler_sampler=cfg.data.nonshuffler_sampler,  # dict(type='DistributedSampler'),
        ) for ds in dataset
    ]

    # Build the student / teacher models (for d2skd distillation)
    use_d2skd = getattr(cfg, 'd2skd', False)
    model_s = model  # student
    model_t = None   # teacher

    if use_d2skd:
        # Make a copy of the teacher model with exactly the same architecture
        model_t = copy.deepcopy(model_s)

        # Teacher weights: prefer cfg.d2skd_load_from, otherwise fall back to cfg.load_from
        teacher_ckpt = getattr(cfg, 'd2skd_load_from', None)
        if teacher_ckpt is None:
            teacher_ckpt = getattr(cfg, 'load_from', None)

        if teacher_ckpt is not None:
            # The teacher's fine->coarse class mapping must match the current GT protocol.
            # Otherwise the temperature KL would transfer the teacher's class prior under the old protocol onto the student
            # logits, directly fighting the segmentation supervision, and rho/temperature only rescale without changing the
            # direction, so they cannot fix it. No loss definition is changed; this only decides whether training may start.
            # d2skd_skip_protocol_check=True bypasses the check explicitly
            # (the rationale and the mapping difference at that time must then be documented in the experiment appendix).
            if not getattr(cfg, 'd2skd_skip_protocol_check', False):
                from .teacher_protocol import assert_teacher_label_protocol
                assert_teacher_label_protocol(teacher_ckpt, cfg, logger=logger)
            load_checkpoint(model_t, teacher_ckpt, map_location='cpu')
        else:
            raise ValueError('d2skd=True requires a teacher checkpoint '
                             '(d2skd_load_from or explicit load_from); refusing a random teacher')

        # Freeze the teacher parameters; inference only
        for p in model_t.parameters():
            p.requires_grad = False
        model_t.eval()
        
        # Set dense_train: False for the student, True for the teacher
        # Set before model wrapping to make sure the setting takes effect
        if hasattr(model_s, 'tpv_aggregator'):
            model_s.tpv_aggregator.dense_train = False
            model_s.tpv_aggregator.d2skd = False
        if hasattr(model_t, 'tpv_aggregator'):
            model_t.tpv_aggregator.dense_train = True
            model_t.tpv_aggregator.d2skd = True
            # Note: dense_train is only read in forward_train/_sampling; distillation serving goes through
            # model_t.extract_feat, where this switch has no effect. Whether the teacher's serving input is truncated
            # is controlled separately by the detector's teacher_serve_max_points (default None = no truncation).
            model_t.dense_train = True
    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        model_s = MMDistributedDataParallel(
            model_s.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        if use_d2skd and model_t is not None:
            # All teacher parameters are frozen, so no DDP is needed; just put it on the GPU
            # Note: the teacher is used for inference only and needs no gradient synchronization
            model_t = model_t.cuda()
    else:
        model_s = MMDataParallel(
            model_s.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
        if use_d2skd and model_t is not None:
            # All teacher parameters are frozen, so no DataParallel is needed; just put it on the GPU
            model_t = model_t.cuda(cfg.gpu_ids[0])

    # From here on, model still points to the student, for compatibility
    model = model_s
    
    # In distributed mode, synchronize model parameters across processes (including the teacher)
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:
            # Rank 0 initializes the parameters; the other processes sync from it
            pass
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
        if use_d2skd and model_t is not None:
            for param in model_t.parameters():
                dist.broadcast(param.data, src=0)
    
    # build runner
    # Build the optimizer for the student only; the teacher stays frozen
    optimizer = build_optimizer(model, cfg.optimizer)

    assert 'runner' in cfg
    if use_d2skd:
        runner = build_runner(
            cfg.runner,
            default_args=dict(
                model=model,
                model_t=model_t,
                optimizer=optimizer,
                work_dir=cfg.work_dir,
                logger=logger,
                meta=meta))
    else:
        runner = build_runner(
            cfg.runner,
            default_args=dict(
                model=model,
                optimizer=optimizer,
                work_dir=cfg.work_dir,
                logger=logger,
                meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting TODO
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    
    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        if val_samples_per_gpu > 1:
            assert NotImplementedError()
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = custom_build_dataset(cfg.data.val, dict(test_mode=True))

        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
            shuffler_sampler=cfg.data.shuffler_sampler,  # dict(type='DistributedGroupSampler'),
            nonshuffler_sampler=cfg.data.nonshuffler_sampler,  # dict(type='DistributedSampler'),
        )
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_cfg['jsonfile_prefix'] = osp.join('val', cfg.work_dir, time.ctime().replace(' ','_').replace(':','_'))
        eval_hook = OccDistEvalHook if distributed else OccEvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            # hook = build_from_cfg(hook_cfg, HOOKS) 
            # FIXME hardcode specifying dataloader as parameter 
            # val_dataloader is only defined when validate=True
            hook_kwargs = {}
            if validate and 'val_dataloader' in locals():
                # Inject only into hooks whose signature actually accepts dataloader (targeted fix of the old FIXME);
                # built-in mmcv hooks (e.g. EMAHook) do not accept this argument, and blind injection raises TypeError.
                hook_cls = HOOKS.get(hook_cfg['type']) if isinstance(hook_cfg.get('type'), str) else hook_cfg.get('type')
                if hook_cls is not None:
                    sig_params = inspect.signature(hook_cls.__init__).parameters
                    accepts_dataloader = ('dataloader' in sig_params) or any(
                        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params.values())
                    if accepts_dataloader:
                        hook_kwargs['dataloader'] = val_dataloader
            hook = build_from_cfg(hook_cfg, HOOKS, hook_kwargs)
            runner.register_hook(hook, priority=priority)

    # Attach the teacher model to the runner for later use
    if use_d2skd and model_t is not None:
        # runner.model is still the student; model_t is attached additionally
        runner.model_t = model_t

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)


def custom_train_multidb_detector(model,
                   dataset_1,
                   dataset_2,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):
    
    logger = get_root_logger(cfg.log_level)
    
    dataset_1 = dataset_1 if isinstance(dataset_1, (list, tuple)) else [dataset_1]
    dataset_2 = dataset_2 if isinstance(dataset_2, (list, tuple)) else [dataset_2]
    
    merged_dataset = ConcatenatedDataset(dataset_1, dataset_2)
    # merged_dataset = [dataset_1[0], dataset_2[0]]
    
    data_loaders = [
        build_dataloader(
            merged_dataset,
            cfg.data_merge.samples_per_gpu,
            cfg.data_merge.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            shuffler_sampler=cfg.data_merge.shuffler_sampler,  # dict(type='DistributedGroupSampler'),
            nonshuffler_sampler=cfg.data_merge.nonshuffler_sampler,  # dict(type='DistributedSampler'),
            drop_last=True,
        )
    ]
    

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        # model._set_static_graph()
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)


    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    assert 'runner' in cfg
    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting TODO
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    
    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data_nu.val.pop('samples_per_gpu', 1)
        
        # if val_samples_per_gpu > 1:
        #     assert NotImplementedError()
        #     # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        #     cfg.data_nu.val.pipeline = replace_ImageToTensor(
        #         cfg.data_nu.val.pipeline)
        #     cfg.data_sk.val.pipeline = replace_ImageToTensor(
        #         cfg.data_sk.val.pipeline)
        # val_dataset_1 = [build_dataset(cfg.data_nu.val, dict(test_mode=True))]
        # val_dataset_2 = [build_dataset(cfg.data_sk.val, dict(test_mode=True))]
        # val_dataset_1 = val_dataset_1 if isinstance(val_dataset_1, (list, tuple)) else [val_dataset_1]
        # val_dataset_2 = val_dataset_2 if isinstance(val_dataset_2, (list, tuple)) else [val_dataset_2]

        # val_dataset_merge = ConcatenatedDataset(val_dataset_1, val_dataset_2)

        # val_dataloader = build_dataloader(
        #     val_dataset,
        #     samples_per_gpu=val_samples_per_gpu,
        #     workers_per_gpu=cfg.data_merge.workers_per_gpu,
        #     dist=distributed,
        #     shuffle=False,
        #     shuffler_sampler=cfg.data_merge.shuffler_sampler,  # dict(type='DistributedGroupSampler'),
        #     nonshuffler_sampler=cfg.data_merge.nonshuffler_sampler,  # dict(type='DistributedSampler'),
        #     drop_last=True,
        # )

        
        # eval_cfg_nu = cfg.get('evaluation_nu', {})
        # eval_cfg_nu['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        # eval_cfg_nu['jsonfile_prefix'] = osp.join('val', cfg.work_dir, time.ctime().replace(' ','_').replace(':','_'))
        # eval_cfg_sk = cfg.get('evaluation_sk', {})
        # eval_cfg_sk['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        # eval_cfg_sk['jsonfile_prefix'] = osp.join('val', cfg.work_dir, time.ctime().replace(' ','_').replace(':','_'))
        # eval_hook = OccDistEvalHook if distributed else OccEvalHook
        # runner.register_hook(eval_hook(val_dataloader, **eval_cfg_nu))
        # runner.register_hook(eval_hook(val_dataloader, **eval_cfg_sk))
        if val_samples_per_gpu > 1:
            assert NotImplementedError()
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = custom_build_dataset(cfg.data.val, dict(test_mode=True))

        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data_merge.workers_per_gpu,
            dist=distributed,
            shuffle=False,
            shuffler_sampler=cfg.data_merge.shuffler_sampler,  # dict(type='DistributedGroupSampler'),
            nonshuffler_sampler=cfg.data_merge.nonshuffler_sampler,  # dict(type='DistributedSampler'),
            drop_last=True,
        )
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_cfg['jsonfile_prefix'] = osp.join('val', cfg.work_dir, time.ctime().replace(' ','_').replace(':','_'))
        eval_hook = OccDistEvalHook if distributed else OccEvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            # hook = build_from_cfg(hook_cfg, HOOKS) 
            # FIXME hardcode specifying dataloader as parameter 
            # val_dataloader is only defined when validate=True
            hook_kwargs = {}
            if validate and 'val_dataloader' in locals():
                # Inject only into hooks whose signature actually accepts dataloader (targeted fix of the old FIXME);
                # built-in mmcv hooks (e.g. EMAHook) do not accept this argument, and blind injection raises TypeError.
                hook_cls = HOOKS.get(hook_cfg['type']) if isinstance(hook_cfg.get('type'), str) else hook_cfg.get('type')
                if hook_cls is not None:
                    sig_params = inspect.signature(hook_cls.__init__).parameters
                    accepts_dataloader = ('dataloader' in sig_params) or any(
                        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params.values())
                    if accepts_dataloader:
                        hook_kwargs['dataloader'] = val_dataloader
            hook = build_from_cfg(hook_cfg, HOOKS, hook_kwargs)
            runner.register_hook(hook, priority=priority)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)
