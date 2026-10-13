'''
DDP static-graph switch, used to make gradient checkpointing (reentrant implementation) usable under DDP.
'''
from mmcv.runner import HOOKS
from mmcv.runner.hooks.hook import Hook


@HOOKS.register_module()
class DDPStaticGraphHook(Hook):
    """Mark DDP as a static graph before training starts so that reentrant gradient checkpointing can be used.

    [Why it is needed]
    The dense teacher runs out of memory at spg=4. Per-stage measurement (single GPU, spg=1, forward hooks):

        lidar_tokenizer   +5.703 GB     lidar_backbone   +7.428 GB
        lidar_neck        +0.019 GB     tpv_aggregator   +2.803 GB
        total forward+backward peak    17.64 GB  ->  extrapolated to 75.67 GB at spg=4, OOM (capacity 79.15 GB)

    With gradient checkpointing the total peak at spg=4 is 51.42 GB, which fits. But this environment uses **torch 1.10.1**,
    whose `torch.utils.checkpoint.checkpoint` signature is (function, *args, **kwargs) and
    **has no use_reentrant** (it only exists since 1.11), so only the reentrant implementation is available; and reentrant
    checkpointing under DDP raises:

        RuntimeError: Expected to mark a variable ready only once.
        Parameter ... marked as ready twice
        (reused parameters in multiple reentrant backward passes)

    The official torch solution is `DDP._set_static_graph()`, which explicitly supports the two
    scenarios "reentrant backwards" and "activation checkpointing".

    [Why this is a hook instead of a change to mmdet_train.py]
    `mmdet_train.py` is shared production code; its DDP construction point (model_s in custom_train_detector)
    is used by all experiments, so adding a line there would affect every run. As a hook:
      - the change is additive and has no effect at all unless it is listed in custom_hooks;
      - it takes effect per config, so it can be enabled only for the spg=4 teacher (the batch size is a property of the hardware anyway);
      - if something goes wrong, only one line has to be removed from the config, without rolling back shared code.
    Also note: the commented-out `# model._set_static_graph()` line in `mmdet_train.py`
    is in **custom_train_multidb_detector**, whereas teacher training goes through
    custom_train_detector, so **uncommenting it would have no effect** -- do not be misled by it.

    [Preconditions -- must be confirmed by measurement, not assumed]
    A static graph requires that the "set of used/unused parameters" and the "computation graph structure" stay the same in every iteration.
      - LiDARCurriculumHook changes augmentation parameters (data), not the graph structure -> safe in theory;
      - but if the d2skd distillation branch switches loss terms on/off per epoch, the graph changes -> **unsafe**.

    WARNING, the most dangerous point: with static_graph=True, if the graph structure really changes between iterations, PyTorch
    **does not raise an error but silently computes wrong gradients** -- much harder to notice than a crash. Therefore this hook
    is only for the dense teacher whose graph is fixed (d2skd=False), **never for distillation students**.

    Hence this hook is not listed in custom_hooks by default (the constructor argument enable defaults to True, but as long as
    the hook is not registered in the config it has no effect at all), and it is used for real runs only after it has run successfully for several epochs.
    """

    def __init__(self, enable=True, check_invariant=True, on_violation='raise',
                 max_names=20):
        # enable          : whether to call DDP._set_static_graph()
        # check_invariant : whether to check the static-graph precondition every iter (see below)
        # on_violation    : 'raise' aborts training immediately; 'warn' only logs
        #
        # [Invariant check]
        # The static-graph precondition is a single statement: **the set of parameters taking part in backward is constant across all iterations**.
        # A violation only shows up once the graph "changes", and the first few iters are exactly when the graph does not change,
        # so "run one step with the static graph on and off and compare gradients" is an invalid check -- it only covers the interval guaranteed to agree.
        # Instead the invariant is checked directly: a gradient hook is attached to every trainable parameter to record the set of parameter names
        # that actually receive a gradient in this iter, which is compared with that of the first iter. As soon as it changes, report
        # rank / epoch / iter / the extra and missing parameter names.
        # Parameter hooks are used instead of checking p.grad is not None: mmcv's zero_grad under torch 1.10 by default
        # sets gradients to zero rather than None, so p.grad is never None from iteration 2 on and unused parameters would be judged as used.
        # The hooks reflect the real autograd and are unaffected by whether the static graph is on or off, so one can:
        #   1) first collect the real participating set with enable=False, check_invariant=True on a run without the static graph;
        #   2) after confirming that it stays constant throughout (crossing at least one epoch boundary that includes the val switch and a curriculum stage change),
        #      use enable=True for real runs, keeping the check enabled.
        # Every rank checks on its own (different ranks see different data and may take different branches; checking only rank 0 would miss them).
        assert on_violation in ('raise', 'warn'), on_violation
        self.enable = enable
        self.check_invariant = check_invariant
        self.on_violation = on_violation
        self.max_names = max_names
        self._handles = []
        self._seen = set()
        self._ref = None
        self._ref_iter = None

    def before_run(self, runner):
        model = runner.model
        if self.check_invariant:
            self._install_param_hooks(runner)
        if not self.enable:
            runner.logger.info('[DDPStaticGraphHook] enable=False, static graph not enabled'
                               ' (invariant check=%s)' % self.check_invariant)
            return
        # [Measured: not usable in this environment, do not enable] mmcv 1.4.0's
        # MMDistributedDataParallel.train_step re-implements forward on its own and misses two steps of torch DDP's
        # forward(): reducer.prepare_for_forward() (torch/nn/parallel/
        # distributed.py:855-857; without it num_iterations_ stays at 0 forever) and _DDPSink
        # (:905-915, the _delay_all_reduce of the first static-graph iteration). Calling only _set_static_graph()
        # was measured to still raise "Expected to mark a variable ready only once", and a SIGSEGV was also observed;
        # adding only prepare_for_forward without _DDPSink is even worse: the gradients of the first iter are not all-reduced
        # (measured on two ranks: 12.559979 vs 18.193453), after which DDP only synchronizes gradients but not parameters,
        # so the weights of the four GPUs diverge permanently without any error. Hence, until both steps are added and verified, enable=True raises directly.
        # The invariant check (check_invariant=True, enable=False) can still be used on its own.
        raise NotImplementedError(
            '[DDPStaticGraphHook] enable=True is not usable with mmcv 1.4.0 MMDistributedDataParallel: '
            'train_step lacks reducer.prepare_for_forward and _DDPSink, which causes a segfault or silently '
            'skips the all-reduce of the first-iteration gradients. For checking only, use enable=False.')
        fn = getattr(model, '_set_static_graph', None)
        if fn is None:
            # Skip silently for non-DDP (single-GPU MMDataParallel): there is no DDP conflict in that case anyway
            runner.logger.info(
                '[DDPStaticGraphHook] current model has no _set_static_graph '
                f'({type(model).__name__}), non-DDP case, skipping')
            # WARNING: a successful single-GPU run does **not** mean that DDP works: the conflict between reentrant checkpointing and DDP
            #   ("Expected to mark a variable ready only once") only shows up with multiple GPUs,
            #   so gradient checkpointing must be validated with real multi-GPU runs, not just on a single GPU.
            return
        fn()
        runner.logger.info(
            '[DDPStaticGraphHook] called DDP._set_static_graph(): '
            'reentrant gradient checkpointing is usable under DDP (torch %s)' % __import__('torch').__version__)

    # ------------------------------------------------------------------
    # Invariant check
    # ------------------------------------------------------------------
    def _install_param_hooks(self, runner):
        m = runner.model
        m = getattr(m, 'module', m)
        self._all = set()
        for name, prm in m.named_parameters():
            if not prm.requires_grad:
                continue
            self._all.add(name)
            self._handles.append(prm.register_hook(self._make_hook(name)))
        runner.logger.info('[DDPStaticGraphHook] invariant check installed: %d trainable parameters'
                           % len(self._all))

    def _make_hook(self, name):
        def _hook(grad):
            self._seen.add(name)
            return grad
        return _hook

    def before_train_iter(self, runner):
        if self.check_invariant:
            self._seen = set()

    def after_train_iter(self, runner):
        if not self.check_invariant:
            return
        cur = frozenset(self._seen)
        rank = getattr(runner, 'rank', 0)
        if self._ref is None:
            self._ref, self._ref_iter = cur, runner.iter
            never = sorted(self._all - cur)
            msg = ('[DDPStaticGraphHook] rank%d reference participating set recorded (iter %d): %d/%d parameters take part in backward'
                   % (rank, runner.iter, len(cur), len(self._all)))
            if never:
                # The most dangerous case: parameters that get no gradient in the first iteration are treated by the static graph as "never used";
                # once one of them suddenly takes part in a later iteration, the gradients are simply wrong.
                msg += '; ⚠ %d not participating in the first iteration: %s' % (len(never), never[:self.max_names])
                runner.logger.warning(msg)
            else:
                runner.logger.info(msg)
            return
        if cur == self._ref:
            return
        extra = sorted(cur - self._ref)
        missing = sorted(self._ref - cur)
        msg = ('[DDPStaticGraphHook] ⚠ static-graph precondition violated: the participating set of rank%d epoch %d iter %d'
               ' differs from the reference (iter %d). %d extra: %s; %d missing: %s. '
               'With static_graph enabled this would **silently compute wrong gradients**.'
               % (rank, runner.epoch, runner.iter, self._ref_iter,
                  len(extra), extra[:self.max_names], len(missing), missing[:self.max_names]))
        if self.on_violation == 'raise':
            raise RuntimeError(msg)
        runner.logger.warning(msg)
        self._ref, self._ref_iter = cur, runner.iter   # in warn mode, update the reference to avoid flooding the log

    def after_run(self, runner):
        for h in self._handles:
            h.remove()
        self._handles = []
