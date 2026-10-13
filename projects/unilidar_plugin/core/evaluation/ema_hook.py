'''
WarmupEMAHook -- fixes the "cold-start contamination" defect of mmcv's EMAHook.

Two problems of mmcv.runner.EMAHook (fatal at the training scale of this project):
  1. The EMA buffer is initialized from the [initial weights] in before_run and then approaches the
     current weights exponentially with a fixed momentum m.
     Residual fraction of the initial weights = (1-m)^N. With m=2e-4 (time constant 5000 iters):
       nuScenes, ~3500 iters per epoch -> at the end of epoch 1 the EMA still holds 0.9998^3500 ≈ 50% initial weights
       SemanticKITTI, ~2400 iters per epoch -> still ≈ 62% initial weights
     Yet EvalHook/CheckpointHook use exactly these swapped-in EMA weights => the val metrics of the first
     few epochs amount to evaluating a "half-randomly-initialized" model, and rare classes (e.g. Living Being) drop straight to 0.
  2. Its warm_up argument is `momentum = min(self.momentum, (1+t)/(warm_up+t))` -- it uses
     min(), which only makes the early EMA slower and cannot mitigate the cold start at all (completely ineffective when m is small).

Fix in this hook:
  * iter < start_iters (default: 1 epoch): the EMA buffer strictly copies the current weights.
    In this phase EMA weights == current weights, so eval/checkpoint results are bit-identical to "no EMA" => zero contamination.
  * iter >= start_iters: accumulate with a bias-corrected momentum
        m_t = max(target_momentum, 1 / ema_steps)
    It starts out equivalent to a "running average" of the weights (step 1 has m=1, i.e. takes the current weights directly) and
    smoothly transitions to the target exponential momentum as the step count grows. The EMA never contains the initial weights.

This keeps the smoothing benefit of EMA on val oscillation without sacrificing the readability of early-epoch metrics.
'''
from mmcv.runner import HOOKS
from mmcv.runner.hooks.ema import EMAHook


@HOOKS.register_module()
class WarmupEMAHook(EMAHook):
    """EMA with warmup: no accumulation during the first start_epochs epochs, bias-corrected accumulation afterwards.

    Args:
        momentum (float): target (steady-state) momentum. The EMA time constant is about 1/momentum iters.
        start_epochs (float): number of epochs after which EMA accumulation actually starts. 0 means start immediately
            (still bias-corrected, so no initial weights remain either). Default 1.
        interval (int): update the EMA every this many iters.
    """

    def __init__(self, momentum=0.0002, start_epochs=1, interval=1,
                 dataloader=None, **kwargs):
        # dataloader: the custom-hook registrar in this project's mmdet_train.py blindly injects
        # dataloader=val_dataloader into any hook whose signature contains **kwargs; accept it explicitly and drop it,
        # otherwise it would be passed via **kwargs into mmcv EMAHook.__init__ and raise TypeError.
        del dataloader
        super().__init__(momentum=momentum, interval=interval, **kwargs)
        # The parent class turns momentum into momentum**interval; take the transformed steady-state momentum here
        self._target_momentum = self.momentum
        self.start_epochs = float(start_epochs)
        self._start_iters = None
        self._ema_steps = 0

    def _resolve_start_iters(self, runner):
        if self._start_iters is not None:
            return
        data_loader = getattr(runner, 'data_loader', None)
        iters_per_epoch = len(data_loader) if data_loader is not None else 0
        self._start_iters = int(self.start_epochs * iters_per_epoch)
        if hasattr(runner, 'logger'):
            runner.logger.info(
                f'[WarmupEMAHook] iters/epoch={iters_per_epoch}, '
                f'EMA will start accumulating after iter {self._start_iters} (= {self.start_epochs} epoch); '
                f'before that, the EMA weights strictly follow the current weights (eval/checkpoint equivalent to no EMA). '
                f'steady-state momentum={self._target_momentum} (time constant about '
                f'{int(1.0 / max(self._target_momentum, 1e-12))} iter).')

    def after_train_iter(self, runner):
        self._resolve_start_iters(runner)
        curr_step = runner.iter

        if curr_step < self._start_iters:
            # Not accumulating yet: the EMA buffer tracks the current weights, guaranteeing zero contamination in this period
            for name, parameter in self.model_parameters.items():
                buffer_parameter = self.model_buffers[self.param_ema_buffer[name]]
                buffer_parameter.data.copy_(parameter.data)
            return

        if curr_step % self.interval != 0:
            return

        self._ema_steps += 1
        # Bias correction: step 1 has m=1 (takes the current weights directly), then transitions to the target momentum
        momentum = max(self._target_momentum, 1.0 / self._ema_steps)
        for name, parameter in self.model_parameters.items():
            buffer_parameter = self.model_buffers[self.param_ema_buffer[name]]
            # Use the new add_ signature to avoid mmcv's deprecation warning for add_(Number, Tensor)
            buffer_parameter.mul_(1 - momentum).add_(parameter.data, alpha=momentum)
