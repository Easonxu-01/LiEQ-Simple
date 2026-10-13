from mmcv import runner
from mmcv.runner import HOOKS
from mmcv.runner.hooks.hook import Hook
import math
import torch


@HOOKS.register_module()
class LiDARCurriculumHook(Hook):

    def __init__(
        self, 
        schedule,
        alpha_key='sensor_mae_total',   # name of the sensor MAE log entry from the head forward
        alpha_step=0.04,                 # step size of each adjustment
        alpha_min=0.0,
        alpha_max=1.0,
        # [Do not tighten further] 0.5/0.6 are validated values: they make film_alpha rise to 1.0 around epoch 2,
        # so that the FiLM condition during training (predicted sensor parameters) matches inference -- this is the prerequisite
        # for cls5 reaching 13.2%. Tightening to 0.25/0.35 was tried: after convergence the sensor MAE settles at ~0.35 and never
        # reaches 0.25, so film_alpha stays 0 throughout -> training uses only GT conditions while inference uses only predicted
        # conditions; the train/test paths mismatch and the all-class metrics drop significantly.
        alpha_target_low=0.5,           # MAE below this -> increase α
        alpha_target_high=0.6,          # MAE above this -> decrease α
        alpha_momentum=0.9,              # EMA momentum
        # Progress floor: α ≥ alpha_max × ramp(training progress), which guarantees that α equals alpha_max at the end
        # of training, independently of the MAE gate. Both parameters are [fractions of the total training progress] (0~1):
        #   before floor_start the floor is 0; from floor_start to floor_end it rises linearly to alpha_max; it stays there afterwards.
        # Why it is needed (lesson measured on waymo): waymo's sensor MAE has a floor of ~0.5σ (the height/theta_low
        # dimensions are not identifiable), so the bang-bang gate either oscillates throughout (thresholds 0.5/0.6, MAE stuck inside the band) or
        # crosses the band too early during the ep2 descent and locks to 1 one-way (thresholds 0.6/0.7) -> both settings fail. The floor does not
        # depend on the MAE trajectory: it kicks in late, ramps up smoothly and is guaranteed to reach 1 at the end. Combined with setting
        # alpha_target_low below the MAE floor (e.g. 0.45), the MAE path stays silent and α is driven entirely by the floor.
        # None (default) = disabled, bit-identical to the previous behaviour -- nusc/SK are unaffected.
        alpha_floor_start=None,
        alpha_floor_end=None,
        # --- NEW: consistency schedule (relative to the total training length, adapts to batch/GPU count) ---
        # frac is relative to runner.max_iters; the defaults reproduce the validated reference schedule (absolute 1000/5000 iters
        # at total batch 20, max_iters≈22800). When the hardware/batch changes, the relative position stays the same,
        # which avoids consistency kicking in relatively earlier (iters/epoch doubles with a small batch) and over-suppressing rare classes.
        consistency_start_frac=0.044,
        consistency_warmup_frac=0.219,
        # Absolute iter override (default None = converted from frac); pass explicitly only when fixed absolute steps are needed.
        consistency_start_iter=None,
        consistency_warmup_iters=None,
        **kwargs
    ):
        """
        schedule: [(start_epoch, end_epoch, cfg_dict), ...]
        cfg_dict format:
        {
            'num_of_beams': (low, high),
            'horizontal_angular_resolution': (low, high),
            'lower_vertical_field_of_view_bound': (low, high),
            'upper_vertical_field_of_view_bound': (low, high),
            'LiDAR_height': (low, high)
        }
        """
        self.schedule = schedule

        # --- dynamic α settings ---
        self.alpha_key = 'sensor_mae_total'
        self.alpha_step = alpha_step
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha_target_low = alpha_target_low
        self.alpha_target_high = alpha_target_high
        self.alpha_momentum = alpha_momentum
        self.alpha_floor_start = alpha_floor_start
        self.alpha_floor_end = alpha_floor_end

        self.ema_mae = None  # used for the EMA computation
        
        # consistency schedule (relative positions, converted to absolute iters at runtime via runner.max_iters)
        self.consistency_start_frac = consistency_start_frac
        self.consistency_warmup_frac = consistency_warmup_frac
        self.consistency_start_iter = consistency_start_iter      # absolute override, None = use frac
        self.consistency_warmup_iters = consistency_warmup_iters

    # ----------------------
    # Utility: compute mean/std from the ranges
    # ----------------------
    def compute_mean_std(self, cfg_range):
        """Compute mean/std for the LiDAR parameter ranges of one stage.

        Note: num_of_beams is not sampled uniformly but from a Gaussian mixture over several canonical beam counts
        (see sample_beams), so its mean/std are computed from the exact moments of that distribution; the other
        dimensions are estimated with a uniform distribution U(a,b). This keeps the normalization consistent with the real sampling distribution.
        """
        # Deferred import to avoid a circular import with the model module
        from projects.unilidar_plugin.pointocc_model.pointtpv_seg import beam_distribution_mean_std

        params = [
            'num_of_beams',
            'LiDAR_height',
            'horizontal_angular_resolution',
            'lower_vertical_field_of_view_bound',
            'upper_vertical_field_of_view_bound',
        ]

        mean_vals = []
        std_vals = []

        for p in params:
            a, b = cfg_range[p]
            if p == 'num_of_beams':
                mean, std = beam_distribution_mean_std(a, b)
                std = max(std, 1e-6)
            else:
                mean = 0.5 * (a + b)
                std = (b - a) / (12 ** 0.5)  # standard deviation of the uniform distribution U(a,b)
            mean_vals.append(mean)
            std_vals.append(std)

        mean = torch.tensor(mean_vals, dtype=torch.float32)
        std = torch.tensor(std_vals, dtype=torch.float32)
        return mean, std

    # ----------------------
    # Every epoch: update the LiDAR sampling ranges + mean/std
    # ----------------------
    def before_train_epoch(self, runner):
        epoch = runner.epoch
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module

        for start, end, cfg in self.schedule:
            if start <= epoch < end:
                # 1. Update the sampler parameters (pre-existing)
                model.sensor_config.num_of_beams = cfg['num_of_beams']
                model.sensor_config.horizontal_angular_resolution = cfg['horizontal_angular_resolution']
                model.sensor_config.lower_vertical_field_of_view_bound = cfg['lower_vertical_field_of_view_bound']
                model.sensor_config.upper_vertical_field_of_view_bound = cfg['upper_vertical_field_of_view_bound']
                model.sensor_config.LiDAR_height = cfg['LiDAR_height']
                # 2. Update mean/std (newly added)
                mean, std = self.compute_mean_std(cfg)
                head = getattr(model, 'tpv_aggregator', None)
                if head is not None and hasattr(head, 'update_sensor_stats'):
                    head.update_sensor_stats(mean, std)
                    runner.logger.info(
                        f"[Epoch {epoch}] Updated Sensor Mean/Std:\n"
                        f"  mean={mean.tolist()}\n"
                        f"  std ={std.tolist()}"
                    )
                runner.logger.info(
                    f"[Epoch {epoch}] Updated Sensor Sampling Config:\n{model.sensor_config}"
                )
                break


    # ----------------------
    # Every iter: adjust α (dynamic FiLM mixing)
    # ----------------------
    def after_train_iter(self, runner):

        cur_hist = runner.log_buffer.val_history.get(self.alpha_key, None)
        cur_mae = float(cur_hist[-1]) if cur_hist else None
        # The real-scan main samples of reset_random have no GT sensor parameters, so their sensor_mae_total is logged as 0;
        # skip 0 / non-finite values to avoid diluting the EMA with invalid samples.
        if cur_mae is not None and (not math.isfinite(cur_mae) or cur_mae <= 0.0):
            cur_mae = None
        if cur_mae is not None:
            # update the EMA
            if self.ema_mae is None:
                self.ema_mae = cur_mae
            else:
                m = self.alpha_momentum
                self.ema_mae = m * self.ema_mae + (1 - m) * cur_mae
            # find the model head
            model = runner.model
            if hasattr(model, 'module'):
                model = model.module
            head = getattr(model, 'tpv_aggregator', None)
            if head is not None and hasattr(head, 'set_film_alpha'):
                # current α
                alpha = float(getattr(head, 'film_alpha', 0.0))
                # adjust α dynamically
                if self.ema_mae < self.alpha_target_low:
                    alpha = min(alpha + self.alpha_step, self.alpha_max)
                elif self.ema_mae > self.alpha_target_high:
                    alpha = max(alpha - self.alpha_step, self.alpha_min)
                # update α
                head.set_film_alpha(alpha)
                # write to the log
                runner.log_buffer.update(
                    {'film_alpha': alpha, 'ema_sensor_mae': self.ema_mae},
                    runner.outputs.get('num_samples', 1)
                )

        # Progress floor: independent of the MAE gate, it must run even if this iter has no valid MAE --
        # otherwise iters with no-GT samples such as reset_random would miss raising the floor. Taking the max means that if the MAE
        # path has already pushed α higher (the nusc/SK case), the floor never takes effect.
        if self.alpha_floor_start is not None:
            max_iters = getattr(runner, 'max_iters', None)
            if max_iters:
                model = runner.model
                if hasattr(model, 'module'):
                    model = model.module
                head = getattr(model, 'tpv_aggregator', None)
                if head is not None and hasattr(head, 'set_film_alpha'):
                    prog = runner.iter / float(max_iters)
                    end = self.alpha_floor_end if self.alpha_floor_end is not None else 1.0
                    if prog >= self.alpha_floor_start:
                        if prog >= end:
                            floor = self.alpha_max
                        else:
                            floor = self.alpha_max * (prog - self.alpha_floor_start) \
                                / max(1e-9, end - self.alpha_floor_start)
                        alpha_now = float(getattr(head, 'film_alpha', 0.0))
                        if floor > alpha_now:
                            head.set_film_alpha(floor)
                            runner.log_buffer.update(
                                {'film_alpha': floor},
                                runner.outputs.get('num_samples', 1)
                            )
            
         # -------------------------
        # NEW: consistency loss warmup
        # -------------------------
        cur_iter = runner.iter

        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        head = getattr(model, 'tpv_aggregator', None)
        if head is None or not hasattr(head, 'consistency_loss_weight'):
            return

        base_w = 1.0

        # relative position -> absolute iter: adapts to batch/GPU count (the absolute parameters act as overrides when not None)
        max_iters = getattr(runner, 'max_iters', None) or 1
        start_iter = (self.consistency_start_iter
                      if self.consistency_start_iter is not None
                      else self.consistency_start_frac * max_iters)
        warmup_iters = (self.consistency_warmup_iters
                        if self.consistency_warmup_iters is not None
                        else self.consistency_warmup_frac * max_iters)
        warmup_iters = max(float(warmup_iters), 1.0)

        if cur_iter < start_iter:
            cur_w = 1e-3
        elif cur_iter < start_iter + warmup_iters:
            progress = (cur_iter - start_iter) / warmup_iters
            cur_w = base_w * progress
        else:
            cur_w = base_w

        # write back to the head (direct overwrite)
        head.consistency_loss_weight = cur_w

        # logging (strongly recommended)
        runner.log_buffer.update(
            {'consistency_weight': cur_w},
            runner.outputs.get('num_samples', 1)
        )