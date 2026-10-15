from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F

from nucleus.data.batching import CollatedBatch
from nucleus.models.modules import MoEConditionedForecastModule
from nucleus.noise import LogUniformNoise, SDFAugmentation

FLOW_TIME_BIN_EDGES = (0.0, 0.25, 0.5, 0.75, 1.0)
PREDICTION_TARGETS = ("velocity", "endpoint")


@dataclass
class FlowPrediction:
    flow_time: torch.Tensor
    v_pred: torch.Tensor
    v_target: torch.Tensor
    delta_pred: torch.Tensor
    delta: torch.Tensor
    moe_outputs: list


def remaining_flow_time(flow_time: torch.Tensor) -> torch.Tensor:
    return (1.0 - flow_time)[:, :, None, None, None]


def per_sample_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction - target).pow(2).flatten(start_dim=1).mean(dim=1)


class FlowMatchingModule(MoEConditionedForecastModule):
    """
    physics-conditioned residual flow matching for forecasting

    residual (top-level key in model_cfg, default true):
        true:  delta = target - input[:, -1:]    -- flow toward the change from the last context frame
        false: delta = target                    -- flow toward the clean target frame, no anchor
    t ~ uniform(0, 1) ** flow_time_power

    zt = (1-t)*noise + t*delta
        -- input frames are stationary anchors, only the target slot moves

    prediction_target (top-level key in model_cfg):
        velocity: model(zt, t) -> v,     loss = MSE(v, delta - noise)
        endpoint: model(zt, t) -> delta, loss = MSE(delta_hat, delta)
                  (the velocity loss weighted by (1-t)^2, without the noise copying)

    inference: start from zt=noise, integrate dz/dt = v(zt, t) to get delta_hat
        -- endpoint models give v = (delta_hat - zt) / (1 - t)
        -- forecast = input[:, -1:] + delta_hat if residual, else delta_hat
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prediction_target = self.model_cfg.get("prediction_target", "velocity")
        if self.prediction_target not in PREDICTION_TARGETS:
            raise ValueError(f"prediction_target must be one of {PREDICTION_TARGETS}, got {self.prediction_target}")
        # training draws t = u ** flow_time_power; powers above 1 put more samples at small t,
        # where the forecast must come from the context alone
        self.flow_time_power = float(self.model_cfg.get("flow_time_power", 1.0))
        # std of the noise source; scalar or per-field list [dfun, temp, velx, vely].
        # For residual=True, matching the empirical delta RMS per field avoids the model
        # spending capacity on noise cancellation for slow fields like dfun (RMS ~0.05).
        noise_std_cfg = self.model_cfg.get("noise_std", 1.0)
        if isinstance(noise_std_cfg, (list, tuple)):
            self.register_buffer(
                "noise_std",
                torch.tensor(noise_std_cfg, dtype=torch.float32),
            )
        else:
            self.noise_std = float(noise_std_cfg)
        # residual: flow toward (target - last context frame); otherwise flow toward the clean target
        self.residual = bool(self.model_cfg.get("residual", True))

        # per-field loss weights: explicitly set via config to upweight specific fields
        # (e.g. field_loss_weights: [4.0, 1.0, 1.0, 1.0] to upweight dfun)
        field_weights_cfg = self.model_cfg.get("field_loss_weights", None)
        if field_weights_cfg is not None:
            weights = torch.tensor(field_weights_cfg, dtype=torch.float32)
            weights = weights / weights.mean()  # keep loss on same scale as unweighted MSE
            self.register_buffer("field_loss_weights", weights)
        else:
            self.field_loss_weights = None

        # reuse NUCLEUS's own augmentation objects so both models see identically corrupted
        # context; field dropout is skipped because zeroing the anchor breaks the residual target
        wanted_types = ((LogUniformNoise,) if self.model_cfg.get("context_noise", False) else ()) \
            + ((SDFAugmentation,) if self.model_cfg.get("sdf_corruptions", False) else ())
        self.context_augmentations = [
            augmentation for augmentation in self.augmentations if wanted_types and isinstance(augmentation, wanted_types)
        ]

    def _augment_context(self, batch: CollatedBatch) -> CollatedBatch:
        # applied before _sample_flow: in residual mode the anchor is then the noisy last frame,
        # in full-field mode the target stays clean and only the conditioning is corrupted
        with torch.no_grad():
            for augmentation in self.context_augmentations:
                batch.input = augmentation(batch.input)
        return batch

    def _sample_flow(
        self, batch: CollatedBatch, flow_time_power: float = 1.0,
        clean_anchor: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (model_input, t, v_target) for residual flow matching."""
        y = batch.target  # (B, T_out, H, W, C)
        # Use the pre-augmentation anchor so corrupted context doesn't shift the residual target.
        anchor = clean_anchor if clean_anchor is not None else batch.input[:, -1:]
        delta = y - anchor if self.residual else y  # (B, T_out, H, W, C)
        B = y.shape[0]

        if isinstance(self.noise_std, torch.Tensor):
            std = self.noise_std.to(delta.device)[None, None, None, None, :]
        else:
            std = self.noise_std
        noise = std * torch.randn_like(delta)
        t = torch.rand(B, device=y.device, dtype=y.dtype) ** flow_time_power

        t_bcast = t[:, None, None, None, None]
        zt = (1.0 - t_bcast) * noise + t_bcast * delta

        # model sees full sequence; input frames are stationary anchors
        model_input = torch.cat([batch.input, zt], dim=1)  # (B, T_in + T_out, H, W, C)
        v_target = delta - noise  # (B, T_out, H, W, C)

        return model_input, t[:, None], v_target  # t: (B, 1) for AdaptiveLayerNorm

    def _predict(self, batch: CollatedBatch, flow_time_power: float = 1.0,
                 clean_anchor: torch.Tensor | None = None) -> FlowPrediction:
        model_input, flow_time, v_target = self._sample_flow(batch, flow_time_power, clean_anchor)
        num_input_frames = batch.input.shape[1]
        model_output, moe_outputs = self.model.step(
            model_input, batch.sim_params_tensor, flow_time, T_in=num_input_frames,
            condition=getattr(batch, "condition", None),
        )
        zt = model_input[:, num_input_frames:]
        remaining = remaining_flow_time(flow_time)
        # straight-line flow: velocity and endpoint are related by delta = zt + (1-t) * v
        delta = zt + remaining * v_target
        if self.prediction_target == "endpoint":
            delta_pred = model_output
            # the floor keeps the implied velocity finite as t -> 1; only diagnostics use it
            v_pred = (delta_pred - zt) / remaining.clamp(min=1e-3)
        else:
            v_pred = model_output
            delta_pred = zt + remaining * v_pred
        return FlowPrediction(flow_time, v_pred, v_target, delta_pred, delta, moe_outputs)

    @torch.no_grad()
    def _integrate_context(
        self,
        context: torch.Tensor,
        future_window: int,
        sim_params: torch.Tensor,
        n_steps: int,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Short Euler integration to produce a predicted frame for the AR context roll.
        Uses unaugmented context so the distribution matches inference."""
        B, T_in, H, W, C = context.shape
        if isinstance(self.noise_std, torch.Tensor):
            std = self.noise_std.to(context.device)[None, None, None, None, :]
        else:
            std = self.noise_std
        state = std * torch.randn(B, future_window, H, W, C,
                                  device=context.device, dtype=context.dtype)
        step_size = 1.0 / n_steps
        for step_idx in range(n_steps):
            t = torch.full((B, 1), step_idx * step_size,
                           device=context.device, dtype=context.dtype)
            model_input = torch.cat([context, state], dim=1)
            v, _ = self.model.step(model_input, sim_params, t,
                                   T_in=T_in, condition=condition)
            if self.prediction_target == "endpoint":
                velocity = (v - state) / max(1.0 - step_idx * step_size, 1e-3)
            else:
                velocity = v
            state = state + step_size * velocity
        return state  # delta_hat (residual) or full-field prediction

    def _field_weighted_mse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.field_loss_weights is None:
            return F.mse_loss(pred, target)
        # per-field MSE: mean over B, T, H, W; weighted sum over C
        per_field = (pred - target).pow(2).mean(dim=(0, 1, 2, 3))  # (C,)
        return (per_field * self.field_loss_weights).mean()

    def _trained_pair(self, prediction: FlowPrediction) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.prediction_target == "endpoint":
            return prediction.delta_pred, prediction.delta
        return prediction.v_pred, prediction.v_target

    def training_step(self, batch: CollatedBatch, batch_idx: int) -> torch.Tensor:
        torch.compiler.cudagraph_mark_step_begin()
        ar_steps = self._current_ar_steps()
        max_ar_steps = int(max(self.model_cfg.get("ar_rollout_schedule", [self.model_cfg.get("ar_rollout_steps", 1)])))
        future_window = batch.target.shape[1] // max_ar_steps

        current_input = batch.input
        total_loss = torch.zeros(1, device=batch.input.device, dtype=batch.input.dtype).squeeze()

        for k in range(ar_steps):
            target_k = batch.target[:, k * future_window : (k + 1) * future_window]
            batch_k = CollatedBatch(
                input=current_input,
                target=target_k,
                sim_params_dict=batch.sim_params_dict,
                x_grid=batch.x_grid,
                y_grid=batch.y_grid,
                dx=batch.dx,
                dy=batch.dy,
                sim_params_tensor=batch.sim_params_tensor,
                condition=batch.condition,
            )

            clean_anchor = current_input[:, -1:].clone() if self.residual else None
            # Only augment on the first step where context is ground truth
            if k == 0:
                batch_k = self._augment_context(batch_k)
            prediction = self._predict(batch_k, self.flow_time_power, clean_anchor)
            trained_pred, trained_target = self._trained_pair(prediction)

            data_loss = self._field_weighted_mse(trained_pred, trained_target)
            aux_loss, _ = self._router_loss(prediction.moe_outputs)
            total_loss = total_loss + data_loss + aux_loss
            self._update_router_bias(prediction.moe_outputs)

            if k < ar_steps - 1:
                n_euler = int(self.model_cfg.get("ar_context_euler_steps", 4))
                delta_hat = self._integrate_context(
                    current_input, future_window, batch.sim_params_tensor,
                    n_euler, condition=batch.condition,
                )
                predicted_frames = (current_input[:, -1:] + delta_hat
                                    if self.residual else delta_hat)
                current_input = torch.cat(
                    [current_input[:, future_window:], predicted_frames], dim=1
                )

        loss = total_loss / ar_steps
        log_dict = {
            "train/loss": loss,
            "train/data_loss": data_loss,
            "train/ar_steps": float(ar_steps),
            "train/step": self.global_step,
            "train/learning_rate": self.get_current_lr(),
        }
        log_dict = self.log_step_metrics(
            log_dict, trained_pred, trained_target,
            batch.dx[0].item(), batch.dy[0].item(), "train",
        )
        log_dict = self._moe_metrics(prediction.moe_outputs, log_dict, "train")
        self.default_log_dict(log_dict)
        return loss

    def validation_step(self, batch: CollatedBatch, batch_idx: int) -> torch.Tensor:
        # always uniform t, so validation averages stay comparable across training t distributions
        prediction = self._predict(batch)
        trained_pred, trained_target = self._trained_pair(prediction)
        loss = self._field_weighted_mse(trained_pred, trained_target)

        if batch_idx == 0:
            self.validation_sample = (batch.target.detach(), trained_target.detach(), trained_pred.detach())

        self._accumulate_flow_time_binned_errors(prediction)
        log_dict = {
            "val/loss": loss,
            "val/endpoint_mse": F.mse_loss(prediction.delta_pred, prediction.delta),
            # copy-last-frame error, computed directly so it means the same thing in both modes
            "val/persistence_mse": F.mse_loss(batch.input[:, -1:].expand_as(batch.target), batch.target),
        }
        log_dict = self.log_step_metrics(
            log_dict, trained_pred, trained_target,
            batch.dx[0].item(), batch.dy[0].item(), "val",
        )
        log_dict = self._moe_metrics(prediction.moe_outputs, log_dict, "val")
        self.default_log_dict(log_dict)
        return loss

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        num_bins = len(FLOW_TIME_BIN_EDGES) - 1
        self._flow_time_bin_boundaries = torch.tensor(FLOW_TIME_BIN_EDGES[1:-1], device=self.device)
        self._flow_time_bin_error_sums = {
            "loss": torch.zeros(num_bins, device=self.device),
            "endpoint": torch.zeros(num_bins, device=self.device),
        }
        self._flow_time_bin_count = torch.zeros(num_bins, device=self.device)

    def _accumulate_flow_time_binned_errors(self, prediction: FlowPrediction):
        bin_index = torch.bucketize(prediction.flow_time.flatten(), self._flow_time_bin_boundaries, right=True)
        errors = {
            "loss": per_sample_mse(prediction.v_pred, prediction.v_target),
            "endpoint": per_sample_mse(prediction.delta_pred, prediction.delta),
        }
        for name, error in errors.items():
            self._flow_time_bin_error_sums[name].index_add_(0, bin_index, error)
        self._flow_time_bin_count.index_add_(0, bin_index, torch.ones_like(errors["loss"]))

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        for bin_index, (low, high) in enumerate(zip(FLOW_TIME_BIN_EDGES[:-1], FLOW_TIME_BIN_EDGES[1:])):
            count = self._flow_time_bin_count[bin_index]
            if count == 0:
                continue
            for name, error_sums in self._flow_time_bin_error_sums.items():
                self.default_log(f"val_flow_time/{name}_t{low:.2f}-{high:.2f}", error_sums[bin_index] / count)
