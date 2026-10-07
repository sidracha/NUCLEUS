from typing import Tuple

import torch
import torch.nn.functional as F

from nucleus.data.batching import CollatedBatch
from nucleus.models.modules import MoEConditionedForecastModule


class FlowMatchingModule(MoEConditionedForecastModule):
    """
    physics-conditioned flow matching for forecasting

    z0 = (input, gaussian noise)    -- starting point, target slot is pure noise
    y  = (input, target)            -- endpoint, target slot is ground truth
    t ~ uniform(0, 1)

    zt = (1-t)*z0 + t*y = (input, (1-t)*noise + t*target)
        -- input frames are stationary anchors, only the target slot moves

    objective = min MSE(model(zt, t)[T_out frames], target - noise)
        -- model sees full (T_in + T_out) sequence via time concatenation
        -- only supervised on the last T_out frames
        -- t conditioned via dedicated per-block FiLM layer

    inference: start from z0 = (input, noise), integrate dz/dt = model(zt, t)
        -- ex. forward euler: zt+dt = zt + dt * model(zt, t)
        -- take last T_out frames of z1 as the forecast
    """

    def _sample_flow(
        self, batch: CollatedBatch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (model_input, t, v_target)."""
        y = batch.target  # (B, T_out, H, W, C)
        B = y.shape[0]

        noise = torch.randn_like(y)
        t = torch.rand(B, device=y.device, dtype=y.dtype)

        t_bcast = t[:, None, None, None, None]
        zt = (1.0 - t_bcast) * noise + t_bcast * y

        # model sees full sequence; input frames are stationary anchors
        model_input = torch.cat([batch.input, zt], dim=1)  # (B, T_in + T_out, H, W, C)
        v_target = y - noise  # (B, T_out, H, W, C)

        return model_input, t[:, None], v_target  # t: (B, 1) for AdaptiveLayerNorm

    def training_step(self, batch: CollatedBatch, batch_idx: int) -> torch.Tensor:
        model_input, t, v_target = self._sample_flow(batch)
        T_in = batch.input.shape[1]

        torch.compiler.cudagraph_mark_step_begin()
        v_pred, moe_outputs = self.model.step(model_input, batch.sim_params_tensor, t, T_in=T_in)

        data_loss = F.mse_loss(v_pred, v_target)
        aux_loss, _ = self._router_loss(moe_outputs)
        loss = data_loss + aux_loss
        self._update_router_bias(moe_outputs)

        log_dict = {
            "train/loss": loss,
            "train/data_loss": data_loss,
            "train/step": self.global_step,
            "train/learning_rate": self.get_current_lr(),
        }
        log_dict = self.log_step_metrics(
            log_dict, v_pred, v_target,
            batch.dx[0].item(), batch.dy[0].item(), "train",
        )
        log_dict = self._moe_metrics(moe_outputs, log_dict, "train")
        self.default_log_dict(log_dict)
        return loss

    def validation_step(self, batch: CollatedBatch, batch_idx: int) -> torch.Tensor:
        model_input, t, v_target = self._sample_flow(batch)
        T_in = batch.input.shape[1]

        v_pred, moe_outputs = self.model.step(model_input, batch.sim_params_tensor, t, T_in=T_in)
        loss = F.mse_loss(v_pred, v_target)

        if batch_idx == 0:
            self.validation_sample = (batch.target.detach(), v_target.detach(), v_pred.detach())

        log_dict = {"val/loss": loss}
        log_dict = self.log_step_metrics(
            log_dict, v_pred, v_target,
            batch.dx[0].item(), batch.dy[0].item(), "val",
        )
        log_dict = self._moe_metrics(moe_outputs, log_dict, "val")
        self.default_log_dict(log_dict)
        return loss
