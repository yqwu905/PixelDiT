import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from diffusion import Scheduler


class DummySRModel(torch.nn.Module):
    def forward(self, x, t=None, timestep=None, y=None, mask=None, data_info=None, repa_tokens=None, sr_condition=None, **kwargs):
        pred = torch.zeros_like(x)
        if sr_condition is not None:
            pred = pred + 0.1 * sr_condition
        return {"x": pred}


def main():
    torch.manual_seed(0)
    diffusion = Scheduler(
        "16",
        noise_schedule="linear_flow",
        predict_flow_v=True,
        learn_sigma=False,
        pred_sigma=False,
        snr=False,
        flow_shift=1.0,
    )
    model = DummySRModel()

    clean = torch.randn(2, 3, 32, 32)
    timesteps = torch.randint(0, 16, (2,)).long()

    low4 = torch.nn.functional.interpolate(clean, scale_factor=0.25, mode="bilinear", align_corners=False)
    sr_base = torch.nn.functional.interpolate(low4, size=clean.shape[-2:], mode="bilinear", align_corners=False)
    residual = clean - sr_base

    low2 = torch.nn.functional.interpolate(clean, scale_factor=0.5, mode="bilinear", align_corners=False)
    loss = diffusion.training_losses(
        model,
        residual,
        timesteps,
        model_kwargs={
            "y": torch.zeros(2, 1, 4, 8),
            "mask": torch.ones(2, 1, 1, 4),
            "sr_condition": sr_base,
            "sr_base": sr_base,
            "sr_targets": [low4, low2, clean],
            "sr_loss_config": {"weight": 0.2, "per_scale_weight_decay": 0.5},
        },
    )
    assert torch.isfinite(loss["loss"]).all(), "training loss contains non-finite values"
    assert "extra" in loss and "sr_aux_loss" in loss["extra"], "sr_aux_loss should be exposed in extra"
    print("smoke test passed", float(loss["loss"].mean().item()))


if __name__ == "__main__":
    main()
