from enum import Enum
from typing import List, Optional, Tuple

import torch

from model import EchoDiT


def _get_uncond_text_input_ids_and_mask(
    batch_size: int, max_length: int, device: str | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unconditional text ids/mask (BOS only)."""
    text_input_ids_uncond = torch.zeros((batch_size, max_length), dtype=torch.int32)
    text_mask_uncond = torch.zeros((batch_size, max_length), dtype=torch.bool)
    text_mask_uncond[:, 0] = True
    if device is not None:
        text_input_ids_uncond = text_input_ids_uncond.to(device)
        text_mask_uncond = text_mask_uncond.to(device)
    return text_input_ids_uncond, text_mask_uncond


def _temporal_score_rescale(
    v_pred: torch.Tensor, x_t: torch.Tensor, t: float, rescale_k: float, rescale_sigma: float
) -> torch.Tensor:
    if t < 1:
        snr = (1 - t) ** 2 / (t**2)
        ratio = (snr * rescale_sigma**2 + 1) / (snr * rescale_sigma**2 / rescale_k + 1)
        return 1 / (1 - t) * (ratio * ((1 - t) * v_pred + x_t) - x_t)
    return v_pred


def _get_first_n_kv_cache(kv_cache: List[Tuple[torch.Tensor, torch.Tensor]], n: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    return [(k[:n], v[:n]) for (k, v) in kv_cache]


def _multiply_speaker_kv_cache(
    kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
    scale: float,
    text_length: int,
    max_layers: int = 24,
) -> None:
    """Scale speaker kv cache in-place (speaker keys start after text keys)."""
    for i in range(min(max_layers, len(kv_cache))):
        k, v = kv_cache[i]
        k.mul_(scale)
        v.mul_(scale)


class GuidanceMode(Enum):
    INDEPENDENT = "independent"


@torch.inference_mode()
def sample_euler_cfg_independent_guidances(
    model: EchoDiT,
    speaker_latent: torch.Tensor,
    speaker_mask: torch.Tensor,
    text_input_ids: torch.Tensor,
    text_mask: torch.Tensor,
    rng_seed: int,
    block_sizes: List[int],
    num_steps: int | List[int],
    cfg_scale_text: float,
    cfg_scale_speaker: float,
    cfg_min_t: float,
    cfg_max_t: float,
    truncation_factor: float | List[float] | None,
    init_scale: float | List[float] | None = None,
    rescale_k: float | None = None,
    rescale_sigma: float | None = None,
    speaker_kv_scale: float | None = None,
    speaker_kv_max_layers: int | None = None,
    speaker_kv_min_t: float | None = None,
    early_stop_on_zero: bool = False,
    zero_eps: float = 0.0,
    zero_tail_min_frac: float = 1.0,
    zero_tail_frames: int = 64,
    text_input_ids_first_block: Optional[torch.Tensor] = None,
    text_mask_first_block: Optional[torch.Tensor] = None,
    profile_stats: Optional[object] = None,
) -> torch.Tensor:
    torch.manual_seed(rng_seed)

    device, dtype = model.device, model.dtype
    batch_size = text_input_ids.shape[0]

    if isinstance(num_steps, int):
        step_counts = [num_steps] * len(block_sizes)
    else:
        step_counts = num_steps
        assert len(step_counts) == len(block_sizes), (
            f"num_steps list length ({len(step_counts)}) must match block_sizes length ({len(block_sizes)})"
        )

    if truncation_factor is None or isinstance(truncation_factor, float):
        truncation_factors = [truncation_factor] * len(block_sizes)
    else:
        truncation_factors = truncation_factor
        assert len(truncation_factors) == len(block_sizes), (
            f"truncation_factor list length ({len(truncation_factors)}) must match block_sizes length ({len(block_sizes)})"
        )

    default_init_scale = 0.999
    if init_scale is None or isinstance(init_scale, float):
        init_scales = [default_init_scale if init_scale is None else init_scale] * len(block_sizes)
    else:
        init_scales = init_scale
        assert len(init_scales) == len(block_sizes), (
            f"init_scale list length ({len(init_scales)}) must match block_sizes length ({len(block_sizes)})"
        )

    text_input_ids_uncond, text_mask_uncond = _get_uncond_text_input_ids_and_mask(
        text_input_ids.shape[0], text_input_ids.shape[1], device=device
    )

    speaker_latent_uncond, speaker_mask_uncond = torch.zeros_like(speaker_latent), torch.zeros_like(speaker_mask)

    full_text_input_ids = torch.cat([text_input_ids, text_input_ids_uncond, text_input_ids], dim=0)
    full_text_mask = torch.cat([text_mask, text_mask_uncond, text_mask], dim=0)

    full_speaker_latent = torch.cat([speaker_latent, speaker_latent, speaker_latent_uncond], dim=0)
    full_speaker_mask = torch.cat([speaker_mask, speaker_mask, speaker_mask_uncond], dim=0)

    if text_input_ids_first_block is not None and text_mask_first_block is not None:
        text_input_ids_first_uncond, text_mask_first_uncond = _get_uncond_text_input_ids_and_mask(
            text_input_ids_first_block.shape[0], text_input_ids_first_block.shape[1], device=device
        )
        full_text_input_ids_first = torch.cat(
            [text_input_ids_first_block, text_input_ids_first_uncond, text_input_ids_first_block], dim=0
        )
        full_text_mask_first = torch.cat(
            [text_mask_first_block, text_mask_first_uncond, text_mask_first_block], dim=0
        )
    else:
        full_text_input_ids_first = None
        full_text_mask_first = None

    kv_cache_text_full = None
    kv_cache_text = None
    if full_text_input_ids_first is not None:
        kv_cache_text_full_first = model.get_text_kv_cache(full_text_input_ids_first, full_text_mask_first)
        kv_cache_text_first = _get_first_n_kv_cache(kv_cache_text_full_first, batch_size)
    else:
        kv_cache_text_full_first = None
        kv_cache_text_first = None

    kv_cache_speaker_full = model.get_speaker_kv_cache(full_speaker_latent.to(dtype))
    kv_cache_speaker = _get_first_n_kv_cache(kv_cache_speaker_full, batch_size)

    prefix_total = sum(block_sizes)
    prefix_latent = torch.zeros((batch_size, prefix_total, 80), device=device, dtype=torch.float32)

    pos_id = 0
    early_stop = False

    for block_idx, (block_size, block_num_steps) in enumerate(zip(block_sizes, step_counts)):
        block_trunc = truncation_factors[block_idx]
        block_init_scale = init_scales[block_idx]

        t_schedule = torch.linspace(1.0, 0.0, block_num_steps + 1, device=device) * block_init_scale

        if block_idx == 0 and kv_cache_text_full_first is not None:
            block_text_full_mask = full_text_mask_first
            block_text_mask = text_mask_first_block
            block_kv_cache_text_full = kv_cache_text_full_first
            block_kv_cache_text = kv_cache_text_first
        else:
            if kv_cache_text_full is None:
                kv_cache_text_full = model.get_text_kv_cache(full_text_input_ids, full_text_mask)
                kv_cache_text = _get_first_n_kv_cache(kv_cache_text_full, batch_size)
            block_text_full_mask = full_text_mask
            block_text_mask = text_mask
            block_kv_cache_text_full = kv_cache_text_full
            block_kv_cache_text = kv_cache_text

        if speaker_kv_scale is not None:
            _multiply_speaker_kv_cache(
                kv_cache_speaker_full, speaker_kv_scale, text_input_ids.shape[-1], speaker_kv_max_layers
            )

        full_prefix_latent = torch.cat([prefix_latent, prefix_latent, prefix_latent], dim=0)
        kv_cache_latent_full = model.get_latent_kv_cache(full_prefix_latent.to(dtype))
        kv_cache_latent = _get_first_n_kv_cache(kv_cache_latent_full, batch_size)

        x_t = torch.randn((batch_size, block_size, 80), device=device, dtype=torch.float32)
        if block_trunc is not None:
            x_t = x_t * block_trunc

        for i in range(block_num_steps):
            t, t_next = t_schedule[i], t_schedule[i + 1]

            has_cfg = ((t >= cfg_min_t) * (t <= cfg_max_t)).item()

            if has_cfg:
                v_cond, v_uncond_text, v_uncond_speaker = model(
                    x=torch.cat([x_t, x_t, x_t], dim=0).to(dtype),
                    t=(torch.ones((batch_size * 3,), device=device) * t).to(dtype),
                    text_mask=block_text_full_mask,
                    speaker_mask=full_speaker_mask,
                    start_pos=pos_id,
                    kv_cache_text=block_kv_cache_text_full,
                    kv_cache_speaker=kv_cache_speaker_full,
                    kv_cache_latent=kv_cache_latent_full,
                ).float().chunk(3, dim=0)

                v_pred = (
                    v_cond
                    + cfg_scale_text * (v_cond - v_uncond_text)
                    + cfg_scale_speaker * (v_cond - v_uncond_speaker)
                )
            else:
                v_pred = model(
                    x=x_t.to(dtype),
                    t=(torch.ones((batch_size,), device=device) * t).to(dtype),
                    text_mask=block_text_mask,
                    speaker_mask=speaker_mask,
                    start_pos=pos_id,
                    kv_cache_text=block_kv_cache_text,
                    kv_cache_speaker=kv_cache_speaker,
                    kv_cache_latent=kv_cache_latent,
                ).float()

            if rescale_k is not None and rescale_sigma is not None:
                v_pred = _temporal_score_rescale(v_pred, x_t, float(t), rescale_k, rescale_sigma)

            if (
                speaker_kv_scale is not None
                and speaker_kv_min_t is not None
                and t_next < speaker_kv_min_t
                and t >= speaker_kv_min_t
            ):
                _multiply_speaker_kv_cache(
                    kv_cache_speaker_full, 1.0 / speaker_kv_scale, text_input_ids.shape[-1], speaker_kv_max_layers
                )

            x_t = x_t + v_pred * (t_next - t)

        prefix_latent[:, pos_id : pos_id + block_size] = x_t
        pos_id += block_size

        if early_stop_on_zero:
            tail_len = min(zero_tail_frames, x_t.shape[1])
            tail = x_t[:, -tail_len:]
            tail_abs = torch.abs(tail)
            zero_frac = float((tail_abs <= zero_eps).float().mean().item())
            tail_absmax = float(tail_abs.max().item())
            zero_ok = zero_frac >= zero_tail_min_frac and tail_absmax <= zero_eps
            if zero_ok:
                early_stop = True
            if profile_stats is not None:
                profile_stats.zero_check = {
                    "tail_len": tail_len,
                    "zero_frac": zero_frac,
                    "tail_absmax": tail_absmax,
                    "zero_eps": zero_eps,
                    "zero_tail_min_frac": zero_tail_min_frac,
                }
        if early_stop:
            break

    return prefix_latent


def sample_euler_cfg_any(
    model: EchoDiT,
    speaker_latent: torch.Tensor,
    speaker_mask: torch.Tensor,
    text_input_ids: torch.Tensor,
    text_mask: torch.Tensor,
    rng_seed: int,
    block_sizes: List[int],
    guidance_mode: GuidanceMode,
    num_steps: int,
    cfg_scale_text: float,
    cfg_scale_speaker: float | None,
    cfg_min_t: float,
    cfg_max_t: float,
    truncation_factor: float | List[float] | None,
    init_scale: float | List[float] | None = None,
    rescale_k: float | None = None,
    rescale_sigma: float | None = None,
    speaker_kv_scale: float | None = None,
    speaker_kv_min_t: float | None = None,
    speaker_kv_max_layers: int | None = None,
    apg_eta_text: float | None = None,
    apg_eta_speaker: float | None = None,
    apg_momentum_text: float | None = None,
    apg_momentum_speaker: float | None = None,
    apg_norm_text: float | None = None,
    apg_norm_speaker: float | None = None,
    early_stop_on_zero: bool = False,
    zero_eps: float = 0.0,
    zero_tail_min_frac: float = 1.0,
    zero_tail_frames: int = 64,
    profile_stats: Optional[object] = None,
    text_input_ids_first_block: Optional[torch.Tensor] = None,
    text_mask_first_block: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if guidance_mode != GuidanceMode.INDEPENDENT:
        raise ValueError(f"Unsupported guidance_mode for API path: {guidance_mode}")
    if cfg_scale_speaker is None:
        raise ValueError("cfg_scale_speaker must be provided for independent guidances")

    return sample_euler_cfg_independent_guidances(
        model=model,
        speaker_latent=speaker_latent,
        speaker_mask=speaker_mask,
        text_input_ids=text_input_ids,
        text_mask=text_mask,
        rng_seed=rng_seed,
        block_sizes=block_sizes,
        num_steps=num_steps,
        cfg_scale_text=cfg_scale_text,
        cfg_scale_speaker=cfg_scale_speaker,
        cfg_min_t=cfg_min_t,
        cfg_max_t=cfg_max_t,
        truncation_factor=truncation_factor,
        init_scale=init_scale,
        rescale_k=rescale_k,
        rescale_sigma=rescale_sigma,
        speaker_kv_scale=speaker_kv_scale,
        speaker_kv_max_layers=speaker_kv_max_layers,
        speaker_kv_min_t=speaker_kv_min_t,
        early_stop_on_zero=early_stop_on_zero,
        zero_eps=zero_eps,
        zero_tail_min_frac=zero_tail_min_frac,
        zero_tail_frames=zero_tail_frames,
        profile_stats=profile_stats,
        text_input_ids_first_block=text_input_ids_first_block,
        text_mask_first_block=text_mask_first_block,
    )
