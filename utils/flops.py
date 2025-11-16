import torch
from typing import Optional

#: estimated FP32 FMA operations per SM per cycle for common compute capabilities
_FP32_FMA_PER_SM = {
    7: 128,  # Turing
    8: 256,  # Ampere
    9: 512,  # Hopper
}


def _guess_fma_per_cycle(major: int) -> int:
    """Pick a conservative throughput estimate for the given compute capability."""
    if major in _FP32_FMA_PER_SM:
        return _FP32_FMA_PER_SM[major]
    # fall back to the lowest capacity we expect
    return 128


def get_peak_flops_per_second(
    device: Optional[torch.device] = None,
    clock_rate_hz: Optional[float] = None,
) -> float:
    """
    Returns an estimate of the device's FP32 FLOPs per second.

    The calculation is based on SM count, clock rate, and an estimated number
    of FP32 FMAs per SM per cycle.  Each FMA counts as 2 FLOPs.
    """
    if device is None:
        device = torch.device("cuda")
    prop = torch.cuda.get_device_properties(device)
    fma_per_sm = _guess_fma_per_cycle(prop.major)
    flops_per_sm_per_cycle = float(fma_per_sm * 2)
    if clock_rate_hz is None:
        clock_attr = getattr(prop, "clock_rate", None)
        if clock_attr is not None:
            clock_rate_hz = float(clock_attr) * 1_000
        else:
            clock_rate_hz = 1.5e9
    peak_flops = (
        prop.multi_processor_count * flops_per_sm_per_cycle * clock_rate_hz
    )
    return peak_flops


def format_flops(flops: float) -> str:
    """Pretty-print a FLOPs/sec value."""
    if flops >= 1e12:
        return f"{flops / 1e12:.2f} TFLOPs"
    if flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOPs"
    if flops >= 1e6:
        return f"{flops / 1e6:.2f} MFLOPs"
    return f"{flops:.2f} FLOPs"
