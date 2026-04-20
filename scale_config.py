from pathlib import Path

import numpy as np


def _default_scales() -> np.ndarray:
    # Conservative fallback if text file is missing or invalid.
    return np.geomspace(2.0, 5986.0, 116).astype(np.float64)


def load_scales_from_txt(path: str | None = None) -> np.ndarray:
    """
    Load CWT scales from a text file.

    File format:
    - one scale value per line
    - empty lines are allowed
    - lines starting with '#' are ignored
    """
    cfg_path = Path(path) if path else Path(__file__).resolve().parent / "wavelet_scales.txt"
    if not cfg_path.exists():
        return _default_scales()

    values: list[float] = []
    with cfg_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            values.append(float(line.replace(",", ".")))

    scales = np.asarray(values, dtype=np.float64)
    if scales.size == 0:
        return _default_scales()
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError(f"Invalid scale values in '{cfg_path}'. All scales must be finite and > 0.")

    return scales
