import numpy as np
from pathlib import Path
from scipy.io import wavfile


def read_wav(file_path: str, *, mono: bool = True, normalize: bool = True):
    """
    Returns (sample_rate_hz, samples_float32).

    - Supports integer PCM WAV via scipy.io.wavfile.
    - If mono=True and file is multi-channel, takes channel 0.
    - If normalize=True, scales to [-1, 1] (best-effort for integer PCM).
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(file_path)
    if p.stat().st_size == 0:
        raise ValueError(f"WAV файл пустой: {file_path}")

    fs, data = wavfile.read(file_path)

    if mono and getattr(data, "ndim", 1) > 1:
        data = data[:, 0]

    x = np.asarray(data)

    if normalize:
        if np.issubdtype(x.dtype, np.integer):
            info = np.iinfo(x.dtype)
            denom = float(max(abs(info.min), info.max))
            x = x.astype(np.float32) / denom
        else:
            x = x.astype(np.float32)
            peak = float(np.max(np.abs(x))) if x.size else 0.0
            if peak > 0:
                x = x / peak
    else:
        x = x.astype(np.float32, copy=False)

    return fs, x
