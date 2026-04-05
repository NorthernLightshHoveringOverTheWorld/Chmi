from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from ssqueezepy import cwt, Wavelet

from audio_io import read_wav


def compute_cwt(
    file_path: str,
    *,
    fmin_hz: float = 500.0,
    fmax_hz: float = 20000.0,
    nv: int = 8,
    max_seconds: float | None = 1.0,
    downsample: int = 4,
):
    fs, x = read_wav(file_path, mono=True, normalize=True)
    if max_seconds is not None:
        max_n = int(max_seconds * fs)
        if max_n > 0 and len(x) > max_n:
            x = x[:max_n]
    if downsample > 1:
        x = x[::downsample]
        fs = int(fs / downsample)

    # Build scales targeting a log-spaced frequency grid.
    # NOTE: ssqueezepy doesn't expose a simple public scale<->freq helper;
    # this mapping matches what is already used in the project.
    freqs_hz = np.geomspace(fmin_hz, fmax_hz, num=max(8, int(nv * np.log2(fmax_hz / fmin_hz))))
    scales = 0.1 * fs / freqs_hz
    order = np.argsort(scales)
    scales = scales[order]
    freqs_hz = freqs_hz[order]

    wavelet = Wavelet("morlet", N=len(x))
    Wx, _ = cwt(x, wavelet=wavelet, scales=scales)
    t = np.arange(len(x)) / fs
    return fs, t, freqs_hz, Wx


def print_cwt_preview(t: np.ndarray, freqs_hz: np.ndarray, Wx: np.ndarray, *, max_freqs: int = 3, max_times: int = 10):
    f_n = min(int(max_freqs), Wx.shape[0])
    t_n = min(int(max_times), Wx.shape[1])
    print("\nCWT preview: Wx[f_idx, t_idx] (complex)")
    for fi in range(f_n):
        f = freqs_hz[fi]
        row = Wx[fi, :t_n]
        row_str = ", ".join(f"{z.real:+.4f}{z.imag:+.4f}j" for z in row)
        print(f"f={f:.1f} Hz: {row_str}")


def save_cwt(
    t: np.ndarray,
    freqs_hz: np.ndarray,
    Wx: np.ndarray,
    *,
    prefix: str = "cwt",
):
    np.save(f"{prefix}_t.npy", t)
    np.save(f"{prefix}_freqs_hz.npy", freqs_hz)
    np.save(f"{prefix}_Wx.npy", Wx)
    np.savetxt(f"{prefix}_amp.csv", np.abs(Wx), delimiter=",")


def plot_cwt_spectrogram(
    file_path: str | None = None,
    *,
    t: np.ndarray | None = None,
    freqs_hz: np.ndarray | None = None,
    Wx: np.ndarray | None = None,
    fmin_hz: float = 500.0,
    fmax_hz: float = 20000.0,
    nv: int = 8,
    max_seconds: float | None = 1.0,
    downsample: int = 4,
    freq_index: int | None = None,
    ax=None,
    show: bool = True,
):
    """
    Действительная часть CWT с вейвлетом Морле по выбранной строке частоты —
    для реального входного сигнала (не дельта-импульс).
    """
    if Wx is None:
        if file_path is None:
            raise ValueError("Нужен file_path или готовые t, freqs_hz, Wx из compute_cwt.")
        _fs, t, freqs_hz, Wx = compute_cwt(
            file_path,
            fmin_hz=fmin_hz,
            fmax_hz=fmax_hz,
            nv=nv,
            max_seconds=max_seconds,
            downsample=downsample,
        )
    else:
        if t is None or freqs_hz is None:
            raise ValueError("Вместе с Wx передайте t и freqs_hz.")

    fi = (Wx.shape[0] // 2) if freq_index is None else int(freq_index)
    fi = max(0, min(fi, Wx.shape[0] - 1))

    y_vals = np.real(Wx[fi, :])
    denom = np.max(np.abs(y_vals))
    if denom > 0:
        y_vals = y_vals / denom

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    ax.plot(t, y_vals, color="#0072BD", linewidth=1.5)
    f_hz = float(freqs_hz[fi])
    ax.set_title(f"Morlet CWT (Re), f ≈ {f_hz:.1f} Гц", fontweight="bold")
    ax.set_xlabel("Время (с)")
    ax.set_ylabel("Норм. амплитуда")
    ax.set_ylim(-1, 1)
    ax.grid(True, which="both", linestyle="-", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)

    if show:
        backend = matplotlib.get_backend().lower()
        if "agg" not in backend:
            plt.show()

    return t, y_vals


def main():
    default_wav = Path(__file__).resolve().parent / "sounds" / "10KHz.wav"
    plot_cwt_spectrogram(str(default_wav), show=True)


if __name__ == "__main__":
    main()
