from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import resample_poly

from audio_io import read_wav


def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (int(n - 1).bit_length())


def _morlet_fft(freqs_fft_hz: np.ndarray, scale: float, w0: float = 6.0) -> np.ndarray:
    """
    Fourier-domain Morlet wavelet (approx. admissible for w0>=5).
    Positive-frequency analytic variant for stable TF ridges.
    """
    omega = 2.0 * np.pi * freqs_fft_hz
    psi_hat = np.exp(-0.5 * ((scale * omega - w0) ** 2))
    psi_hat = psi_hat * np.sqrt(scale)
    psi_hat[freqs_fft_hz < 0] = 0.0
    return psi_hat


def compute_cwt(
    file_path: str,
    *,
    fmin_hz: float = 500.0,
    fmax_hz: float = 96000.0,
    nv: int = 8,
    max_seconds: float | None = 1.0,
    downsample: int = 1,
    target_fs: int = 192000,
):
    fs, x = read_wav(file_path, mono=True, normalize=True)
    if target_fs > 0 and fs != int(target_fs):
        x = resample_poly(x, int(target_fs), int(fs)).astype(np.float32, copy=False)
        fs = int(target_fs)

    if max_seconds is not None:
        max_n = int(max_seconds * fs)
        if max_n > 0 and len(x) > max_n:
            x = x[:max_n]
    if downsample > 1:
        x = x[::downsample]
        fs = int(fs / downsample)

    nyquist_hz = 0.5 * fs
    if fmax_hz > nyquist_hz:
        fmax_hz = nyquist_hz

    # Build scales targeting a log-spaced frequency grid.
    freqs_hz = np.geomspace(fmin_hz, fmax_hz, num=max(8, int(nv * np.log2(fmax_hz / fmin_hz))))
    w0 = 6.0
    scales = w0 * fs / (2.0 * np.pi * freqs_hz)
    order = np.argsort(scales)
    scales = scales[order]
    freqs_hz = freqs_hz[order]

    n = len(x)
    n_fft = _next_pow2(2 * n - 1)
    X = np.fft.fft(x, n=n_fft)
    f_fft = np.fft.fftfreq(n_fft, d=1.0 / fs)

    Wx = np.empty((len(scales), n), dtype=np.complex64)
    for i, s in enumerate(scales):
        psi_hat = _morlet_fft(f_fft, s, w0=w0)
        conv = np.fft.ifft(X * np.conj(psi_hat))
        Wx[i, :] = conv[:n]

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


def save_time_frequency_coords(
    t: np.ndarray,
    freqs_hz: np.ndarray,
    Wx: np.ndarray,
    *,
    path: str,
):
    """
    Сохраняет 2 колонки: время (с) и частота (Гц) для каждого момента времени.
    Частота выбирается как argmax по |Wx| (гребень скалограммы).
    """
    amp = np.abs(Wx)
    ridge_idx = np.argmax(amp, axis=0)  # shape: (time,)
    ridge_freqs = freqs_hz[ridge_idx]
    data = np.column_stack([t, ridge_freqs])
    np.savetxt(path, data, fmt="%.10g", delimiter="\t", header="t_s\tfreq_hz")


def save_time_frequency_coords_csv(
    t: np.ndarray,
    freqs_hz: np.ndarray,
    Wx: np.ndarray,
    *,
    path: str,
):
    """То же, что save_time_frequency_coords, но в CSV (delimiter=','), 2 колонки."""
    amp = np.abs(Wx)
    ridge_idx = np.argmax(amp, axis=0)  # shape: (time,)
    ridge_freqs = freqs_hz[ridge_idx]
    data = np.column_stack([t, ridge_freqs])
    np.savetxt(path, data, fmt="%.10g", delimiter=",", header="t_s,freq_hz")


def plot_cwt_spectrogram(
    file_path: str | None = None,
    *,
    t: np.ndarray | None = None,
    freqs_hz: np.ndarray | None = None,
    Wx: np.ndarray | None = None,
    fmin_hz: float = 500.0,
    fmax_hz: float = 96000.0,
    nv: int = 8,
    max_seconds: float | None = 2.0,
    downsample: int = 1,
    target_fs: int = 192000,
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
            target_fs=target_fs,
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


def plot_cwt_scalogram(
    file_path: str | None = None,
    *,
    t: np.ndarray | None = None,
    freqs_hz: np.ndarray | None = None,
    Wx: np.ndarray | None = None,
    fmin_hz: float = 500.0,
    fmax_hz: float = 96000.0,
    nv: int = 8,
    max_seconds: float | None = None,
    downsample: int = 1,
    target_fs: int = 192000,
    ax=None,
    show: bool = True,
    log_freq: bool = True,
):
    """
    Скалограмма Morlet CWT: |Wx| по времени и частоте (оконный по сути TF-анализ).
    Возвращает (t_sec, freqs_hz, amp).
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
            target_fs=target_fs,
        )
    else:
        if t is None or freqs_hz is None:
            raise ValueError("Вместе с Wx передайте t и freqs_hz.")

    amp = np.abs(Wx)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    m = ax.pcolormesh(t, freqs_hz, amp, shading="auto")
    if log_freq:
        ax.set_yscale("log")
    ax.set_ylabel("Частота (Гц)")
    ax.set_xlabel("Время (с)")
    ax.set_title("Morlet CWT скалограмма |Wx|", fontweight="bold")
    plt.colorbar(m, ax=ax, label="Амплитуда")

    if show:
        backend = matplotlib.get_backend().lower()
        if "agg" not in backend:
            plt.show()

    return t, freqs_hz, amp


def main():
    default_wav = Path(__file__).resolve().parent / "sounds" / "10KHz.wav"
    plot_cwt_spectrogram(str(default_wav), show=True)


if __name__ == "__main__":
    main()
