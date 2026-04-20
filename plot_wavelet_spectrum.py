from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import ScalarFormatter
from scipy.signal import resample_poly

from audio_io import read_wav
from scale_config import load_scales_from_txt


def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (int(n - 1).bit_length())


def _morlet_fft(freqs_fft_hz: np.ndarray, scale: float, fs: float, w0: float = 2.0 * np.pi) -> np.ndarray:
    """
    Fourier-domain Morlet (аналитический). Шкала scale согласована с
    scales = w0*fs/(2π f_center) при цифровой частоте ω_d = 2πf/fs (рад/отсчёт).
    """
    omega_d = 2.0 * np.pi * freqs_fft_hz / fs
    psi_hat = np.exp(-0.5 * ((scale * omega_d - w0) ** 2))
    psi_hat = psi_hat * np.sqrt(scale)
    psi_hat[freqs_fft_hz < 0] = 0.0
    return psi_hat


def compute_cwt(
    file_path: str,
    *,
    fmin_hz: float = 0.0,
    fmax_hz: float = 100000.0,
    nv: int = 8,
    max_seconds: float | None = 1.0,
    downsample: int = 1,
    target_fs: int = 192000,
    scales_path: str | None = None,
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

    w0 = 2.0 * np.pi
    # Scales are controlled from external text config and reused in all CWT calls.
    scales = load_scales_from_txt(scales_path)
    freqs_hz = w0 * fs / (2.0 * np.pi * scales)
    band = (freqs_hz >= float(fmin_hz)) & (freqs_hz <= float(fmax_hz))
    if not np.any(band):
        raise ValueError(
            f"Нет шкал в диапазоне {fmin_hz:.1f}..{fmax_hz:.1f} Гц. "
            "Проверьте wavelet_scales.txt."
        )
    scales = scales[band]
    freqs_hz = freqs_hz[band]

    n = len(x)
    n_fft = _next_pow2(2 * n - 1)
    X = np.fft.fft(x, n=n_fft)
    f_fft = np.fft.fftfreq(n_fft, d=1.0 / fs)

    Wx = np.empty((len(scales), n), dtype=np.complex64)
    for i, s in enumerate(scales):
        psi_hat = _morlet_fft(f_fft, s, fs=float(fs), w0=w0)
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
    fmax_hz: float = 100000.0,
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
    fmax_hz: float = 100000.0,
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
        pos_freqs = freqs_hz[freqs_hz > 0]
        if pos_freqs.size == 0:
            raise ValueError("Для логарифмической шкалы нужны частоты > 0 Гц.")
        ax.set_yscale("log")
        ax.set_ylim(float(np.min(pos_freqs)), float(np.max(pos_freqs)))
        # Keep human-readable frequencies on the log axis (avoid scientific notation).
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(ScalarFormatter())
    ax.set_ylabel("Частота (Гц)")
    ax.set_xlabel("Время (с)")
    ax.set_title("Morlet CWT скалограмма |Wx|", fontweight="bold")
    plt.colorbar(m, ax=ax, label="Амплитуда")

    if show:
        backend = matplotlib.get_backend().lower()
        if "agg" not in backend:
            plt.show()

    return t, freqs_hz, amp


def _default_xlim_us_morlet_slice(
    t: np.ndarray, f_hz: float | None
) -> tuple[float, float]:
    """
    Окно по оси времени в мкс. ±4 мкс уместны при fs ~100 МГц (много отсчётов в окне).
    При fs ~192 кГц шаг ~5.2 мкс: в ±4 мкс всего 1–2 точки — график вырождается в ломаную.
    Берём max(±4 мкс, ~40 шагов дискретизации, ~1.5 периода несущей полосы).
    """
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return (-4.0, 4.0)
    by_samples = 40.0 * dt * 1e6
    by_period = (1.5 * 1e6 / float(f_hz)) if (f_hz is not None and f_hz > 0) else 0.0
    half_us = max(4.0, by_samples, by_period)
    return (-half_us, half_us)


def plot_longitudinal_slice_morlet_re(
    t: np.ndarray,
    wx_row: np.ndarray,
    *,
    f_hz: float | None = None,
    ax=None,
    t_center_s: float | None = None,
    xlim_us: tuple[float, float] | None = None,
    show: bool = True,
):
    """
    Продольный срез одной полосы CWT: нормированная Re(Wx) во времени.
    Для δ(t−t₀) форма — как материнский вейвлет Морле (как test2.py).
    Ось — увеличение вокруг центра вейвлета (мкс/мс/с), не длительность всей записи.
    Возвращает (t−t₀) в секундах, y_vals.
    """
    y_vals = np.real(np.asarray(wx_row, dtype=np.complex128))
    denom = np.max(np.abs(y_vals))
    if denom > 0:
        y_vals = y_vals / denom

    if t_center_s is None:
        t0 = float(t[int(np.argmax(np.abs(y_vals)))])
    else:
        t0 = float(t_center_s)
    t_rel_s = t - t0

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    if xlim_us is None:
        xlim_us = _default_xlim_us_morlet_slice(t, f_hz)

    half_w_s = max(abs(xlim_us[0]), abs(xlim_us[1])) * 1e-6
    # Единицы оси: мкс/мс/с — не путать с длительностью всей записи (секунды звучания).
    if half_w_s >= 0.1:
        x_plot = t_rel_s
        xlim_plot = (xlim_us[0] * 1e-6, xlim_us[1] * 1e-6)
        x_unit = "с"
    elif half_w_s >= 1e-3:
        x_plot = t_rel_s * 1e3
        xlim_plot = (xlim_us[0] * 1e-3, xlim_us[1] * 1e-3)
        x_unit = "мс"
    else:
        x_plot = t_rel_s * 1e6
        xlim_plot = (xlim_us[0], xlim_us[1])
        x_unit = "мкс"

    t_total = float(t[-1] - t[0])
    ax.plot(x_plot, y_vals, color="#0072BD", linewidth=1.5, label="Morlet Wavelet (Re)")
    title = "Morlet Wavelet"
    if f_hz is not None:
        title = f"{title}\nf ≈ {float(f_hz):.1f} Гц"
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(
        f"Время относительно центра ({x_unit})\n"
        f"фрагмент вокруг вейвлета; длительность всей записи {t_total:.3f} с",
        fontsize=9,
    )
    ax.set_ylabel("Норм. амплитуда")
    ax.set_xlim(xlim_plot)
    ax.set_ylim(-1, 1)
    ax.grid(True, which="both", linestyle="-", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.legend(loc="upper right", fontsize=8)

    if show:
        backend = matplotlib.get_backend().lower()
        if "agg" not in backend:
            plt.show()

    return t_rel_s, y_vals


def save_longitudinal_slice_txt(
    t: np.ndarray,
    wx_row: np.ndarray,
    *,
    path: str,
    f_hz: float | None = None,
    t_center_s: float | None = None,
):
    """
    Записывает продольный срез одной полосы CWT в TXT (табуляция).

    Колонки: t_s, t_rel_us (мкс), Re_Wx. В начале — комментарии # с метаданными.
    """
    t = np.asarray(t, dtype=np.float64)
    z = np.asarray(wx_row, dtype=np.complex128)
    re = np.real(z)

    if t_center_s is None:
        t0 = float(t[int(np.argmax(np.abs(re)))])
    else:
        t0 = float(t_center_s)
    t_rel_us = (t - t0) * 1e6

    lines = [
        f"# longitudinal_slice Morlet CWT",
        f"# f_center_hz\t{f_hz if f_hz is not None else 'nan'}",
        f"# t_center_s\t{t0:.12g}",
        f"# duration_recording_s\t{float(t[-1] - t[0]):.12g}",
        "# columns: t_s\tt_rel_us\tRe_Wx",
    ]
    data = np.column_stack([t, t_rel_us, re])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
        np.savetxt(f, data, fmt="%.10g", delimiter="\t")


def main():
    default_wav = Path(__file__).resolve().parent / "sounds" / "10KHz.wav"
    plot_cwt_spectrogram(str(default_wav), show=True)


if __name__ == "__main__":
    main()
