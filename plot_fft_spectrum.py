import numpy as np
import matplotlib.pyplot as plt
from audio_io import read_wav
import matplotlib
from pathlib import Path


def plot_fft_spectrum(file_path, *, ax=None, show: bool = True):
    fs, data = read_wav(file_path, mono=True, normalize=True)

    window = np.hanning(len(data))
    data = data * window

    fft_data = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(fft_data), 1 / fs)

    mask = freqs > 0
    freqs = freqs[mask]
    fft_data = np.abs(fft_data[mask])

    band = (freqs >= 500) & (freqs <= 20000)
    freqs_band = freqs[band]
    fft_band = fft_data[band]

    peak_idx = np.argmax(fft_band)
    peak_freq = freqs_band[peak_idx]
    peak_amp = fft_band[peak_idx]

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    ax.plot(freqs_band, fft_band)
    ax.set_xlabel("Частота (Гц)")
    ax.set_ylabel("Амплитуда")
    ax.set_title(f"FFT спектр (500–20000 Гц) - {file_path}")
    ax.grid(True)

    ax.plot(peak_freq, peak_amp, "ro", markersize=6, label=f"Пик: {peak_freq:.1f} Гц")
    ax.axvline(x=peak_freq, color="r", linestyle="--", alpha=0.5)

    ax.annotate(
        f"{peak_freq:.1f} Гц",
        xy=(peak_freq, peak_amp),
        xytext=(peak_freq + (freqs_band[-1] - freqs_band[0]) * 0.05, peak_amp * 0.8),
        arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
    )

    ax.legend()
    if show:
        backend = matplotlib.get_backend().lower()
        if "agg" not in backend:
            plt.show()

    return freqs, fft_data


def plot_stft_spectrogram(
    file_path: str,
    *,
    ax=None,
    show: bool = True,
    nperseg: int = 4096,
    noverlap: int = 3072,
    fmin_hz: float = 0.0,
    fmax_hz: float = 100000.0,
    db: bool = True,
):
    """
    Оконное БПФ (STFT) со спектрограммой.
    Возвращает (t_sec, freqs_hz, S) где S = magnitude (или dB, если db=True).
    """
    fs, x = read_wav(file_path, mono=True, normalize=True)
    x = np.asarray(x, dtype=np.float32)

    nperseg = int(nperseg)
    noverlap = int(noverlap)
    if nperseg <= 0:
        raise ValueError("nperseg must be > 0")
    if noverlap < 0 or noverlap >= nperseg:
        raise ValueError("noverlap must be in [0, nperseg)")

    hop = nperseg - noverlap
    if x.size < nperseg:
        pad = nperseg - x.size
        x = np.pad(x, (0, pad))

    n_frames = 1 + (x.size - nperseg) // hop
    window = np.hanning(nperseg).astype(np.float32)

    # Frame and FFT.
    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(n_frames, nperseg),
        strides=(x.strides[0] * hop, x.strides[0]),
        writeable=False,
    )
    X = np.fft.rfft(frames * window[None, :], axis=1)
    mag = np.abs(X).T  # (freq, time)

    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    t = (np.arange(n_frames) * hop + (nperseg / 2.0)) / fs

    band = (freqs >= fmin_hz) & (freqs <= fmax_hz)
    freqs_b = freqs[band]
    mag_b = mag[band, :]

    if db:
        S = 20.0 * np.log10(np.maximum(mag_b, 1e-12))
    else:
        S = mag_b

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    m = ax.pcolormesh(t, freqs_b, S, shading="auto")
    ax.set_ylabel("Частота (Гц)")
    ax.set_xlabel("Время (с)")
    ax.set_title(f"STFT спектрограмма ({int(fmin_hz)}–{int(fmax_hz)} Гц) - {file_path}")
    ax.grid(False)
    plt.colorbar(m, ax=ax, label=("dB" if db else "Амплитуда"))

    if show:
        backend = matplotlib.get_backend().lower()
        if "agg" not in backend:
            plt.show()

    return t, freqs_b, S


def save_stft(
    t: np.ndarray,
    freqs_hz: np.ndarray,
    S: np.ndarray,
    *,
    prefix: str = "stft",
):
    """Saves STFT arrays for further post-processing."""
    np.save(f"{prefix}_t.npy", t)
    np.save(f"{prefix}_freqs_hz.npy", freqs_hz)
    np.save(f"{prefix}_S.npy", S)
    np.savetxt(f"{prefix}_amp.csv", S, delimiter=",")


def save_time_frequency_coords(
    t: np.ndarray,
    freqs_hz: np.ndarray,
    S: np.ndarray,
    *,
    path: str,
):
    """Save 2 columns: time (s) and dominant STFT frequency (Hz)."""
    if S.ndim != 2:
        raise ValueError("S must be 2D (freq, time).")
    ridge_idx = np.argmax(S, axis=0)
    ridge_freqs = freqs_hz[ridge_idx]
    data = np.column_stack([t, ridge_freqs])
    np.savetxt(path, data, fmt="%.10g", delimiter="\t", header="t_s\tfreq_hz")


def save_time_frequency_coords_csv(
    t: np.ndarray,
    freqs_hz: np.ndarray,
    S: np.ndarray,
    *,
    path: str,
):
    """Same as save_time_frequency_coords, but CSV format."""
    if S.ndim != 2:
        raise ValueError("S must be 2D (freq, time).")
    ridge_idx = np.argmax(S, axis=0)
    ridge_freqs = freqs_hz[ridge_idx]
    data = np.column_stack([t, ridge_freqs])
    np.savetxt(path, data, fmt="%.10g", delimiter=",", header="t_s,freq_hz")


def save_stft_figure(fig, wav_path: str, project_dir: str):
    stem = Path(wav_path).stem
    out_path = Path(project_dir) / f"{stem}_fourier_spectrogram.png"
    fig.savefig(out_path, dpi=150)