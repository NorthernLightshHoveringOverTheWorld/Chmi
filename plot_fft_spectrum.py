import numpy as np
import matplotlib.pyplot as plt
from audio_io import read_wav
import matplotlib


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