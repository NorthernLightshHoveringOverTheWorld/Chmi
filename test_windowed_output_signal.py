from pathlib import Path

import matplotlib.pyplot as plt

from plot_fft_spectrum import plot_stft_spectrogram
from plot_wavelet_spectrum import compute_cwt, plot_cwt_scalogram, save_cwt, save_time_frequency_coords, save_time_frequency_coords_csv


def main():
    project_dir = Path(__file__).resolve().parent
    wav_path_str = "C:/Users/0/vsu/Chmi/sounds/output_signal.wav"
    wav_path = Path(wav_path_str)
    if not wav_path.exists() or wav_path.stat().st_size == 0:
        raise SystemExit(f"Файл пустой или не найден: {wav_path}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)

    plot_stft_spectrogram(str(wav_path), ax=ax1, show=False)

    _fs, t, freqs_hz, Wx = compute_cwt(str(wav_path), max_seconds=None, nv=8, downsample=4)
    plot_cwt_scalogram(t=t, freqs_hz=freqs_hz, Wx=Wx, ax=ax2, show=False)

    out_png = project_dir / "output_signal_windowed.png"
    fig.savefig(out_png, dpi=150)

    prefix = str(project_dir / "output_signal")
    save_cwt(t, freqs_hz, Wx, prefix=prefix)
    save_time_frequency_coords(t, freqs_hz, Wx, path=str(project_dir / "output_signal_tf_coords.txt"))
    save_time_frequency_coords_csv(t, freqs_hz, Wx, path=str(project_dir / "output_signal_tf_coords.csv"))

    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()

