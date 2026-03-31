import os
import matplotlib.pyplot as plt
from pathlib import Path

from plot_fft_spectrum import plot_fft_spectrum
from plot_wavelet_spectrum import plot_cwt_spectrogram, print_cwt_preview, save_cwt

if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    files = [
        "sounds/10KHz.wav",
    ]

    for file in files:
        try:
            fig, (ax_fft, ax_cwt) = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)

            freqs, fft_data = plot_fft_spectrum(file, ax=ax_fft, show=False)
            t, wfreqs, amp, Wx = plot_cwt_spectrogram(
                file, max_seconds=1.0, nv=8, downsample=4, ax=ax_cwt, show=False
            )

            print_cwt_preview(t, wfreqs, Wx)
            save_cwt(t, wfreqs, Wx, prefix="sounds_10KHz")

            out_path = project_dir / f"{Path(file).stem}_fft_cwt.png"
            fig.savefig(out_path, dpi=150)
            print(f"Сохранил картинку: {out_path.resolve()} (exists={out_path.exists()})")

            if os.environ.get("SHOW_PLOTS") == "1":
                plt.show()
            print(f"Файл {file} успешно обработан")
        except FileNotFoundError:
            print(f"Файл {file} не найден!")
        except Exception as e:
            print(f"Ошибка при обработке файла {file}: {e}")