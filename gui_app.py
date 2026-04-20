import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Must be set before importing pyplot/creating figures for Tk embedding.
matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from plot_fft_spectrum import plot_stft_spectrogram  # noqa: E402
from plot_wavelet_spectrum import (  # noqa: E402
    compute_cwt,
    plot_cwt_scalogram,
    plot_longitudinal_slice_morlet_re,
    save_cwt,
    save_longitudinal_slice_txt,
    save_time_frequency_coords,
    save_time_frequency_coords_csv,
)
from txt_to_wav import txt_to_wav  # noqa: E402


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Chmi: FFT / Morlet CWT")
        self.geometry("1100x800")

        self.project_dir = Path(__file__).resolve().parent
        default_wav = self.project_dir / "sounds" / "output_signal.wav"
        self.loaded_path: str | None = str(default_wav) if default_wav.exists() and default_wav.stat().st_size > 0 else None

        top = tk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.lbl_file = tk.Label(top, text=f"Файл: {self.loaded_path}" if self.loaded_path else "Файл: (не выбран)", anchor="w")
        self.lbl_file.pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Label(top, text="Morlet f (Гц):").pack(side=tk.RIGHT, padx=(10, 4))
        self.ent_target_freq = tk.Entry(top, width=10)
        self.ent_target_freq.insert(0, "10000")
        self.ent_target_freq.pack(side=tk.RIGHT)

        btn_load = tk.Button(top, text="Загрузить WAV…", command=self.on_load)
        btn_load.pack(side=tk.RIGHT, padx=(6, 0))

        btn_txt_to_wav = tk.Button(top, text="TXT -> WAV", command=self.on_txt_to_wav)
        btn_txt_to_wav.pack(side=tk.RIGHT, padx=(6, 0))

        btn_fft = tk.Button(top, text="Преобразовать по Фурье", command=self.on_fft)
        btn_fft.pack(side=tk.RIGHT, padx=(6, 0))

        btn_morlet = tk.Button(top, text="Преобразовать (вейвлет Морле)", command=self.on_morlet)
        btn_morlet.pack(side=tk.RIGHT, padx=(6, 0))

        self.fig = Figure(figsize=(10, 7), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.canvas.mpl_connect("button_press_event", self._on_spectrogram_click)

        self.current_spec: dict | None = None

        self._draw_placeholder()

    def _draw_placeholder(self):
        self.ax.clear()
        self.ax.set_title("Загрузите WAV и нажмите кнопку преобразования")
        self.ax.set_axis_off()
        self.current_spec = None
        self.canvas.draw_idle()

    def _on_spectrogram_click(self, event):
        if event.inaxes != self.ax or self.current_spec is None:
            return
        if event.button not in (1, 3):
            return

        t = self.current_spec["t"]
        freqs_hz = self.current_spec["freqs_hz"]
        spec = self.current_spec["spec"]
        title = self.current_spec["title"]
        y_label = self.current_spec["y_label"]
        Wx = self.current_spec.get("Wx")

        if event.button == 1:  # левый: боковой срез (спектр во время t)
            if event.xdata is None:
                return
            ti = int(np.argmin(np.abs(t - event.xdata)))
            t_sel = float(t[ti])
            slice_vals = spec[:, ti]
            order = np.argsort(freqs_hz)
            freqs_plot = freqs_hz[order]
            slice_plot = slice_vals[order]

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(freqs_plot, slice_plot, color="#1f77b4", linewidth=1.5)
            ax.set_title(f"Боковой срез: t={t_sel:.6f} c\n{title}")
            ax.set_xlabel("Частота (Гц)")
            ax.set_ylabel(y_label)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(float(freqs_plot[0]), float(freqs_plot[-1]))
            if title == "Morlet CWT":
                ax.set_xscale("log")
        else:  # правый: продольный срез Re(Wx) по времени (как test2 / форма Морле для дельты)
            if event.ydata is None:
                return
            fi = int(np.argmin(np.abs(freqs_hz - event.ydata)))
            f_sel = float(freqs_hz[fi])

            fig, ax = plt.subplots(figsize=(8, 4))
            if Wx is not None and title == "Morlet CWT":
                plot_longitudinal_slice_morlet_re(
                    t,
                    Wx[fi, :],
                    f_hz=f_sel,
                    ax=ax,
                    show=False,
                )
                ax.set_title(
                    f"Продольный срез: Re(Wx), f ≈ {f_sel:.1f} Гц\n({title})",
                    fontweight="bold",
                )
                if self.loaded_path:
                    stem = Path(self.loaded_path).stem
                    safe_f = f"{f_sel:.0f}".replace(".", "p")
                    out_txt = self.project_dir / f"{stem}_longitudinal_slice_f{safe_f}Hz.txt"
                    save_longitudinal_slice_txt(
                        t,
                        Wx[fi, :],
                        path=str(out_txt),
                        f_hz=f_sel,
                    )
            else:
                slice_vals = spec[fi, :]
                ax.plot(t, slice_vals, color="#1f77b4", linewidth=1.5)
                ax.set_title(f"Продольный срез: f={f_sel:.2f} Гц\n{title}")
                ax.set_xlabel("Время (с)")
                ax.set_ylabel(y_label)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(float(t[0]), float(t[-1]))

        fig.tight_layout()
        fig.show()

    def _require_file(self) -> str:
        if not self.loaded_path:
            raise RuntimeError("Сначала загрузите WAV-файл.")
        return self.loaded_path

    def _save_current_figure(self, suffix: str):
        if not self.loaded_path:
            return
        stem = Path(self.loaded_path).stem
        out_path = self.project_dir / f"{stem}_{suffix}.png"
        self.fig.savefig(out_path, dpi=150)

    def on_load(self):
        path = filedialog.askopenfilename(
            title="Выберите WAV файл",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
            initialdir=str(self.project_dir),
        )
        if not path:
            return
        self.loaded_path = path
        self.lbl_file.config(text=f"Файл: {path}")
        self._draw_placeholder()

    def on_txt_to_wav(self):
        try:
            input_txt = filedialog.askopenfilename(
                title="Выберите TXT/CSV файл",
                filetypes=[("Text/CSV files", "*.txt *.csv"), ("All files", "*.*")],
                initialdir=str(self.project_dir),
            )
            if not input_txt:
                return

            default_out = str(Path(input_txt).with_suffix(".wav"))
            output_wav = filedialog.asksaveasfilename(
                title="Сохранить WAV как",
                defaultextension=".wav",
                initialfile=Path(default_out).name,
                initialdir=str(Path(default_out).parent),
                filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
            )
            if not output_wav:
                return

            sample_rate = simpledialog.askinteger(
                "TXT -> WAV",
                "Sample rate (Гц).\nДля 2-колоночного TXT можно оставить пустым (Cancel).",
                minvalue=1,
                parent=self,
            )
            amp_col = simpledialog.askinteger(
                "TXT -> WAV",
                "Индекс колонки амплитуды (по умолчанию последняя = -1):",
                initialvalue=-1,
                parent=self,
            )
            if amp_col is None:
                amp_col = -1
            normalize = messagebox.askyesno("TXT -> WAV", "Нормировать сигнал в диапазон [-1, 1]?")

            fs, n = txt_to_wav(
                input_txt,
                output_wav,
                sample_rate=sample_rate,
                amplitude_col=amp_col,
                normalize=normalize,
            )

            self.loaded_path = output_wav
            self.lbl_file.config(text=f"Файл: {output_wav}")
            self._draw_placeholder()
            messagebox.showinfo("TXT -> WAV", f"Готово:\n{output_wav}\nfs={fs} Гц\nsamples={n}")
        except Exception as e:
            messagebox.showerror("Ошибка TXT -> WAV", str(e))

    def on_fft(self):
        try:
            file_path = self._require_file()
            self.ax.clear()
            t, freqs_hz, S = plot_stft_spectrogram(
                file_path,
                ax=self.ax,
                show=False,
                nperseg=2048,
                noverlap=1536,
                fmin_hz=0.0,
                fmax_hz=100000.0,
                db=True,
            )
            self.current_spec = {
                "t": t,
                "freqs_hz": freqs_hz,
                "spec": S,
                "title": "STFT",
                "y_label": "Уровень (dB)",
            }
            self.fig.tight_layout()
            self.canvas.draw_idle()
            self._save_current_figure("stft")
        except Exception as e:
            messagebox.showerror("Ошибка FFT", str(e))

    def on_morlet(self):
        try:
            file_path = self._require_file()
            self.ax.clear()
            _fs, t, freqs_hz, Wx = compute_cwt(
                file_path,
                fmin_hz=0.1,
                fmax_hz=100000.0,
                max_seconds=None,
                nv=12,
                downsample=1,
                target_fs=192000,
            )
            amp = np.abs(Wx)

            target_hz = float(self.ent_target_freq.get().strip().replace(",", "."))
            plot_cwt_scalogram(t=t, freqs_hz=freqs_hz, Wx=Wx, ax=self.ax, show=False, log_freq=True)
            self.ax.axhline(target_hz, color="white", linewidth=1.0, alpha=0.8)
            self.current_spec = {
                "t": t,
                "freqs_hz": freqs_hz,
                "spec": amp,
                "Wx": Wx,
                "title": "Morlet CWT",
                "y_label": "Амплитуда",
            }
            self.fig.tight_layout()
            self.canvas.draw_idle()

            stem = Path(file_path).stem
            save_cwt(t, freqs_hz, Wx, prefix=str(self.project_dir / stem))
            coords_path = self.project_dir / f"{stem}_tf_coords.txt"
            save_time_frequency_coords(t, freqs_hz, Wx, path=str(coords_path))
            coords_csv_path = self.project_dir / f"{stem}_tf_coords.csv"
            save_time_frequency_coords_csv(t, freqs_hz, Wx, path=str(coords_csv_path))
            self._save_current_figure("morlet_scalogram")
        except Exception as e:
            messagebox.showerror("Ошибка Morlet CWT", str(e))


def main():
    # Ensure relative paths work similarly to other scripts.
    os.chdir(Path(__file__).resolve().parent)
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()

