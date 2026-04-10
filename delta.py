import numpy as np


def main():
    # 192 kHz, длительность записи; импульс строго на t = 0.5 с (слышен в этот момент).
    fs = 192_000
    duration_s = 5.0
    n_samples = int(round(duration_s * fs))
    t = np.arange(n_samples, dtype=np.float64) / fs
    x = np.zeros_like(t)
    t_impulse = 0.5
    idx = int(round(t_impulse * fs))
    if not (0 <= idx < n_samples):
        raise ValueError(f"Импульс t={t_impulse} с вне диапазона [0, {duration_s}) с")
    x[idx] = 1.0

    with open("delta_data.txt", "w") as file:
        file.write("Time(s), Amplitude\n")
        for time_val, x_val in zip(t, x):
            # Keep enough precision so sample-rate estimation is stable.
            file.write(f"{time_val:.12f}, {x_val:.7f}\n")

    duration = n_samples / fs
    print(f"Готово! Файл 'delta_data.txt' создан. Записано {len(t)} точек.")
    print(f"fs={fs} Гц, длительность={duration:.3f} c")
    print(f"Дельта: отсчёт {idx}, время t={t[idx]:.9f} с (цель {t_impulse} с)")


if __name__ == '__main__':
    main()

