import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# ==================== 1. ГЕНЕРАЦИЯ СИГНАЛА ====================
def generate_ultrasound_signal(fs=100e6, duration=20e-6, f_center=5e6, defect_position=None):
    t = np.linspace(0, duration, int(fs * duration))
    envelope_main = np.exp(-((t - duration / 2) ** 2) / (2 * (0.5e-6) ** 2))
    carrier = np.sin(2 * np.pi * f_center * t)
    x = envelope_main * carrier * np.exp(-t * 0.5e6)

    if defect_position is not None:
        envelope_defect = 0.3 * np.exp(-((t - defect_position) ** 2) / (2 * (0.3e-6) ** 2))
        x += envelope_defect * np.sin(2 * np.pi * (f_center - 0.2e6) * (t - defect_position))

    envelope = np.abs(hilbert(x))
    return {'t': t, 'x': x, 'envelope': envelope, 'fs': fs}

# ==================== 2. АНАЛИЗ ФУРЬЕ (FFT) ====================
def compute_fft(signal_data):
    # всё, что было получено при генерации сигнала
    x = signal_data['x']  # амплитуды сигнала
    fs = signal_data['fs']  # частота дискретизации (как часто снимались показания)
    t = signal_data['t']  # массив моментов времени
    n = len(x)  # общее количество точек в сигнале

    fft_values = np.fft.fft(x) # массив комплексных чисел с информацией об амплитуде и фазе каждой частоты
    freqs = np.fft.fftfreq(n, 1 / fs) # значения координаты х

    mask = freqs >= 0 # отсекает всё, что меньше 0
    pos_freqs = freqs[mask]
    pos_amplitude = np.abs(fft_values[mask]) / n # находит нормализованную амплитуду

    # находим одну пиковую частоту для всего сигнала
    peak_freq = pos_freqs[np.argmax(pos_amplitude)]

    # создаем массив "мгновенной" (она будет константой) частоты той же длины что и t
    # это демонстрирует, что метод Фурье не видит изменений во времени
    inst_freq_array = np.full_like(t, peak_freq)

    return {
        'freqs': pos_freqs,
        'amplitude': pos_amplitude,
        'instantaneous_freq': inst_freq_array
    }

# ==================== 3. СОХРАНЕНИЕ ====================
def save_coordinates(signal_data, fft_data):
    t = signal_data['t']
    x = signal_data['x']
    inst_freq = fft_data['instantaneous_freq']

    np.savetxt("signal_coords.csv", np.column_stack((t, x)),
               delimiter=",", header="t,x", comments='')
    np.savetxt("fft_coords.csv", np.column_stack((t, inst_freq)),
               delimiter=",", header="t,f_inst", comments='')

    print(f"\nФайлы сохранены. Пиковая частота Фурье: {inst_freq[0]/1e6:.2f} МГц")

# ==================== 4. ВИЗУАЛИЗАЦИЯ ====================
def plot_comparison(signal_data, fft_data):
    t = signal_data['t'] * 1e5
    x = signal_data['x']

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(t, x, color='blue')
    plt.title("Сигнал во времени")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    f_mask = fft_data['freqs'] <= 15e6
    plt.plot(fft_data['freqs'][f_mask] / 1e6, fft_data['amplitude'][f_mask], 'g')
    plt.fill_between(fft_data['freqs'][f_mask] / 1e6, fft_data['amplitude'][f_mask], color='g', alpha=0.2)
    plt.title("Спектр Фурье (Глобальный)")
    plt.xlabel("Частота (МГц)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def main():
    signal_data = generate_ultrasound_signal(defect_position=12e-6)
    fft_data = compute_fft(signal_data)
    save_coordinates(signal_data, fft_data)
    plot_comparison(signal_data, fft_data)

if __name__ == '__main__':
    main()