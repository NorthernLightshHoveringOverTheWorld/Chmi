import numpy as np
import matplotlib.pyplot as plt
from ssqueezepy import cwt, Wavelet
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


# ==================== 2. CWT ====================
def compute_cwt(signal_data):
    t = signal_data['t']
    x = signal_data['x']
    fs = signal_data['fs']

    wavelet = Wavelet(('gmw', {'beta': 20, 'gamma': 30}), N=len(t))
    Wx, scales = cwt(x, wavelet=wavelet, scales='log-piecewise', nv=32)

    frequencies = 0.1 * fs / scales
    amplitude = np.abs(Wx)

    peak_idx = np.argmax(amplitude, axis=0)
    inst_freq = frequencies[peak_idx]

    return {
        'Wx': Wx,
        'frequencies': frequencies,
        'amplitude': amplitude,
        'instantaneous_freq': inst_freq
    }


# ==================== 3. ВЫВОД КООРДИНАТ ====================
def print_coordinates(signal_data, cwt_data, max_points=20):
    t = signal_data['t']
    x = signal_data['x']
    inst_freq = cwt_data['instantaneous_freq']

    print("\nИСХОДНЫЙ СИГНАЛ (t, x):")
    for i in range(min(max_points, len(t))):
        print(f"{t[i]:.6e}, {x[i]:.6f}")

    print("\nПОСЛЕ CWT (t, f_inst):")
    for i in range(min(max_points, len(t))):
        print(f"{t[i]:.6e}, {inst_freq[i]:.6f}")


# ==================== 4. СОХРАНЕНИЕ ====================
def save_coordinates(signal_data, cwt_data):
    t = signal_data['t']
    x = signal_data['x']
    inst_freq = cwt_data['instantaneous_freq']

    # TXT
    np.savetxt("signal_coords.txt", np.column_stack((t, x)),
               header="t x", comments='')
    np.savetxt("cwt_coords.txt", np.column_stack((t, inst_freq)),
               header="t f_inst", comments='')

    # CSV
    np.savetxt("signal_coords.csv", np.column_stack((t, x)),
               delimiter=",", header="t,x", comments='')
    np.savetxt("cwt_coords.csv", np.column_stack((t, inst_freq)),
               delimiter=",", header="t,f_inst", comments='')

    print("\nФайлы сохранены:")
    print("signal_coords.txt / .csv")
    print("cwt_coords.txt / .csv")


# ==================== 5. ВИЗУАЛИЗАЦИЯ ====================
def plot(signal_data, cwt_data):
    t = signal_data['t']
    x = signal_data['x']
    inst_freq = cwt_data['instantaneous_freq']

    plt.figure(figsize=(10, 5))

    plt.subplot(2, 1, 1)
    plt.plot(t, x)
    plt.title("Исходный сигнал")

    plt.subplot(2, 1, 2)
    plt.plot(t, inst_freq)
    plt.title("Мгновенная частота (после CWT)")

    plt.tight_layout()
    plt.show()


# ==================== MAIN ====================
def main():
    signal_data = generate_ultrasound_signal(defect_position=12e-6)
    cwt_data = compute_cwt(signal_data)

    print_coordinates(signal_data, cwt_data)
    save_coordinates(signal_data, cwt_data)
    plot(signal_data, cwt_data)

    print("\n✅ Готово!")


if __name__ == "__main__":
    main()