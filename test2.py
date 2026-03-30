import numpy as np
import matplotlib.pyplot as plt
from ssqueezepy import cwt, Wavelet
from scipy.signal import hilbert


# ==================== 1. ГЕНЕРАЦИЯ СИГНАЛА ====================
def generate_ultrasound_signal(fs=100e6, duration=20e-6):
    t = np.linspace(0, duration, int(fs * duration))
    x = np.zeros_like(t)
    x[len(t) // 2] = 1.0
    return {'t': t, 'x': x, 'fs': fs}


# ==================== 2. CWT ====================
def compute_cwt(signal_data):
    t = signal_data['t']
    x = signal_data['x']
    fs = signal_data['fs']

    wavelet = Wavelet(('morlet', {'mu': 2*3.14}), N=len(t))

    Wx, scales = cwt(x, wavelet=wavelet, scales='log-piecewise', nv=32)

    frequencies = 0.1 * fs / scales
    return {'Wx': Wx, 'frequencies': frequencies}


# ==================== 5. ВИЗУАЛИЗАЦИЯ ====================
def plot(signal_data, cwt_data):
    t = signal_data['t']
    x = signal_data['x']
    Wx = cwt_data['Wx']

    # Берем центральную частоту для отображения формы
    mid_idx = Wx.shape[0] // 2

    # Масштабируем время в микросекунды для удобства сетки (как на фото)
    t_mks = (t - t[len(t) // 2]) * 1e6

    plt.figure(figsize=(10, 6))

    # Визуализируем только результат CWT, так как он должен соответствовать картинке
    # Используем нормализацию, чтобы пик был равен 1.0
    y_vals = np.real(Wx[mid_idx, :])
    y_vals /= np.max(np.abs(y_vals))

    plt.subplot(2, 1, 1)
    plt.plot(t, x, color='black', label='Delta function')
    plt.title("Входной сигнал: Дельта-функция")
    plt.xlabel("Время [сек]")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(t_mks, y_vals, color='#0072BD', linewidth=1.5, label='Morlet Wavelet (Re)')

    # Оформление в стиле предоставленного фото
    plt.title("Morlet Wavelet", fontweight='bold')
    plt.xlim(-4, 4)  # Ограничиваем окно, чтобы форма была крупной
    plt.ylim(-1, 1)
    plt.grid(True, which='both', linestyle='-', linewidth=0.5)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.axvline(0, color='black', linewidth=0.8)

    plt.tight_layout()
    plt.show()


def main():
    signal_data = generate_ultrasound_signal()
    cwt_data = compute_cwt(signal_data)
    plot(signal_data, cwt_data)
    print("\n✅ График адаптирован под образец!")


if __name__ == '__main__':
    main()