import numpy as np
import wave
import struct

#def txt_to_wav(txt_filename, wav_filename, sample_rate=100000):
#    amplitudes = []
#
#    print("Чтение файла...")
#    with open(txt_filename, 'r') as f:
#        # Пропускаем заголовок, если он есть
#        header = f.readline()
#
#        for line in f:
#            try:
#                # Берем второе значение (амплитуду) после запятой
#                _, amp = line.strip().split(',')
#                amplitudes.append(float(amp))
#            except ValueError:
#                continue
#
#    # Преобразование в массив numpy
#    audio_data = np.array(amplitudes)
#
#    # Нормализация и перевод в 16-битный формат (PCM_16)
#    # Это стандарт для WAV, значения должны быть от -32768 до 32767
#    audio_data = (audio_data * 32767).astype(np.int16)
#
#    print(f"Запись в {wav_filename}...")
#    with wave.open(wav_filename, 'w') as wav_file:
#        n_channels = 1  # Моно
#        sampwidth = 2  # 2 байта (16 бит)
#
#        wav_file.setparams((n_channels, sampwidth, sample_rate, len(audio_data), 'NONE', 'not compressed'))
#
#        # Записываем упакованные бинарные данные
#        for value in audio_data:
#            wav_file.writeframes(struct.pack('h', value))
#
#    print("Готово!")

def main():
    # Параметры
    frequency = 5000  # Частота 10 кГц
    duration = 3  # Длительность 5 секунд
    sampling_rate = 50000  # Частота дискретизации (100 кГц)

    # Генерация временной оси (от 0 до duration с шагом 1/sampling_rate)
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

    # Генерация значений синусоиды: y = sin(2 * pi * f * t)
    y = np.sin(2 * np.pi * frequency * t)

    # Запись в файл
    with open("sinusoid_data.txt", "w") as file:
        file.write("Time(s), Amplitude\n")  # Заголовок
        # Объединяем t и y для быстрой записи
        for time_val, amp_val in zip(t, y):
            file.write(f"{time_val:.7f}, {amp_val:.7f}\n")

    print(f"Готово! Файл 'sinusoid_data.txt' создан. Записано {len(t)} точек.")
    #txt_to_wav("sinusoid_data.txt", "sounds/output_signal.wav", sample_rate=100000)

if __name__ == '__main__':
    main()