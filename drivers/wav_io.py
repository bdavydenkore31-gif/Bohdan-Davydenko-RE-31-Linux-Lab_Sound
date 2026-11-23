"""Читання заголовка I/Q WAV і перетворення PCM-відліків."""
import wave
import numpy as np

def pcm_block(raw, width, channels):
    # У 8-бітному PCM нуль сигналу кодується числом 128.
    if width == 1:
        data = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128
    elif width in (2, 4):
        data = (np.frombuffer(raw, '<i2' if width == 2 else '<i4').astype(np.float32)
                / (2 ** (8 * width - 1)))
    else:
        raise ValueError('Підтримується PCM WAV: 8, 16 або 32 біти.')
    return data.reshape(-1, channels).astype(np.float32, copy=False)


def capture_info(source):
    # Для FM-демодуляції очікуємо два компоненти високочастотного I/Q.
    with wave.open(str(source), 'rb') as src:
        if src.getnchannels() != 2 or src.getsampwidth() not in (1, 2, 4) or not 240000 <= src.getframerate() <= 5_000_000:
            raise ValueError('Для налаштування потрібен двоканальний PCM I/Q WAV.')
        return src.getframerate(), src.getnframes()
