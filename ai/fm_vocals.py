"""Separate a vocal stem from mono FM using the Spleeter ONNX export."""
import math
from pathlib import Path

import numpy as np
from scipy import fft, signal

REPO = 'https://huggingface.co/csukuangfj/sherpa-onnx-spleeter-2stems/resolve/7001ba316a615cacddb3f9ef3ec416661a277e26/'
MODELS = {
    'vocals': ('vocals.onnx', 'bdc16ab6bf6117ddd4842c19e80e40e2be188fc555295064d424616b0224ac97'),
    'accompaniment': ('accompaniment.onnx', '671ace17acd3720674a2bc14de32ac6292453dec20d9eb0ba4255d4ad8e3d8c0'),
}
NFFT, HOP, FRAMES, BINS = 4096, 1024, 512, 1024
RATE = 16000


class VocalSeparator:
    def __init__(self, folder, digest):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        opts.enable_cpu_mem_arena = False
        # Дві моделі оцінюють голос та акомпанемент для спектральної маски.
        self.models = {}
        for key, (name, sha) in MODELS.items():
            p = Path(folder) / ('spleeter_' + name)
            if not p.is_file() or digest(p) != sha:
                raise ValueError('Не знайдено перевірену модель Spleeter. Виконай --download-vocals.')
            self.models[key] = ort.InferenceSession(str(p), sess_options=opts, providers=['CPUExecutionProvider'])

    def separate(self, audio):
        # Spleeter operates at 44.1 kHz with two channels; mono FM is duplicated.
        up = signal.resample_poly(audio, 441, 160).astype(np.float32, copy=False)
        win = signal.get_window('hann', NFFT).astype(np.float32)
        frames = math.ceil((len(up) + NFFT // 2) / HOP)
        padded_len = math.ceil(frames / FRAMES) * FRAMES
        x = np.pad(up, (NFFT // 2, (padded_len - 1) * HOP + NFFT - NFFT // 2 - len(up)))
        mixture = np.lib.stride_tricks.sliding_window_view(x, NFFT)[::HOP]
        y = np.zeros(len(x), np.float32)
        norm = np.zeros(len(x), np.float32)
        for start in range(0, frames, FRAMES):
            block = fft.rfft(mixture[start:start + FRAMES] * win, axis=-1)
            mag = np.abs(block[:, :BINS])
            tensor = np.broadcast_to(mag, (2, 1, FRAMES, BINS)).copy()
            vocal, backing = (session.run(['y'], {'x': tensor})[0][0, 0] ** 2
                              for session in self.models.values())
            block[:, :BINS] *= vocal / np.maximum(vocal + backing, 1e-10)
            block[:, BINS:] = 0
            rebuilt = fft.irfft(block, n=NFFT, axis=-1) * win
            for row, frame in enumerate(rebuilt):
                at = (start + row) * HOP
                y[at:at+NFFT] += frame
                norm[at:at+NFFT] += win * win
        stem = y[NFFT//2:NFFT//2+len(up)] / np.maximum(norm[NFFT//2:NFFT//2+len(up)], 1e-8)
        return signal.resample_poly(stem, 160, 441)[:len(audio)].astype(np.float32, copy=False)
