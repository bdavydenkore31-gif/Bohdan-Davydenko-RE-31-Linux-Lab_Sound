"""Offline speech enhancement using the published DTLN ONNX model (mono, 16 kHz)."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from urllib.request import urlopen
import wave

import numpy as np
from scipy import fft

ROOT = Path(__file__).resolve().parent
SOURCE = 'https://raw.githubusercontent.com/breizhn/DTLN/1de1f15a8b5b7e1c44905618ff2ef70ca8277fbc/pretrained_model/'
MODELS = {'model_1.onnx': '22b91cae3855e5a0620e66a917ca6c82c58db0e842c770f58d86751c5e8d4ae3',
          'model_2.onnx': 'e20c92f9233fccf29cddf86970d0d0161a03aebccc26d6f4d5639c4d5ec2e639'}
BLOCK, STEP = 512, 128


def download(folder):
    folder.mkdir(parents=True, exist_ok=True)
    for name, sha in MODELS.items():
        path = folder / name
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == sha:
            continue
        with urlopen(SOURCE + name, timeout=90) as response:
            data = response.read(4_000_000)
        if hashlib.sha256(data).hexdigest() != sha:
            raise ValueError('Контрольна сума моделі DTLN не збігається.')
        pending = path.with_suffix('.tmp')
        pending.write_bytes(data)
        pending.replace(path)


class Stage:
    def __init__(self, path, sha, options):
        import onnxruntime
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError('Немає перевіреної моделі DTLN. Запусти з --download-model.')
        self.session = onnxruntime.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        inputs = self.session.get_inputs()
        self.samples, self.memory = inputs[0].name, inputs[1].name
        self.state = np.zeros([v if isinstance(v, int) else 1 for v in inputs[1].shape], np.float32)

    def __call__(self, frame):
        samples, self.state = self.session.run(None, {self.samples: frame.reshape(1, 1, -1), self.memory: self.state})
        return samples.ravel()


def enhance(samples, folder):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    xf, xt = (Stage(folder / k, sha, opts) for k, sha in MODELS.items())
    latency = BLOCK - STEP
    frames = math.ceil((len(samples) + latency) / STEP)
    padded = np.pad(samples, (0, frames * STEP - len(samples)))
    queue = np.zeros(BLOCK, np.float32)
    reconstructed = np.zeros(frames * STEP + latency, np.float32)
    for i in range(0, len(padded), STEP):
        queue[:-STEP] = queue[STEP:]
        queue[-STEP:] = padded[i:i+STEP]
        spectrum = fft.rfft(queue)
        weights = xf(np.abs(spectrum))
        time_frame = fft.irfft(spectrum * weights, BLOCK)
        reconstructed[i:i+BLOCK] += xt(time_frame)
    # The overlap-add stream initially represents the zero prefix, not time zero.
    return reconstructed[latency:latency+len(samples)]


def main():
    parser = argparse.ArgumentParser(description='Нейромережеве шумозаглушення мовлення DTLN')
    parser.add_argument('input', type=Path, nargs='?')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--report', type=Path, help='Позначити, що нейродоріжка готова в цьому FM-звіті.')
    parser.add_argument('--models', type=Path, default=ROOT / 'models' / 'dtln')
    parser.add_argument('--download-model', action='store_true')
    args = parser.parse_args()
    if args.download_model:
        download(args.models)
    if args.input is None:
        if args.download_model:
            return
        parser.error('Укажи вхідний WAV або --download-model.')
    if args.out is None:
        parser.error('Укажи вихідний WAV через --out.')
    with wave.open(str(args.input), 'rb') as f:
        if (f.getnchannels(), f.getsampwidth(), f.getframerate()) != (1, 2, 16000):
            raise ValueError('DTLN потребує моно WAV: 16 біт / 16 кГц.')
        a = np.frombuffer(f.readframes(f.getnframes()), '<i2').astype(np.float32) / 32768
    report = json.loads(args.report.read_bytes()) if args.report else None
    if report is not None and abs(report['duration_s'] - len(a)/16000) > .001:
        raise ValueError('Тривалість WAV не відповідає FM-звіту.')
    b = enhance(a, args.models)
    if not np.isfinite(b).all():
        raise ValueError('Нейромережа повернула некоректні відліки.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out.with_suffix('.tmp')
    with wave.open(str(temp), 'wb') as f:
        f.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        f.writeframes(np.rint(np.clip(b, -1, .999969) * 32768).astype('<i2').tobytes())
    temp.replace(args.out)
    if report is not None:
        report['neural_audio'] = {'available': True, 'method': 'DTLN', 'purpose': 'speech_enhancement'}
        pending = args.report.with_suffix('.dtln.tmp')
        pending.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        pending.replace(args.report)
    print(f'Доріжка DTLN: {args.out} ({len(b)/16000:g} с)', flush=True)


if __name__ == '__main__':
    main()
