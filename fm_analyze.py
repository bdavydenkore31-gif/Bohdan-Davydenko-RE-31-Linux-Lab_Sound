"""Convert a local FM I/Q WAV to mono audio and classify the audio in time."""
import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import urllib.request
import wave
from pathlib import Path

import numpy as np
from scipy import fft, signal, special

ROOT = Path(__file__).resolve().parent
RATE = 16000
MODEL_URL = ('https://storage.googleapis.com/mediapipe-models/audio_classifier/'
             'yamnet/float32/1/yamnet.tflite')
MODEL_PATH = ROOT / 'models' / 'yamnet.tflite'
MODEL_SHA256 = '4d8b4a53282dc83ef04e3e7dbc4fbc98082e34e44ed798e16c3a0cdd4c584faf'
VOICE_URL = ('https://huggingface.co/syntropicsignal-ai/gender-voice-classifier/'
             'resolve/412d58a/gender_classifier_200k.onnx')
VOICE_PATH = ROOT / 'models' / 'voice_classifier.onnx'
VOICE_SHA256 = 'e98f8bc6d7960a8a2169368fe4533636903e712790e96dbff81b679ede5de252'
WINDOW = 15600
HOP = 8000
MAX_SECONDS = 600


def pcm_block(raw, width, channels):
    if width == 1:
        data = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128
    elif width in (2, 4):
        data = (np.frombuffer(raw, '<i2' if width == 2 else '<i4').astype(np.float32)
                / (2 ** (8 * width - 1)))
    else:
        raise ValueError('Підтримується PCM WAV: 8, 16 або 32 біти.')
    return data.reshape(-1, channels).astype(np.float32, copy=False)


def read_input(path, kind, center, station, swap_iq, deemphasis, limit, *, fast_iq=False):
    with wave.open(str(path), 'rb') as source:
        channels, width, rate, total, *_ = source.getparams()
        if not 8000 <= rate <= 5000000 or channels not in (1, 2) or width not in (1, 2, 4):
            raise ValueError('Непідтримувані параметри PCM WAV: потрібні 1–2 канали та 8/16/32 біти.')
        is_iq = kind == 'iq'
        if is_iq and (channels != 2 or rate < 240000):
            raise ValueError('Для I/Q потрібен двоканальний WAV із частотою від 240 тис. відліків/с. Для звуку: --kind audio.')
        if not is_iq and rate > 192000:
            raise ValueError('Частота зависока для звукового WAV. Можливо, це I/Q; спробуй --kind iq.')
        if is_iq:
            if center is None:
                match = re.search(r'(?:^|[_-])(\d{7,10})\s*Hz(?:[_\-.]|$)', path.name, re.I)
                if match is None:
                    raise ValueError('Укажи центральну частоту запису через --center-mhz.')
                center = int(match[1]) / 1e6
            station = center if station is None else station
            if not all(math.isfinite(x) and x > 0 for x in (center, station)):
                raise ValueError('Частоти мають бути скінченними додатними числами в МГц.')
            offset = (station - center) * 1e6
            if abs(offset) + 110000 > rate / 2:
                raise ValueError('Станція надто близько до краю смуги I/Q-запису.')
            dec = next(d for d in range(max(1, rate // 240000), 0, -1) if rate % d == 0)
            rf_rate = rate // dec
            fir = signal.firwin(64 * dec + 1, 100000, fs=rate).astype(np.float32)
            audio_filter = signal.butter(5, [45, 7200], btype='bandpass', fs=rf_rate, output='sos')
            a = math.exp(-1 / (rf_rate * deemphasis * 1e-6))
        else:
            rf_rate = rate
            center = station = None
        n = min(total, round(limit * rate)) if limit is not None else total
        if n > MAX_SECONDS * rate:
            raise ValueError('Запис довший за 10 хвилин. Для першого фрагмента додай --seconds 600 або менше.')
        if n < rate // 2:
            raise ValueError('Потрібно щонайменше 0,5 с запису.')
        if is_iq and fast_iq and not swap_iq:
            from fm_channel import demodulate
            mono = demodulate(path,center,station,0,n/rate,deemphasis)
            return mono, {'filename': path.name, 'input_kind': kind, 'input_sample_rate_hz': rate,
                          'input_bits': width * 8, 'input_channels': channels, 'center_mhz': center,
                          'station_mhz': station, 'deemphasis_us': deemphasis,
                          'duration_s': len(mono) / RATE}
        gcd = math.gcd(RATE, rf_rate)
        # Align overlap to the final audio sample grid; discard filter transients.
        boundary = rate // math.gcd(rate, RATE)
        overlap = max(1, math.ceil(.045 * rate / boundary)) * boundary
        samples = []
        for start in range(0, n, rate):
            left, stop = max(0, start - overlap), min(n, start + rate)
            right = min(total, stop + overlap)
            source.setpos(left)
            raw = source.readframes(right - left)
            if len(raw) != (right - left) * width * channels:
                raise ValueError('Дані WAV закінчуються раніше, ніж зазначено в заголовку.')
            data = pcm_block(raw, width, channels)
            if is_iq:
                i, q = (1, 0) if swap_iq else (0, 1)
                iq = (data[:, i] + 1j * data[:, q]).astype(np.complex64)
                if offset:
                    phase = np.arange(left, right, dtype=np.float64) * (-2 * np.pi * offset / rate)
                    iq *= np.exp(1j * phase).astype(np.complex64)
                # Symmetric FIR; discard its group delay, then demodulate adjacent samples.
                rf = signal.upfirdn(fir, iq, down=dec)[32:32 + (len(iq) + dec - 1) // dec]
                phase_change = np.angle(rf[1:] * rf[:-1].conj())
                phase_change = np.pad(phase_change, (1, 0), mode='edge')
                demod = signal.lfilter([1 - a], [1, -a], phase_change)
                audio = signal.sosfilt(audio_filter, demod)
            else:
                audio = data.mean(axis=1)
            audio = signal.resample_poly(audio.astype(np.float32), RATE // gcd, rf_rate // gcd)
            skip = (start - left) * RATE // rate
            take = round(stop * RATE / rate) - round(start * RATE / rate)
            samples.append(audio[skip:skip + take])
        mono = np.concatenate(samples).astype(np.float32, copy=False)
        mono -= mono.mean()
        if is_iq:
            peak = float(np.quantile(np.abs(mono), .999))
            mono = np.clip(mono * min(25, .65 / max(peak, 1e-6)), -.99, .99)
        return mono, {'filename': path.name, 'input_kind': kind, 'input_sample_rate_hz': rate,
                      'input_bits': width * 8, 'input_channels': channels, 'center_mhz': center,
                      'station_mhz': station, 'deemphasis_us': deemphasis if is_iq else None,
                      'duration_s': len(mono) / RATE}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_model(path, url, sha256, max_bytes=20_000_000):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        temp = Path(tmp.name)
        try:
            with urllib.request.urlopen(url, timeout=90) as response:
                data = response.read(max_bytes + 1)
            if len(data) > max_bytes or hashlib.sha256(data).hexdigest() != sha256:
                raise ValueError('Контрольна сума завантаженої моделі не збігається.')
            tmp.write(data)
            tmp.close()
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)


class Yamnet:
    def __init__(self, path):
        if not path.is_file():
            raise ValueError('Немає моделі YAMNet. Спочатку виконай: python fm_analyze.py --download-model')
        if digest(path) != MODEL_SHA256:
            raise ValueError('Неочікувана версія YAMNet. Завантаж модель через --download-model.')
        from ai_edge_litert.interpreter import Interpreter
        self.engine = Interpreter(model_path=str(path), num_threads=2)
        self.engine.allocate_tensors()
        inp, = self.engine.get_input_details()
        out, = self.engine.get_output_details()
        if int(np.prod(inp['shape'])) != WINDOW or int(np.prod(out['shape'])) != 521:
            raise ValueError('Неочікуваний формат входу або виходу моделі YAMNet.')
        self.inp, self.out = inp, out

    def score(self, samples):
        self.engine.set_tensor(self.inp['index'], samples.reshape(self.inp['shape']))
        self.engine.invoke()
        scores = self.engine.get_tensor(self.out['index']).reshape(-1)
        # Keep singing separate from music; otherwise it disappears into the music label.
        return np.array([scores[0], scores[132], max(scores[k] for k in (24, 25, 26, 27, 28, 29, 30, 31))], dtype=np.float32)


def classify_scores(speech, music):
    if speech >= .30 and music >= .30:
        return 'mixed'
    if speech >= .30 and speech - music >= .12:
        return 'speech'
    if music >= .30 and music - speech >= .12:
        return 'music'
    return 'unknown'


def mel_filterbank():
    # Slaney mel scale and area normalization, as used in the model's MFCC recipe.
    step = 200 / 3
    min_log_mel = 1000 / step
    log_step = math.log(6.4) / 27
    max_mel = min_log_mel + math.log((RATE / 2) / 1000) / log_step
    mel = np.linspace(0, max_mel, 82)
    edges = np.where(mel >= min_log_mel, 1000 * np.exp(log_step * (mel - min_log_mel)), step * mel)
    bins = fft.rfftfreq(512, 1 / RATE)
    rise = (bins[None, :] - edges[:-2, None]) / (edges[1:-1] - edges[:-2])[:, None]
    fall = (edges[2:, None] - bins[None, :]) / (edges[2:] - edges[1:-1])[:, None]
    return (np.maximum(0, np.minimum(rise, fall)) * (2 / (edges[2:] - edges[:-2]))[:, None]).astype(np.float32)


class VoiceModel:
    def __init__(self, path):
        if not path.is_file() or digest(path) != VOICE_SHA256:
            raise ValueError('Немає потрібної моделі голосу. Виконай: python fm_analyze.py --download-model')
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        self.bank = mel_filterbank()
        self.hann = signal.get_window('hann', 512).astype(np.float32)

    def estimate(self, audio):
        if len(audio) < 1.5 * RATE:
            return 'unknown', None
        # Match the published 16 kHz, up-to-three-second MFCC preprocessing.
        clip = np.pad(audio[:3 * RATE], 256)
        frames = np.lib.stride_tricks.sliding_window_view(clip, 512)[::160]
        power = np.abs(fft.rfft(frames * self.hann, axis=-1)) ** 2
        mel = self.bank @ power.T
        log_mel = 10 * np.log10(np.maximum(mel, 1e-10))
        log_mel = np.maximum(log_mel, log_mel.max() - 80)
        mfcc = fft.dct(log_mel, type=2, axis=0, norm='ortho')[:40]
        mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-8)
        logit = self.session.run(['logits'], {'mfcc': mfcc[None].astype(np.float32)})[0][0, 0]
        score = float(special.expit(logit))
        if not math.isfinite(score):
            return 'unknown', None
        label = 'female_likely' if score >= .8 else 'male_likely' if score <= .2 else 'unknown'
        return label, round(score, 3)


def pitch_hz(x):
    """YIN-style cumulative mean normalized difference with fixed overlap."""
    lo, hi = math.floor(RATE / 360), math.ceil(RATE / 65)
    base = len(x) - hi
    if base < hi or float(np.sqrt(np.mean(x * x))) < .005:
        return None
    squares = np.r_[0, np.cumsum(x * x, dtype=np.float64)]
    corr = signal.correlate(x, x[:base], mode='valid', method='fft')
    lags = np.arange(hi + 1)
    diff = np.maximum(0, squares[base] + squares[base + lags] - squares[lags] - 2 * corr)
    cm = np.ones(hi + 1)
    cm[1:] = diff[1:] * lags[1:] / np.maximum(np.cumsum(diff[1:]), 1e-12)
    choices = np.flatnonzero((cm[lo:hi] < .19) & (cm[lo:hi] <= cm[lo - 1:hi - 1])
                             & (cm[lo:hi] < cm[lo + 1:hi + 1]))
    if len(choices) == 0:
        return None
    p = int(choices[0] + lo)
    a, b, c = cm[p - 1:p + 2]
    shift = .5 * (a - c) / max(1e-12, a - 2 * b + c)
    return RATE / (p + shift)


def voice_stats(samples):
    filtered = signal.sosfilt(signal.butter(3, [70, 1250], btype='bandpass', fs=RATE, output='sos'), samples)
    candidates = [p for start in range(0, len(filtered) - 800 + 1, 480)
                  if (p := pitch_hz(filtered[start:start + 800])) is not None]
    if len(candidates) < 10:
        return None, len(candidates)
    return round(float(np.median(candidates)), 1), len(candidates)


def window_scores(audio, model):
    windows = []
    half = WINDOW // 2
    for start in range(0, len(audio), HOP):
        center = (start + min(start + HOP, len(audio))) // 2
        left, right = center - half, center + half
        clip = np.pad(audio[max(0, left):min(right, len(audio))],
                      (max(0, -left), max(0, right - len(audio))))
        windows.append(model.score(clip) if np.sqrt(np.mean(clip * clip)) >= .0025 else np.zeros(3))
    windows = np.array(windows)
    # Smooth adjacent scores with a compact three-window average.
    padded = np.pad(windows, ((1, 1), (0, 0)), mode='edge')
    return (padded[:-2] + 2 * padded[1:-1] + padded[2:]) / 4


def classify(audio, model, voice_model, meta, stem=None):
    smoothed = window_scores(audio, model)
    evidence = None
    if stem is not None:
        # Use the stem to recover voice masked by music; limit gain so near-silence is not amplified arbitrarily.
        gain = min(12, .65 / max(float(np.quantile(np.abs(stem), .998)), 1e-6))
        evidence = window_scores(np.clip(stem * gain, -.99, .99), model)
    runs = []
    for index, (speech, music, singing) in enumerate(smoothed):
        kind = classify_scores(speech, max(music, singing))
        voice_kind = 'speech' if kind in ('speech', 'mixed') else None
        use_stem = False
        if kind in ('music', 'mixed') and singing >= .23 and singing >= speech:
            kind, voice_kind = 'music_vocals', 'singing'
            use_stem = stem is not None
        elif evidence is not None and kind in ('music', 'mixed'):
            stem_speech, _, stem_singing = evidence[index]
            i, j = index * HOP, min((index + 1) * HOP, len(audio))
            level = np.linalg.norm(stem[i:j]) / max(np.linalg.norm(audio[i:j]), 1e-6)
            if level >= .045:
                if kind == 'music' and stem_singing >= .18 and stem_singing > stem_speech * 1.15:
                    kind, voice_kind, use_stem = 'music_vocals', 'singing', True
                elif kind == 'music' and stem_speech >= .29 and stem_speech > stem_singing * 1.15:
                    # An isolated track can make singing sound speech-like; label it as voice.
                    kind, voice_kind, use_stem = 'music_voice', 'unclear', True
                elif kind == 'mixed' and stem_speech >= .22:
                    use_stem = True
        signature = (kind, voice_kind, use_stem)
        if runs and runs[-1][2] == signature:
            runs[-1] = (runs[-1][0], index + 1, signature)
        else:
            runs.append((index, index + 1, signature))
    segments = []
    for first, end, (kind, voice_kind, use_stem) in runs:
        # Balance chunks to avoid a tiny trailing fragment after a 3-second block.
        count = math.ceil((end - first) / 6) if voice_kind else 1
        cuts = [first + round(k * (end - first) / count) for k in range(count + 1)]
        for i, j in zip(cuts, cuts[1:]):
            start, stop = i * HOP, min(j * HOP, len(audio))
            voice, voice_score, pitch, voiced = None, None, None, 0
            if voice_kind:
                clip = (stem if use_stem else audio)[start:stop]
                voice, voice_score = 'unknown', None
                if kind == 'speech':
                    voice, voice_score = voice_model.estimate(clip)
                elif use_stem and len(clip) >= RATE:
                    flank = max(0, (int(1.75 * RATE) - len(clip)) // 2)
                    context = stem[max(0, start - flank):min(len(stem), stop + flank)]
                    span = max(int(len(context) * .8), int(1.5 * RATE))
                    a, p = voice_model.estimate(context[:span])
                    b, q = voice_model.estimate(context[-span:])
                    if a == b and a != 'unknown' and min(abs(p - .5), abs(q - .5)) >= .4:
                        voice, voice_score = a, round((p + q) / 2, 3)
                pitch, voiced = voice_stats(clip)
            speech, music, singing = smoothed[i:j].mean(axis=0)
            segments.append({'start_s': round(start / RATE, 4), 'end_s': round(stop / RATE, 4),
                             'content': kind, 'voice': voice, 'f0_hz': pitch,
                             'voice_kind': voice_kind, 'voice_source': 'separated' if use_stem else None,
                             'voice_female_score': voice_score,
                             'voiced_frames': voiced, 'speech_score': round(float(speech), 3),
                             'music_score': round(float(music), 3), 'singing_score': round(float(singing), 3)})
    # An isolated contradictory label may be a misread or a very short speaker change.
    # Keep the original model score for inspection, and leave the voice undecided.
    labels = [segment['voice'] for segment in segments]
    types = {'male_likely', 'female_likely'}
    for i in range(1, len(labels) - 1):
        if labels[i - 1] == labels[i + 1] and {labels[i - 1], labels[i]} == types:
            segments[i]['voice'] = 'unknown'
    totals = {key: round(sum(s['end_s'] - s['start_s'] for s in segments if s['content'] == key), 3)
              for key in ('speech', 'music', 'mixed', 'music_vocals', 'music_voice', 'unknown')}
    return {**meta, 'schema': 2, 'separated_vocals': stem is not None, 'audio_sample_rate_hz': RATE, 'window_s': WINDOW / RATE,
            'step_s': HOP / RATE, 'method': ('Spleeter + ' if stem is not None else '')
            + 'YAMNet + Syntropic Signal MFCC/Bi-LSTM; YIN-style F0 diagnostic',
            'totals_s': totals, 'segments': segments}


def write_wav(path, audio):
    with wave.open(str(path), 'wb') as output:
        output.setparams((1, 2, RATE, 0, 'NONE', 'not compressed'))
        output.writeframes(np.rint(np.clip(audio, -1, .999) * 32767).astype('<i2').tobytes())


def save_result(folder, audio, report, stem=None):
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=folder) as temp:
        temp = Path(temp)
        wav = temp / 'fm_audio.wav'
        write_wav(wav, audio)
        if stem is not None:
            stem_wav = temp / 'fm_vocals.wav'
            gain = min(12, .65 / max(float(np.quantile(np.abs(stem), .998)), 1e-6))
            write_wav(stem_wav, stem * gain)
            stem_wav.replace(folder / stem_wav.name)
        path = temp / 'fm_report.json'
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        wav.replace(folder / wav.name)
        path.replace(folder / path.name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', nargs='?', type=Path)
    p.add_argument('--kind', choices=['iq', 'audio'], default='iq')
    p.add_argument('--center-mhz', type=float)
    p.add_argument('--station-mhz', type=float)
    p.add_argument('--swap-iq', action='store_true')
    p.add_argument('--deemphasis-us', type=int, choices=[50, 75], default=50)
    p.add_argument('--seconds', type=float)
    p.add_argument('--model', type=Path, default=MODEL_PATH)
    p.add_argument('--voice-model', type=Path, default=VOICE_PATH)
    p.add_argument('--out-dir', type=Path, default=Path(os.environ.get('SDR_DATA_DIR', ROOT / 'data')))
    p.add_argument('--download-model', action='store_true')
    p.add_argument('--download-vocals', action='store_true')
    p.add_argument('--separate-vocals', action='store_true')
    p.add_argument('--fast-iq', action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--reanalyze-existing', action='store_true', help='Use the existing decoded audio and preserve its FM metadata.')
    args = p.parse_args()
    if args.input is None and not (args.download_model or args.download_vocals or args.reanalyze_existing):
        p.error('Укажи шлях до WAV або скористайся --download-model.')
    if args.seconds is not None and not .5 <= args.seconds <= MAX_SECONDS:
        p.error('--seconds: від 0.5 до 600.')
    try:
        if args.download_model:
            fetch_model(args.model, MODEL_URL, MODEL_SHA256)
            fetch_model(args.voice_model, VOICE_URL, VOICE_SHA256)
            print(f'Моделі збережено: {args.model} та {args.voice_model}', flush=True)
        if args.download_vocals:
            from fm_vocals import MODELS, REPO
            for name, sha in MODELS.values():
                fetch_model(ROOT / 'models' / ('spleeter_' + name), REPO + name, sha, max_bytes=50_000_000)
        if args.input is not None or args.reanalyze_existing:
            model = Yamnet(args.model)
            voice_model = VoiceModel(args.voice_model)
            print('Читаю WAV і готую монофонічний звук…', flush=True)
            if args.reanalyze_existing:
                old = json.loads((args.out_dir / 'fm_report.json').read_text(encoding='utf-8'))
                audio, _ = read_input(args.out_dir / 'fm_audio.wav', 'audio', None, None, False, 50, None)
                meta = {k: old[k] for k in ('filename', 'input_kind', 'input_sample_rate_hz', 'input_bits',
                                             'input_channels', 'center_mhz', 'station_mhz', 'deemphasis_us', 'duration_s')}
            else:
                audio, meta = read_input(args.input, args.kind, args.center_mhz, args.station_mhz,
                                         args.swap_iq, args.deemphasis_us, args.seconds,fast_iq=args.fast_iq)
            stem = None
            if args.separate_vocals:
                from fm_vocals import VocalSeparator
                print('Виділяю голос із музики…', flush=True)
                stem = VocalSeparator(ROOT / 'models', digest).separate(audio)
            print(f"Аналізую {meta['duration_s']:g} с звуку…", flush=True)
            report = classify(audio, model, voice_model, meta, stem)
            save_result(args.out_dir, audio, report, stem)
            print(json.dumps(report['totals_s'], ensure_ascii=False))
            print(f'Готово: {args.out_dir / "fm_report.json"} та {args.out_dir / "fm_audio.wav"}')
    except (ValueError, RuntimeError, OSError, wave.Error, EOFError, ImportError) as exc:
        print(f'Помилка: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
