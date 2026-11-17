"""Build a compact RF waterfall and an optional listening track with less stationary noise."""
import argparse
import json
import math
from pathlib import Path
import wave

import numpy as np
from scipy import fft, ndimage, signal

NFFT = 1024


def radio_rows(path, center_mhz, station_mhz=None):
    with wave.open(str(path), 'rb') as f:
        channels, width, rate, frames, *_ = f.getparams()
        if width not in (1, 2, 4) or channels not in (1, 2) or rate <= 0 or frames < NFFT:
            raise ValueError('Потрібен PCM WAV з 1–2 каналами і розрядністю 8, 16 або 32 біти.')
        is_iq = channels == 2 and rate >= 240_000
        if is_iq and (center_mhz is None or center_mhz <= 0 or not math.isfinite(center_mhz)):
            raise ValueError('Для I/Q WAV потрібна центральна частота в МГц.')
        duration = frames / rate
        count = min(960, max(1, int(np.ceil(duration * 4))))
        win = signal.get_window('hann', NFFT).astype(np.float32)
        scale = np.sum(win * win) * NFFT
        rows = []
        for i in range(count):
            a, b = i * frames // count, (i + 1) * frames // count
            # Distribute eight non-overlapping windows across each displayed time bin.
            last = min(frames - NFFT, max(a, b - NFFT))
            starts = np.linspace(min(a, last), last, min(8, max(1, (b-a)//NFFT)), dtype=np.int64)
            powers = []
            for start in starts:
                f.setpos(int(start))
                raw = f.readframes(NFFT)
                if len(raw) != NFFT * channels * width:
                    raise ValueError('Файл WAV обривається до кінця запису.')
                d = np.frombuffer(raw, {1: 'u1', 2: '<i2', 4: '<i4'}[width]).astype(np.float32)
                if width == 1:
                    d -= 128
                d = (d / (2 ** (width * 8 - 1))).reshape(-1, channels)
                x = d[:, 0] + 1j*d[:, 1] if is_iq else d.mean(axis=1)
                x -= x.mean()
                s = fft.fftshift(fft.fft(x*win)) if is_iq else fft.rfft(x*win)
                powers.append(np.abs(s)**2 / scale)
            lev = 10*np.log10(np.maximum(np.mean(powers, axis=0), 1e-12))
            lev = np.maximum(-120, np.minimum(10, lev))
            rows.append(np.rint(lev).astype(np.int16).tolist())
    return {'schema': 1, 'filename': path.name, 'mode': 'iq' if is_iq else 'audio',
            'duration_s': duration, 'sample_rate_hz': rate, 'center_mhz': center_mhz if is_iq else None,
            'station_mhz': (station_mhz or center_mhz) if is_iq else None, 'rows': rows}


def denoise(audio, rate=16000):
    """A conservative spectral gate; infer a floor from the quietest frames per band."""
    nperseg, hop = 512, 128
    # Select a short sample from every second to estimate the floor across the full recording.
    pieces = [audio[i:i+rate//5] for i in range(0, len(audio), rate) if i+rate//5 <= len(audio)]
    snippets = np.concatenate(pieces) if pieces else audio
    _, _, quiet = signal.stft(snippets, rate, nperseg=nperseg, noverlap=nperseg-hop)
    noise = np.quantile(np.abs(quiet)**2, .15, axis=1).astype(np.float32)
    noise = ndimage.gaussian_filter1d(noise, 1.1)
    result = np.empty_like(audio)
    chunk, halo = rate * 12, rate // 8
    for i in range(0, len(audio), chunk):
        stop, left = min(i + chunk, len(audio)), max(0, i - halo)
        x = audio[left:min(stop + halo, len(audio))]
        _, _, spec = signal.stft(x, rate, nperseg=nperseg, noverlap=nperseg-hop)
        power = np.abs(spec)**2
        attenuation = np.sqrt(np.clip(1 - 2.25 * noise[:, None] / np.maximum(power, 1e-12), .24, 1))
        attenuation = ndimage.gaussian_filter(attenuation, (.75, 1.15))
        _, clean = signal.istft(spec * attenuation, rate, nperseg=nperseg, noverlap=nperseg-hop)
        result[i:stop] = clean[i-left:stop-left]
    return result


def refine_report(report, clean, yamnet, voice_model):
    """Recheck only ambiguous intervals on the cleaned track with conservative thresholds."""
    from fm_analyze import RATE, WINDOW, voice_stats
    original = report['totals_s'].copy()
    changed = 0
    for segment in report['segments']:
        if segment['content'] != 'unknown':
            continue
        start, end = (round(segment[k] * RATE) for k in ('start_s', 'end_s'))
        if end-start < RATE:  # Do not relabel brief transitions from adjacent audio.
            continue
        scores = []
        for center in np.linspace(start + WINDOW//2, end - WINDOW//2, min(5, max(2, (end-start)//(RATE//2))), dtype=int):
            clip = clean[center-WINDOW//2:center+WINDOW//2]
            if len(clip) != WINDOW or np.sqrt(np.mean(clip*clip)) < .005:
                continue
            scores.append(yamnet.score(clip))
        if len(scores) < 2:
            continue
        s, m, g = np.median(scores, axis=0)
        if s >= .52 and max(m, g) < .30:
            kind, voice_kind = 'speech', 'speech'
        elif m >= .65 and s < .28 and g < .25:
            kind, voice_kind = 'music', None
        elif s >= .43 and m >= .43 and g < .30:
            kind, voice_kind = 'mixed', 'speech'
        elif g >= .47 and s < .30:
            kind, voice_kind = 'music_vocals', 'singing'
        else:
            continue
        segment.update(content=kind, voice_kind=voice_kind, noise_rechecked=True,
                       speech_score=round(float(s), 3), music_score=round(float(m), 3), singing_score=round(float(g), 3))
        if voice_kind is not None:
            segment['voice'] = 'unknown'
        if kind == 'speech':
            clip = clean[start:end]
            label, score = voice_model.estimate(clip)
            pitch, n = voice_stats(clip)
            segment.update(voice=label, voice_female_score=score, f0_hz=pitch,
                           voiced_frames=n, voice_source='cleaned')
        changed += 1
    totals = {k: 0.0 for k in original}
    for s in report['segments']:
        totals[s['content']] += s['end_s'] - s['start_s']
    report['totals_s'] = {k: round(v, 3) for k, v in totals.items()}
    report['noise_reduction'] = {'available': True, 'method': 'spectral gate', 'rechecked_segments': changed,
                                 'unknown_before_s': original.get('unknown', 0), 'unknown_after_s': report['totals_s']['unknown']}
    return report


def main():
    p = argparse.ArgumentParser(description='Підготувати водоспад із запису WAV або приглушити фоновий шум.')
    p.add_argument('--iq-wav', type=Path)
    p.add_argument('--center-mhz', type=float)
    p.add_argument('--station-mhz', type=float)
    p.add_argument('--audio-wav', type=Path)
    p.add_argument('--refine-report', type=Path)
    p.add_argument('--model', type=Path)
    p.add_argument('--voice-model', type=Path)
    p.add_argument('--out-dir', type=Path, required=True)
    args = p.parse_args()
    if args.refine_report and not args.audio_wav:
        p.error('--refine-report потребує --audio-wav')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.iq_wav:
        r = radio_rows(args.iq_wav, args.center_mhz, args.station_mhz)
        dest = args.out_dir / 'fm_radio.json'
        dest.write_text(json.dumps(r, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
        (args.out_dir / 'fm_radio_source.txt').write_text(r['filename']+'\n',encoding='utf-8')
        print(f'{len(r["rows"])} кадрів водоспаду: {dest}', flush=True)
    if args.audio_wav:
        with wave.open(str(args.audio_wav), 'rb') as f:
            if (f.getnchannels(), f.getsampwidth(), f.getframerate()) != (1, 2, 16000):
                raise ValueError('Для шумозаглушення потрібен декодований моно WAV, 16 біт / 16 кГц.')
            a = np.frombuffer(f.readframes(f.getnframes()), '<i2').astype(np.float32) / 32768
        c = denoise(a)
        dest = args.out_dir / 'fm_clean.wav'
        with wave.open(str(dest), 'wb') as f:
            f.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            f.writeframes(np.rint(np.clip(c, -1, .999) * 32767).astype('<i2').tobytes())
        print(f'Очищена доріжка: {dest}', flush=True)
        if args.refine_report:
            from fm_analyze import MODEL_PATH, VOICE_PATH, VoiceModel, Yamnet
            report = json.loads(args.refine_report.read_text(encoding='utf-8'))
            if abs(report['duration_s'] - len(a) / 16000) > .001:
                raise ValueError('Тривалість WAV не відповідає звіту.')
            if report.get('noise_reduction'):
                raise ValueError('Для повторної оцінки потрібен початковий звіт без шумозаглушення.')
            report = refine_report(report, c, Yamnet(args.model or MODEL_PATH), VoiceModel(args.voice_model or VOICE_PATH))
            dest = args.out_dir / 'fm_report.json'
            dest.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
            print(report['noise_reduction'], flush=True)


if __name__ == '__main__':
    main()
