"""Look for a persistent 19 kHz stereo pilot in FM-discriminated I/Q snippets."""
import argparse
import json
import math
from pathlib import Path
import wave

import numpy as np
from scipy import fft, signal


def pilot_spectrum(x, rate):
    nfft = fft.next_fast_len(2 * len(x), real=True)
    f = fft.rfftfreq(nfft, 1/rate)
    s = np.abs(fft.rfft((x-np.mean(x))*signal.get_window('hann', len(x)), nfft))**2
    narrow = np.flatnonzero(np.abs(f-19000) <= 22)
    guard = (np.abs(f-19000) >= 70) & (np.abs(f-19000) <= 400)
    b = narrow[np.argmax(s[narrow])]
    floor = max(float(np.quantile(s[guard], .75)), 1e-14)
    return float(f[b]), float(10*np.log10(max(s[b]/floor, 1e-12)))


def inspect(path, center, station):
    from fm_analyze import pcm_block
    with wave.open(str(path), 'rb') as wav:
        channels, width, rate, count, *_ = wav.getparams()
        offset = (station-center)*1e6
        if channels != 2 or rate < 240000 or abs(offset)+115000 >= rate/2:
            raise ValueError('Для перевірки пілот-тону потрібен I/Q WAV із повною смугою станції.')
        dec = next(i for i in range(max(1,rate//240000),0,-1) if rate % i == 0)
        rf = rate//dec; size = min(count, round(.21*rate))
        if size < round(.12*rate):
            raise ValueError('Запис надто короткий для перевірки пілот-тону.')
        taps = signal.firwin(48*dec+1,103000,fs=rate).astype(np.float32)
        starts = np.linspace(0,count-size,min(60,max(1,math.ceil(count/rate/4))),dtype=np.int64)
        if offset:
            osc = np.exp(-2j*np.pi*offset*np.arange(size)/rate).astype(np.complex64)
        samples = []
        for start in starts:
            wav.setpos(int(start)); raw = wav.readframes(size)
            if len(raw) != size*channels*width:
                raise ValueError('Запис WAV обірвано.')
            p = pcm_block(raw,width,channels)
            z = p[:,0]+1j*p[:,1]
            if offset: z *= osc
            z = signal.upfirdn(taps,z,down=dec)[48:-48]
            x = np.angle(z[1:]*z[:-1].conj())
            samples.append(pilot_spectrum(x,rf))
    bins, strengths = map(np.array, zip(*samples))
    first = strengths >= 12
    peak_hz = float(np.median(bins[first])) if first.any() else None
    matches = int(np.sum(first & (np.abs(bins-(peak_hz or 19000)) <= 6)))
    present = len(samples) >= 4 and matches/len(samples) >= .60
    return {'center_mhz': center, 'station_mhz': station, 'matches': matches, 'checked': len(samples),
            'present': present, 'peak_hz': round(peak_hz,1) if peak_hz is not None else None,
            'median_db_above_local_background': round(float(np.median(strengths)),1)}


def main():
    p = argparse.ArgumentParser(description='Перевірити стереоознаку FM у вихідному I/Q WAV.')
    p.add_argument('iq', type=Path)
    p.add_argument('--radio', type=Path, required=True)
    a = p.parse_args()
    radio = json.loads(a.radio.read_bytes())
    if radio['mode'] != 'iq' or radio['filename'] != a.iq.name:
        raise ValueError('Спектр і I/Q WAV мають відповідати одному запису.')
    result = inspect(a.iq,radio['center_mhz'],radio['station_mhz'] or radio['center_mhz'])
    radio['stereo_pilot'] = result
    pending = a.radio.with_suffix('.tmp')
    pending.write_text(json.dumps(radio,ensure_ascii=False,separators=(',',':')),encoding='utf8')
    pending.replace(a.radio)
    print(result,flush=True)


if __name__ == '__main__':
    main()
