"""Коротке прослуховування іншої FM-станції з того самого I/Q WAV."""
import math
from pathlib import Path
import wave

MARGIN_HZ = 110_000
PREVIEW_SECONDS = 12
PREVIEW_VERSION = 2
RATE = 16_000


def channel_bounds(center_mhz, fs):
    # Залишаємо запас біля країв, щоб умістити смугу FM-каналу.
    half = (fs / 2 - MARGIN_HZ) / 1e6
    return center_mhz - half, center_mhz + half


from drivers.wav_io import capture_info


def extract(source: Path, dest: Path, center_mhz, station_mhz, start_s, seconds=PREVIEW_SECONDS):
    import numpy as np

    if not 0 < seconds <= PREVIEW_SECONDS:
        raise ValueError('Некоректна довжина фрагмента.')
    x = demodulate(source,center_mhz,station_mhz,start_s,seconds)
    with wave.open(str(dest),'wb') as out:
        out.setparams((1,2,RATE,0,'NONE','not compressed'))
        out.writeframes(np.rint(x*32767).astype('<i2').tobytes())


def demodulate(source: Path, center_mhz, station_mhz, start_s, seconds, deemphasis_us=50):
    import numpy as np
    from scipy import signal

    fs, total = capture_info(source)
    low, high = channel_bounds(center_mhz, fs)
    if not low <= station_mhz <= high or not 0 <= start_s < total/fs or not 0 < seconds <= 600:
        raise ValueError('Частота або час поза межами запису.')
    offset = round((station_mhz - center_mhz)*1e6)
    dec = next(d for d in range(max(1, fs//240000), 0, -1) if fs%d==0)
    first = next((d for d in (4, 3, 2) if dec>d and dec%d==0), 1)
    second, rf = dec//first, fs//dec
    intermediate = fs//first
    # Першому каскаду достатньо прибрати спектр, що накладеться на робочу смугу.
    prefilter = None
    if first>1:
        stop = min(.49*fs, intermediate-120000)
        prefilter = signal.iirdesign(120000, stop, .2, 54, fs=fs, output='sos').astype(np.float32)
    channel_filter = signal.iirdesign(92000, min(121000,.49*intermediate), .3, 54,
                                     fs=intermediate, output='sos').astype(np.float32)
    deemphasis = math.exp(-1/(rf*deemphasis_us*1e-6))
    voice_filter = signal.butter(4,[45,7200], btype='bandpass', fs=rf, output='sos')
    phase_step = -2*np.pi*offset/fs
    grid = math.lcm(dec,fs//math.gcd(fs,RATE))
    halo = math.ceil(.025*fs/grid)*grid
    start = round(start_s*fs/grid)*grid
    end = min(total,start+round(seconds*fs))
    if end<=start:
        raise ValueError('Відрізок наприкінці запису надто короткий.')
    chunk = max(grid,round(.5*fs/grid)*grid)
    parts = []
    cache = None
    if offset:
        period = fs//math.gcd(abs(offset),fs)
        # Часовий індекс фази не скидається на межі блоків.
        if period<=32768:
            cache = np.exp(1j*phase_step*np.arange(period)).astype(np.complex64)
    with wave.open(str(source),'rb') as inp:
        width = inp.getsampwidth()
        for a in range(start,end,chunk):
            b = min(end,a+chunk)
            left,right = max(0,a-halo),min(total,b+halo)
            inp.setpos(left); raw = inp.readframes(right-left)
            if len(raw)!=(right-left)*2*width:
                raise ValueError('Запис WAV обірвано.')
            x = np.frombuffer(raw,{1:'u1',2:'<i2',4:'<i4'}[width]).astype(np.float32).reshape(-1,2)
            if width==1:x-=128
            if offset:
                z = x.view(np.complex64).ravel()
                if cache is not None:
                    begin = left%len(cache)
                    z *= np.tile(cache, math.ceil((begin+len(z))/len(cache)))[begin:begin+len(z)]
                else:
                    z *= np.exp(1j*phase_step*np.arange(left,right)).astype(np.complex64)
            if prefilter is not None:
                x = signal.sosfilt(prefilter,x,axis=0)[::first]
            x = signal.sosfilt(channel_filter,x,axis=0)[::second]
            i, q = x.T
            # Зміна фази сусідніх I/Q-відліків відновлює FM-повідомлення.
            ph = np.arctan2(q[1:]*i[:-1]-i[1:]*q[:-1],i[1:]*i[:-1]+q[1:]*q[:-1])
            ph = np.concatenate((ph[:1],ph))
            mono = signal.lfilter([1-deemphasis],[1,-deemphasis],ph)
            mono = signal.sosfilt(voice_filter,mono).astype(np.float32)
            gcd = math.gcd(RATE,rf)
            mono = signal.resample_poly(mono,RATE//gcd,rf//gcd)
            skip = (a-left)*RATE//fs
            size = round((b-start)*RATE/fs)-round((a-start)*RATE/fs)
            parts.append(mono[skip:skip+size])
    x = np.concatenate(parts)
    peak = float(np.quantile(np.abs(x),.998))
    return np.clip(x*min(16,.65/max(peak,1e-6)),-.99,.99)
