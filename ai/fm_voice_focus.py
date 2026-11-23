"""Enhance speech with separated vocals, and check uncertain FM labels with Silero VAD."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
import wave

import numpy as np
from scipy import ndimage

from ai.fm_analyze import VoiceModel, voice_stats, write_wav
from ai.fm_neural_audio import enhance

from config import ROOT, MODEL_DIR
RATE, STEP = 16000, 512
VAD_URL = 'https://raw.githubusercontent.com/snakers4/silero-vad/v6.2.1/src/silero_vad/data/silero_vad.onnx'
VAD_SHA = '1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3'


def prepare_model(path):
    with urlopen(VAD_URL, timeout=90) as f:
        data = f.read(4_000_000)
    if hashlib.sha256(data).hexdigest() != VAD_SHA:
        raise ValueError('Контрольна сума Silero VAD не збігається.')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); tmp.write_bytes(data); tmp.replace(path)


class Vad:
    def __init__(self, path):
        import onnxruntime as ort
        if hashlib.sha256(path.read_bytes()).hexdigest() != VAD_SHA:
            raise ValueError('Контрольна сума Silero VAD не збігається.')
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=opts, providers=['CPUExecutionProvider'])

    def run(self, x):
        tail = (-len(x))%STEP
        blocks = np.pad(x,(64,tail))
        # Для нової доріжки стан VAD скидаємо, далі передаємо між кадрами.
        mem = np.zeros((2,1,128),np.float32)
        fs = np.asarray(RATE,dtype=np.int64)
        result = np.zeros((len(x)+STEP-1)//STEP,np.float32)
        for frame in range(len(result)):
            t = frame*STEP
            s,mem = self.session.run(None, {'input': blocks[t:t+STEP+64][None,:], 'state':mem, 'sr':fs})
            result[frame] = s.item()
        return result


def wav_read(p):
    with wave.open(str(p), 'rb') as f:
        if (f.getnchannels(), f.getsampwidth(), f.getframerate()) != (1,2,RATE):
            raise ValueError('Потрібні доріжки моно WAV, 16 біт / 16 кГц.')
        return np.frombuffer(f.readframes(f.getnframes()),'<i2').astype(np.float32)/32768


def isolated_voice_context(report, isolated, p_iso, voice):
    """Оцінюємо голос у кількох вікнах тієї ж репліки, не змінюючи таймкоди."""
    # Закриваємо короткі паузи між словами. Довгі паузи обмежують контекст.
    connected = ndimage.binary_closing(p_iso >= .40, structure=np.ones(9, bool), border_value=1)
    resolved = 0
    for s in report['segments']:
        kind = s['content']
        if kind not in ('speech','mixed','music_voice'):
            continue
        prior = s.get('voice')
        if kind != 'speech' or prior in (None,'unknown'):
            s.update(voice='unknown', voice_source=None, voice_female_score=None, f0_hz=None, voiced_frames=0)
        else:
            continue  # Однозначну оцінку чистого мовлення не переглядаємо.
        if s.get('speech_activity') != 'confirmed' or s.get('singing_score',0) >= .12:
            continue
        start,stop = (round(s[k]*RATE) for k in ('start_s','end_s'))
        begin,end = (start+STEP//2)//STEP,(stop+STEP//2)//STEP
        hits = np.flatnonzero(p_iso[begin:end]>=.65)+begin
        if len(hits) < 4:
            continue
        left=right=int(hits[len(hits)//2]); halo=round(1.15*RATE/STEP)
        while left>max(0,begin-halo) and connected[left-1]:left-=1
        while right<min(len(connected)-1,end+halo) and connected[right+1]:right+=1
        low, high = left*STEP, min(len(isolated),(right+1)*STEP)
        length=min(high-low,round(2.65*RATE))
        if length < 1.5*RATE:
            continue
        pivot=(start+stop)//2
        # Вікна трохи зсунуті одне відносно одного, але обидва містять потрібну репліку.
        starts=[int(np.clip(pivot - length//2 + round(d*RATE),low,high-length)) for d in (-.22,.22)]
        estimates=[voice.estimate(isolated[k:k+length]) for k in starts]
        (a,v),(b,w)=estimates
        if a==b and a!='unknown' and v is not None and w is not None and min(abs(v-.5),abs(w-.5))>=.38:
            k=starts[0]
            pitch, n = voice_stats(isolated[k:k+length])
            s.update(voice=a, voice_source='isolated_context', voice_female_score=round((v+w)/2,3),
                     f0_hz=pitch, voiced_frames=n)
            resolved+=prior in (None,'unknown')
    return resolved


def review(report, isolated, p_raw, p_main, p_iso, voice):
    before = report['totals_s'].copy()
    counts = {'reclassified':0, 'music_with_speech':0}
    for s in report['segments']:
        start,stop = round(s['start_s']*RATE),round(s['end_s']*RATE)
        begin,end = (start+STEP//2)//STEP,(stop+STEP//2)//STEP
        direct, clean, stem = (p[begin:end] for p in (p_raw,p_main,p_iso))
        if not len(direct):
            continue
        # A music-backed voice must be evident in the vocal stem and an independent mix view.
        # Requiring the raw speech branch as well helps reject sung or instrumental leakage.
        main = np.minimum(direct,np.maximum(clean,stem))
        both = (main>=.68)&(stem>=.7)
        overlap = float(np.mean(both))
        active = float(np.mean((direct>=.72)&(clean>=.68)))
        s['speech_activity'] = 'confirmed' if overlap>=.50 or active>=.60 else 'possible' if max(overlap,active)>=.20 else 'unconfirmed'
        possible_music = s['content'] in ('music','music_voice','mixed') or (s['content']=='unknown' and s.get('music_score',0)>=.25)
        singing = s['content']=='music_vocals' or s.get('singing_score',0)>=.12
        old = s['content']
        if possible_music and not singing and stop-start >= .72*RATE and both.sum() >= 16 and overlap>=.50:
            s['content']='mixed';s['voice_kind']='speech'
            if old!='mixed': counts['music_with_speech'] += 1
        elif old=='unknown' and not singing and not possible_music and stop-start>=.65*RATE and active>=.62 and np.sum((direct>=.78)&(clean>=.75))>=15:
            s['content']='speech';s['voice_kind']='speech'
        if s['content']!=old:
            s['speech_rechecked']=True; counts['reclassified']+=1
    counts['voice_context_resolved'] = isolated_voice_context(report, isolated, p_iso, voice)
    totals = {k:0. for k in before}
    for s in report['segments']:
        totals[s['content']] += s['end_s']-s['start_s']
    report['totals_s'] = {k:round(v,3) for k,v in totals.items()}
    report['voice_focus'] = {'available':True, 'method':'Spleeter + DTLN + Silero VAD', **counts,
                             'unknown_before_s':before.get('unknown',0), 'unknown_after_s':report['totals_s']['unknown']}
    return report


def main():
    p=argparse.ArgumentParser(description='Посилити мовлення на тлі музики і перевірити його Silero VAD.')
    p.add_argument('--original',type=Path)
    p.add_argument('--neural',type=Path)
    p.add_argument('--vocals',type=Path)
    p.add_argument('--report',type=Path)
    p.add_argument('--out-dir',type=Path)
    p.add_argument('--model',type=Path,default=MODEL_DIR/'silero_vad.onnx')
    p.add_argument('--voice-model',type=Path,default=MODEL_DIR/'voice_classifier.onnx')
    p.add_argument('--dtln-models',type=Path,default=MODEL_DIR/'dtln')
    p.add_argument('--clean-vocals',type=Path,help='Використати готову очищену копію тієї самої виділеної доріжки.')
    p.add_argument('--download-model',action='store_true')
    a=p.parse_args()
    if a.download_model:
        prepare_model(a.model)
        if not a.original:return
    if not all((a.original,a.neural,a.vocals,a.report,a.out_dir)):
        p.error('Потрібні --original, --neural, --vocals, --report та --out-dir.')
    raw,mix,stem = map(wav_read,(a.original,a.neural,a.vocals))
    report=json.loads(a.report.read_bytes())
    if not (len(raw)==len(mix)==len(stem)==round(report['duration_s']*RATE)) or not report.get('separated_vocals'):
        raise ValueError('Доріжки та звіт мають відповідати тому самому запису.')
    if report.get('voice_focus'):
        raise ValueError('Потрібен звіт, ще не перевірений Silero VAD.')
    print('Приглушую залишковий шум на виділеній доріжці…',flush=True)
    cleaned_stem = wav_read(a.clean_vocals) if a.clean_vocals else enhance(stem,a.dtln_models)
    if len(cleaned_stem)!=len(stem) or not np.isfinite(cleaned_stem).all():
        raise ValueError('Очищена доріжка має відповідати довжині запису.')
    print('Перевіряю мовлення на трьох доріжках…',flush=True)
    vad=Vad(a.model)
    # Зіставляємо мовлення в оригіналі, очищеній суміші й очищеному голосі.
    p_raw,p_main,p_iso=(vad.run(x) for x in (raw,mix,cleaned_stem))
    report=review(report,stem,p_raw,p_main,p_iso,VoiceModel(a.voice_model))
    report['neural_audio']={'available':True,'method':'DTLN','purpose':'speech_enhancement'}
    a.out_dir.mkdir(parents=True,exist_ok=True)
    # У плеєрі лишається одна нейродоріжка: DTLN після відокремлення вокалу, без підмішування оригіналу.
    write_wav(a.out_dir/'fm_voice_focus.wav',cleaned_stem)
    (a.out_dir/'fm_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(report['voice_focus'],flush=True)


if __name__=='__main__':
    main()
