"""Підготовка WAV для спектра, FM-звіту й двох доріжок вебплеєра."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import wave

ROOT = Path(__file__).resolve().parent


def run(script, *args):
    subprocess.run([sys.executable, str(ROOT / script), *map(str,args)], cwd=ROOT, check=True)


def prepare(source, dest, kind, center, station, *, browser=False):
    with wave.open(str(source), 'rb') as f:
        if kind == 'auto':
            kind = 'iq' if f.getnchannels() == 2 and f.getframerate() >= 240000 else 'audio'
    if kind == 'iq' and center is None:
        match = re.search(r'(?:^|[_-])(\d{7,10})\s*Hz(?:[_\-.]|$)',source.name,re.I)
        if match is None:
            raise ValueError('Для I/Q вкажи центральну частоту через --center-mhz.')
        center = int(match[1])/1e6
    extra = [] if kind == 'audio' else ['--center-mhz',center,'--station-mhz',station or center]
    dest.mkdir(parents=True,exist_ok=True)

    def step(text):
        if browser:
            progress = dest/'progress.tmp'
            progress.write_text(text,encoding='utf-8')
            progress.replace(dest/'progress.txt')

    # Готуємо всі результати окремо: помилка в моделі не зачіпає чинний звіт.
    with tempfile.TemporaryDirectory(dir=dest,prefix='.prepare-') as tmp:
        p = Path(tmp)
        report = p/'fm_report.json'
        decoded = p/'fm_audio.wav'
        step('Крок 1 із 4 · Декодую звук, відокремлюю голос і розпізнаю музику та мовлення…')
        faster_iq = ['--fast-iq'] if browser and kind=='iq' else []
        run('fm_analyze.py',source,'--kind',kind,'--separate-vocals','--out-dir',p,*extra,*faster_iq)
        step('Крок 2 із 4 · Перевіряю фрагменти після приглушення шуму…')
        waterfall = ['--iq-wav',source] if not browser else []
        run('fm_quality.py',*waterfall,'--audio-wav',decoded,'--refine-report',report,'--out-dir',p,*extra)
        if kind=='iq' and not browser:
            run('fm_station.py',source,'--radio',p/'fm_radio.json')
        step('Крок 3 із 4 · Очищую звук нейромережею…')
        run('fm_neural_audio.py',decoded,'--out',p/'fm_neural.wav','--report',report)
        step('Крок 4 із 4 · Перевіряю мовлення й оцінюю тип голосу…')
        run('fm_voice_focus.py','--original',decoded,'--neural',p/'fm_neural.wav',
            '--vocals',p/'fm_vocals.wav','--report',report,'--out-dir',p)
        result = json.loads(report.read_bytes())
        with wave.open(str(p/'fm_voice_focus.wav'),'rb') as f:
            if result['filename']!=source.name or abs(f.getnframes()/f.getframerate()-result['duration_s'])>.001:
                raise ValueError('Доріжка та звіт не відповідають одному запису.')
        (p/'fm_source.json').write_text(json.dumps({'path':str(source),'size':source.stat().st_size}),encoding='utf8')
        # Звіт публікуємо останнім: він посилається на готові WAV та водоспад.
        for path in sorted(p.iterdir(),key=lambda x:x==report):
            path.replace(dest/path.name)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input',type=Path,nargs='?')
    ap.add_argument('--center-mhz',type=float)
    ap.add_argument('--station-mhz',type=float)
    ap.add_argument('--kind',choices=('iq','audio','auto'),default='auto')
    ap.add_argument('--data-dir',type=Path,default=Path(os.environ.get('SDR_DATA_DIR',str(ROOT/'data'))))
    ap.add_argument('--download-models',action='store_true')
    ap.add_argument('--browser-analysis',action='store_true',help=argparse.SUPPRESS)
    a=ap.parse_args()
    if a.input is None and not a.download_models:ap.error('Укажи WAV або --download-models.')
    try:
        if a.download_models:
            run('fm_analyze.py','--download-model','--download-vocals')
            run('fm_neural_audio.py','--download-model')
            run('fm_voice_focus.py','--download-model')
        if a.input:
            prepare(a.input.resolve(strict=True),a.data_dir.resolve(),a.kind,a.center_mhz,a.station_mhz,browser=a.browser_analysis)
            print(f'Готово: {a.data_dir / "fm_report.json"}')
    except (ValueError,OSError,wave.Error,subprocess.CalledProcessError) as e:
        ap.exit(1,f'Помилка: {e}\n')


if __name__=='__main__':
    main()
