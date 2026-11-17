"""Вибрані в браузері записи та окремі звіти для кожної станції."""
import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import uuid
import wave
from dataclasses import dataclass
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Path as RoutePath, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.requests import ClientDisconnect

from fm_channel import channel_bounds

MAX_WAV_BYTES = 4 * 1024**3
MAX_SECONDS = 600
MIN_FREE_BYTES = 64 * 1024**2
CaptureId = Annotated[str, RoutePath(pattern=r'^[0-9a-f]{64}$')]
JobId = Annotated[str, RoutePath(pattern=r'^[0-9a-f]{32}$')]
Frequency = Annotated[float, Field(gt=0, le=100000, allow_inf_nan=False)]


class Selection(BaseModel):
    center_mhz: Frequency | None = None
    station_mhz: Frequency | None = None


def wav_info(path, filename, size):
    try:
        with wave.open(str(path), 'rb') as wav:
            channels, width, rate, frames, *_ = wav.getparams()
            iq = channels == 2 and rate >= 240000
            if (channels not in (1, 2) or width not in (1, 2, 4) or not 8000 <= rate <= 5_000_000
                    or not iq and rate > 192000):
                raise ValueError()
            if frames:
                wav.setpos(frames-1)
                if len(wav.readframes(1)) != channels*width:
                    raise ValueError()
    except (ValueError, wave.Error, EOFError, OverflowError) as exc:
        raise HTTPException(422, 'Потрібен повний PCM WAV із 1–2 каналами та 8, 16 або 32 бітами.') from exc
    if not .5 <= frames/rate <= MAX_SECONDS:
        raise HTTPException(422, 'Вибери запис тривалістю від 0,5 с до 10 хвилин.')
    match = re.search(r'(?:^|[_-])(\d{7,10})\s*Hz(?:[_\-.]|$)', filename, re.I) if iq else None
    return {'filename':filename, 'kind':'iq' if iq else 'audio', 'sample_rate_hz':rate,
            'duration_s':frames/rate, 'center_mhz':int(match[1])/1e6 if match else None, 'size':size}


@dataclass
class Job:
    path: Path
    args: list[str]
    capture: dict
    stage: str = 'queued'
    process: asyncio.subprocess.Process | None = None
    task: asyncio.Task | None = None

    async def stop(self):
        self.stage = 'cancelled'
        process = self.process
        if process is None or process.returncode is not None:
            return
        try:
            if os.name == 'nt':
                kill = await asyncio.create_subprocess_exec('taskkill', '/T', '/F', '/PID', str(process.pid),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
                if await kill.wait() and process.returncode is None:
                    process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


class Analyses:
    def __init__(self, folder, script):
        self.root = folder / 'browser_fm'
        self.script = script
        self.active = {}
        self.by_channel = {}
        self.worker = asyncio.Lock()
        self.routes = APIRouter(prefix='/api/fm/captures')
        self.routes.add_api_route('', self.upload, methods=['POST'], status_code=201)
        self.routes.add_api_route('/{capture}/analysis', self.start, methods=['POST'], status_code=202)
        self.routes.add_api_route('/{capture}/analysis/{job}', self.status, methods=['GET'])
        self.routes.add_api_route('/{capture}/analysis/{job}', self.cancel, methods=['DELETE'])
        self.routes.add_api_route('/{capture}/analysis/{job}/{artifact}', self.artifact, methods=['GET'])

    async def upload(self, request: Request, name: Annotated[str, Query(min_length=1, max_length=255)]):
        if '/' in name or '\\' in name or any(ord(c)<32 for c in name):
            raise HTTPException(422, 'Некоректна назва файлу.')
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            expected = int(request.headers.get('content-length','0'))
        except ValueError as exc:
            raise HTTPException(400, 'Некоректний розмір файлу.') from exc
        if expected > MAX_WAV_BYTES:
            raise HTTPException(413, 'Файл WAV більший за 4 ГіБ.')
        if shutil.disk_usage(self.root).free < expected+MIN_FREE_BYTES:
            raise HTTPException(507, 'На Raspberry Pi бракує місця для цього WAV.')
        sha = hashlib.sha256(name.encode('utf-8')+b'\0')
        length = 0
        staging = Path(tempfile.mkdtemp(prefix='.upload-', dir=self.root))
        source = staging/'input.wav'
        try:
            with source.open('wb') as target:
                async for block in request.stream():
                    length += len(block)
                    if length > MAX_WAV_BYTES:
                        raise HTTPException(413, 'Файл WAV більший за 4 ГіБ.')
                    sha.update(block)
                    await asyncio.to_thread(target.write,block)
            if not length or expected and expected != length:
                raise HTTPException(422, 'WAV завантажився не повністю.')
            info = await asyncio.to_thread(wav_info,source,name,length)
            key = sha.hexdigest()
            info['id'] = key
            destination = self.root/key
            if not destination.exists():
                (staging/'capture.json').write_text(json.dumps(info,ensure_ascii=False),encoding='utf-8')
                staging.rename(destination)
            return info
        except ClientDisconnect as exc:
            raise HTTPException(400, 'Передавання WAV перервано.') from exc
        except OSError as exc:
            logging.warning('Збереження вибраного WAV: %s',exc)
            raise HTTPException(507, 'Не вдалося зберегти WAV на Raspberry Pi.') from exc
        finally:
            shutil.rmtree(staging,ignore_errors=True)

    def capture_info(self, capture):
        try:
            return json.loads((self.root/capture/'capture.json').read_bytes())
        except FileNotFoundError as exc:
            raise HTTPException(404, 'Спочатку вибери WAV на сторінці водоспаду.') from exc

    async def start(self, capture: CaptureId, selection: Selection):
        info = self.capture_info(capture)
        center, station = selection.center_mhz, selection.station_mhz
        if info['kind']=='iq':
            center = info['center_mhz'] if info['center_mhz'] is not None else center
            if center is None or station is None:
                raise HTTPException(422, 'Вкажи центр I/Q та частоту станції.')
            center, station = round(center,6),round(station,6)
            low, high = channel_bounds(center,info['sample_rate_hz'])
            if not low-1e-9 <= station <= high+1e-9:
                raise HTTPException(422, f'Обери частоту в межах {low:.3f}–{high:.3f} МГц.')
        else:
            center = station = None
        key = (capture,center,station)
        previous = self.by_channel.get(key)
        if previous and previous.stage == 'ready':
            return {'id':previous.path.name, 'state':previous.stage}
        path = self.root/capture/uuid.uuid4().hex
        path.mkdir()
        args = [str(path.parent/'input.wav'),'--kind',info['kind'],'--data-dir',str(path),'--browser-analysis']
        if center is not None:
            args += ['--center-mhz',str(center),'--station-mhz',str(station)]
        item = Job(path,args,info)
        self.active[path.name] = self.by_channel[key] = item
        item.task = asyncio.create_task(self.run(item))
        return {'id':path.name, 'state':item.stage}

    async def run(self, job):
        try:
            async with self.worker:
                if job.stage=='cancelled':
                    return
                job.stage = 'running'
                with (job.path/'analysis.log').open('wb') as log:
                    options = {'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {'start_new_session':True}
                    job.process = await asyncio.create_subprocess_exec(sys.executable, str(self.script), *job.args,
                        stdout=log, stderr=subprocess.STDOUT, cwd=self.script.parent, **options)
                    if job.stage=='cancelled':
                        await job.stop()
                    code = await job.process.wait()
                if job.stage == 'cancelled':
                    return
                if code:
                    raise RuntimeError(f'Код завершення {code}')
                report = job.path/'fm_report.json'
                result = json.loads(report.read_bytes())
                result['filename'] = job.capture['filename']
                report.write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8')
                (job.path/'ready').touch()
                job.stage = 'ready'
        except (OSError, RuntimeError, ValueError) as exc:
            logging.warning('Аналіз WAV %s: %s',job.path.name,exc)
            job.stage = 'failed'
        finally:
            self.active.pop(job.path.name,None)
            if job.stage!='ready':
                for partial in job.path.glob('.prepare-*'):
                    shutil.rmtree(partial,ignore_errors=True)
                (job.path/job.stage).touch()

    def find(self, capture, job):
        path = self.root/capture/job
        if not path.is_dir():
            raise HTTPException(404, 'Аналіз цього запису не знайдено.')
        return path

    async def status(self, capture: CaptureId, job: JobId):
        path = self.find(capture,job)
        item = self.active.get(job)
        if item:
            state = item.stage
        else:
            state = next((s for s in ('ready','cancelled','failed') if (path/s).is_file()), 'failed')
        text = {'queued':'Чекаю на звільнення Raspberry Pi…', 'running':'Починаю аналіз…',
                'ready':'Аналіз готовий.', 'cancelled':'Аналіз зупинено.',
                'failed':'Не вдалося завершити аналіз. Спробуй ще раз.'}[state]
        if state=='running' and (path/'progress.txt').is_file():
            text = (path/'progress.txt').read_text(encoding='utf-8') or text
        return {'id':job,'state':state,'message':text}

    async def cancel(self, capture: CaptureId, job: JobId):
        self.find(capture,job)
        item = self.active.get(job)
        if item:
            await item.stop()
        return await self.status(capture,job)

    async def artifact(self, capture: CaptureId, job: JobId, artifact: Literal['report','audio','neural']):
        path = self.find(capture,job)
        name = {'report':'fm_report.json','audio':'fm_audio.wav','neural':'fm_voice_focus.wav'}[artifact]
        target = path/name
        if not (path/'ready').is_file() or not target.is_file():
            raise HTTPException(404, 'Результат ще готується.')
        return FileResponse(target, media_type='application/json' if artifact=='report' else 'audio/wav',
                            headers={'Cache-Control':'no-store'})

    async def close(self):
        tasks = list(self.active.values())
        await asyncio.gather(*(job.stop() for job in tasks))
        await asyncio.gather(*(job.task for job in tasks))
