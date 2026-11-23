"""Local SDR recording viewer. Python 3.10+, one Uvicorn worker."""
import asyncio
import hashlib
import json
import logging
import math
import os
import tempfile
import time
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from config import ROOT, MODEL_DIR, STATIC_DIR, EXAMPLES_DIR
MAX_BYTES = 16 * 1024 * 1024
Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Frequency = Annotated[Number, Field(gt=0, le=100000)]


class Frame(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frequency: Frequency  # Center frequency in MHz, not an individual FFT bin.
    fft_data: Annotated[list[Number], Field(min_length=2, max_length=4096)]
    timestamp: Annotated[str, Field(max_length=80)] | None = None


class Recording(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sample_interval_ms: Annotated[int, Field(strict=True, ge=50, le=10000)] = 100
    sample_rate_hz: Annotated[Number, Field(gt=0, le=1e9)] | None = None
    power_unit: Annotated[str, Field(min_length=1, max_length=32)] = "умовні одиниці"
    source_kind: Literal["recording", "demo"] = "recording"
    frames: Annotated[list[Frame], Field(min_length=1, max_length=5000)]

    @model_validator(mode="after")
    def consistent_bins(self):
        if len({len(frame.fft_data) for frame in self.frames}) != 1:
            raise ValueError("Кількість точок FFT має бути однаковою в усіх кадрах")
        return self


class FrequencyCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frequency_mhz: Frequency


def parse_recording(raw: bytes) -> Recording:
    if len(raw) > MAX_BYTES:
        raise ValueError("Файл більший за 16 МіБ")
    value = json.loads(raw.decode("utf-8-sig"))
    # Accept a single message in the format used in the assignment.
    if isinstance(value, dict) and "fft_data" in value:
        value = {"frames": [value]}
    return Recording.model_validate(value)


def demo_recording() -> Recording:
    # Explicitly synthetic data. No SDR hardware or radio reception is involved.
    frames = []
    for frequency in (100.5, 101.2):
        for t in range(60):
            bins = [round(-88 + 3 * math.sin(i * 1.7 + t * .3)
                          + 47 * math.exp(-((i - 90 - 8 * math.sin(t / 12)) / 6) ** 2)
                          + 28 * math.exp(-((i - 175) / 10) ** 2), 2)
                    for i in range(256)]
            frames.append(Frame(frequency=frequency, fft_data=bins))
    return Recording(source_kind="demo", power_unit="дБ (синтетичні дані)",
                     sample_rate_hz=2400000, frames=frames)


class Player:
    def __init__(self, recording: Recording | None = None):
        self.recording = recording
        self.indices: list[int] = []
        self.frequency: float | None = None
        # Монотонний годинник задає позицію відтворення збережених FFT-кадрів.
        self.started = time.monotonic()
        self.revision = 0
        if recording:
            self.select(recording.frames[0].frequency)

    def select(self, frequency: float):
        indices = [i for i, f in enumerate(self.recording.frames)
                   if math.isclose(f.frequency, frequency, rel_tol=0, abs_tol=1e-6)]
        if not indices:
            raise HTTPException(409, "Цієї частоти немає у файлі. Виберіть частоту зі списку.")
        self.indices = indices
        self.frequency = self.recording.frames[indices[0]].frequency
        self.started = time.monotonic()
        self.revision += 1

    def status(self):
        rec = self.recording
        return {"loaded": bool(rec), "frequency": self.frequency,
                "frequencies": sorted({f.frequency for f in rec.frames}) if rec else [],
                "frames": len(rec.frames) if rec else 0,
                "selected_frames": len(self.indices),
                "source_kind": rec.source_kind if rec else None,
                "sample_interval_ms": rec.sample_interval_ms if rec else None,
                "revision": self.revision}

    def packet(self):
        if not self.recording:
            return {"type": "waiting", "message": "Завантажте JSON-файл або увімкніть демо"}
        rec = self.recording
        tick = int((time.monotonic() - self.started) * 1000 / rec.sample_interval_ms)
        position = tick % len(self.indices)
        frame = rec.frames[self.indices[position]]
        return {"type": "spectrum", "frequency": frame.frequency,
                "fft_data": frame.fft_data, "timestamp": frame.timestamp,
                "sample_rate_hz": rec.sample_rate_hz, "power_unit": rec.power_unit,
                "source_kind": rec.source_kind, "frame_index": position,
                "frame_count": len(self.indices), "loop": tick // len(self.indices),
                "revision": self.revision}


def create_app(data_dir: Path | None = None) -> FastAPI:
    from processing.fm_channel import PREVIEW_SECONDS, PREVIEW_VERSION, capture_info, channel_bounds, extract
    from web.jobs import Analyses

    folder = (data_dir or Path(os.environ.get("SDR_DATA_DIR", str(ROOT / "data")))).resolve()
    saved = folder / "recording.json"
    analyses = Analyses(folder, ROOT / 'prepare_recording.py')

    @asynccontextmanager
    async def lifespan(application):
        application.state.player = Player()
        application.state.upload_lock = asyncio.Lock()
        application.state.tune_lock = asyncio.Lock()
        if saved.exists():
            try:
                if saved.stat().st_size > MAX_BYTES:
                    raise ValueError("Файл більший за 16 МіБ")
                application.state.player = Player(parse_recording(saved.read_bytes()))
            except (OSError, ValueError, RecursionError) as exc:
                logging.warning("Не вдалося завантажити збережений файл: %s", exc)
        try:
            yield
        finally:
            await analyses.close()

    application = FastAPI(title="Перегляд записів SDR", lifespan=lifespan)
    application.include_router(analyses.routes)

    @application.get("/")
    async def ui():
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    @application.get("/radio.js", include_in_schema=False)
    async def radio_script():
        return FileResponse(STATIC_DIR / "radio.js", media_type="text/javascript", headers={"Cache-Control": "no-cache"})

    @application.get('/fm_report.js',include_in_schema=False)
    async def report_script():
        return FileResponse(STATIC_DIR/'fm_report.js',media_type='text/javascript',headers={'Cache-Control':'no-cache'})

    @application.get('/style.css', include_in_schema=False)
    async def stylesheet():
        return FileResponse(STATIC_DIR/'style.css',media_type='text/css',headers={'Cache-Control':'no-cache'})

    @application.get("/api/radio/demo", include_in_schema=False)
    async def radio_demo():
        return FileResponse(EXAMPLES_DIR / "example_demo.json", media_type="application/json")

    @application.get("/fm")
    async def fm_ui():
        return RedirectResponse('/#fmSection',headers={'Cache-Control':'no-cache'})

    def tuning_source(station):
        try:
            report = json.loads((folder / 'fm_report.json').read_bytes())
            name, center = report['filename'], report['center_mhz']
            if Path(name).name != name or name in ('.','..') or not center or not report['station_mhz']:
                raise ValueError()
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise HTTPException(404,'Поточний запис не містить I/Q для вибору станції.') from exc
        source = Path(os.environ.get('SDR_RECORDINGS_DIR', str(ROOT/'recordings'))) / name
        if not source.is_file():
            # Цей файл створює локальна команда підготовки, а не вебзапит.
            try:
                pin = json.loads((folder/'fm_source.json').read_bytes())
                candidate = Path(pin['path'])
                if candidate.name == name and candidate.is_absolute() and candidate.stat().st_size == pin['size']:
                    source = candidate
            except (ValueError, OSError, KeyError, TypeError):
                pass
        try:
            fs, n = capture_info(source)
        except (ValueError, OSError, wave.Error, EOFError) as exc:
            raise HTTPException(404,'Немає доступу до вихідного I/Q WAV на Raspberry Pi.') from exc
        low, high = channel_bounds(center,fs)
        if not low-1e-9 <= station <= high+1e-9:
            raise HTTPException(422,f'Цієї станції немає у доступній смузі. Обери від {low:.3f} до {high:.3f} МГц.')
        return source, {'station_mhz':station, 'center_mhz':center, 'report_station_mhz':report['station_mhz'],
                        'min_mhz':low, 'max_mhz':high, 'duration_s':n/fs, 'preview_seconds':PREVIEW_SECONDS}

    @application.get('/api/fm/station')
    async def station_info(station_mhz: Annotated[float,Query(gt=0,le=100000)]):
        _, info = await asyncio.to_thread(tuning_source,round(station_mhz,6))
        return info

    @application.get('/api/fm/station/audio')
    async def station_audio(station_mhz: Annotated[float,Query(gt=0,le=100000)],
                            start_s: Annotated[float,Query(ge=0,le=21600)]=0):
        source, info = await asyncio.to_thread(tuning_source,round(station_mhz,6))
        start = round(start_s,3)
        if start >= info['duration_s']-.001:
            raise HTTPException(422,'Час поза межами запису.')
        def prepare():
            stat = source.stat()
            stamp = (PREVIEW_VERSION, str(source.resolve()), stat.st_size, stat.st_mtime_ns,
                     info['center_mhz'], info['station_mhz'], start)
            key = hashlib.sha256(repr(stamp).encode()).hexdigest()[:24]
            cache = folder/'.station_previews'
            cache.mkdir(exist_ok=True)
            result = cache/f'{key}.wav'
            if not result.is_file():
                fd, filename = tempfile.mkstemp(prefix=f'{key}.',suffix='.tmp',dir=cache)
                os.close(fd)
                temp = Path(filename)
                try:
                    extract(source,temp,info['center_mhz'],info['station_mhz'],start)
                    temp.replace(result)
                    for old in sorted(cache.glob('*.wav'),key=lambda p:p.stat().st_mtime,reverse=True)[48:]:
                        old.unlink(missing_ok=True)
                finally:
                    temp.unlink(missing_ok=True)
            return result
        # Спільний кеш фрагментів заповнюється послідовно.
        async with application.state.tune_lock:
            try:
                path = await asyncio.to_thread(prepare)
            except (ValueError, OSError, wave.Error, ImportError, EOFError) as exc:
                logging.warning('Не вдалося декодувати станцію %.6f: %s',station_mhz,exc)
                raise HTTPException(422,'Не вдалося декодувати цей відрізок I/Q.') from exc
        return FileResponse(path,media_type='audio/wav',headers={'Cache-Control':'no-store'})

    @application.get("/api/fm/radio")
    async def fm_radio():
        radio = folder / "fm_radio.json"
        manifest = folder / "fm_radio_source.txt"
        def matches():
            try:
                return (radio.is_file() and manifest.read_text(encoding="utf-8").strip()
                        == json.loads((folder / "fm_report.json").read_text(encoding="utf-8"))["filename"])
            except (OSError, ValueError, KeyError):
                return False
        if not await asyncio.to_thread(matches):
            raise HTTPException(404, "Для поточного запису водоспад ще не підготовлено.")
        return FileResponse(radio, media_type="application/json", headers={"Cache-Control": "no-store"})

    @application.get("/api/fm/report")
    async def fm_report():
        report = folder / "fm_report.json"
        if not report.is_file():
            raise HTTPException(404, "Спочатку оброби WAV за інструкцією.")
        return FileResponse(report, media_type="application/json", headers={"Cache-Control": "no-store"})

    @application.get("/api/fm/audio")
    async def fm_audio():
        audio = folder / "fm_audio.wav"
        if not audio.is_file():
            raise HTTPException(404, "Звукову доріжку ще не підготовлено.")
        return FileResponse(audio, media_type="audio/wav", headers={"Cache-Control": "no-store"})

    @application.get("/api/fm/vocals")
    async def fm_vocals():
        vocals = folder / "fm_vocals.wav"
        if not vocals.is_file():
            raise HTTPException(404, "Доріжку виділеного голосу ще не підготовлено.")
        return FileResponse(vocals, media_type="audio/wav", headers={"Cache-Control": "no-store"})

    @application.get("/api/fm/clean")
    async def fm_clean():
        audio = folder / "fm_clean.wav"
        def available():
            try:
                report = json.loads((folder / "fm_report.json").read_text(encoding="utf-8"))
                return audio.is_file() and report.get("noise_reduction", {}).get("available")
            except (OSError, ValueError):
                return False
        if not await asyncio.to_thread(available):
            raise HTTPException(404, "Доріжку з приглушеним шумом ще не підготовлено.")
        return FileResponse(audio, media_type="audio/wav", headers={"Cache-Control": "no-store"})

    @application.get("/api/fm/neural")
    async def fm_neural():
        audio = folder / "fm_neural.wav"
        def available():
            try:
                report = json.loads((folder / "fm_report.json").read_text(encoding="utf-8"))
                return audio.is_file() and report.get("neural_audio", {}).get("available")
            except (OSError, ValueError):
                return False
        if not await asyncio.to_thread(available):
            raise HTTPException(404, "Нейродоріжку мовлення ще не підготовлено.")
        return FileResponse(audio, media_type="audio/wav", headers={"Cache-Control": "no-store"})

    @application.get("/api/fm/voice-focus")
    async def fm_voice_focus():
        audio = folder / "fm_voice_focus.wav"
        def available():
            try:
                report = json.loads((folder / "fm_report.json").read_text(encoding="utf-8"))
                return audio.is_file() and report.get("voice_focus", {}).get("available")
            except (OSError, ValueError):
                return False
        if not await asyncio.to_thread(available):
            raise HTTPException(404, "Доріжку мовлення над музикою ще не підготовлено.")
        return FileResponse(audio, media_type="audio/wav", headers={"Cache-Control": "no-store"})

    @application.get("/api/status")
    async def status():
        return application.state.player.status()

    @application.post("/api/data")
    async def upload(request: Request):
        # The browser sends the file itself as the request body, not multipart.
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > MAX_BYTES:
                raise HTTPException(413, "Файл більший за 16 МіБ")
        try:
            rec = await asyncio.to_thread(parse_recording, bytes(raw))
        except ValidationError as exc:
            error = exc.errors(include_input=False, include_context=False)[0]
            location = ".".join(map(str, error["loc"]))
            raise HTTPException(422, f"Некоректний формат у полі {location}") from exc
        except (ValueError, RecursionError) as exc:
            raise HTTPException(422, "Потрібен JSON у UTF-8 з масивами frames і fft_data") from exc
        async with application.state.upload_lock:
            try:
                folder.mkdir(parents=True, exist_ok=True)
                temp = folder / "recording.tmp"
                temp.write_bytes(raw)
                temp.replace(saved)  # Replace only after validation and successful write.
            except OSError as exc:
                raise HTTPException(500, "Не вдалося зберегти файл у папці data") from exc
            revision = application.state.player.revision
            application.state.player = Player(rec)
            application.state.player.revision = revision + 1
        return application.state.player.status()

    @application.post("/api/demo")
    async def demo():
        async with application.state.upload_lock:
            revision = application.state.player.revision
            application.state.player = Player(demo_recording())
            application.state.player.revision = revision + 1
        # Demo does not overwrite a saved real recording.
        return application.state.player.status()

    @application.post("/api/frequency")
    async def frequency(command: FrequencyCommand):
        player = application.state.player
        if not player.recording:
            raise HTTPException(409, "Спочатку завантажте файл або увімкніть демо")
        player.select(command.frequency_mhz)
        return player.status()

    @application.websocket("/ws/spectrum")
    async def spectrum(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                player = application.state.player
                await websocket.send_json(player.packet())
                interval = player.recording.sample_interval_ms / 1000 if player.recording else 1
                # Read disconnects while waiting; no orphan background tasks per tab.
                deadline = asyncio.get_running_loop().time() + interval
                while True:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        break
                    try:
                        await asyncio.wait_for(websocket.receive_text(), timeout=remaining)
                    except asyncio.TimeoutError:
                        break
        except (WebSocketDisconnect, asyncio.CancelledError):
            return

    return application


app = create_app()

