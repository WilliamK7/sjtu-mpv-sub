# AGENTS.md — sjtu-mpv-sub

Guidance for AI coding agents working inside `D:\Program Files\sjtu-mpv-sub` (Python 3.12 via `uv`, Windows).

## 1. Architecture & File Map

Lightweight streaming lecture player for SJTU 课堂实录 (`https://v.sjtu.edu.cn/jy-application-resourcemanage-ui/`) built on zero video download + local `mpv`:

- `src/sjtu_mpv_sub/cli.py`: CLI entrypoints (`check`, `serve --ephemeral`).
- `src/sjtu_mpv_sub/config.py`: `AppConfig` loader (`config.yaml`), executable lookup (`mpv`, `ffmpeg`, `ffprobe`), thread-safe R/W for `courses.yaml`, Bailian workspace endpoint derivation.
- `src/sjtu_mpv_sub/models.py`: Pydantic v2 schemas (`ViewMode`, `JobState`, `PlayRequest`, `SegmentInput`, `CoursePreference`, `AsrSentence`, `BilingualCue`, `JobStatus`).
- `src/sjtu_mpv_sub/media.py`: `ffprobe` stream inspection, `viewNum` track resolution, silent-stream audio fallback, `ffmpeg` HTTP-206 `-ss`-before-`-i` 16kHz mono 48kbps MP3 extraction.
- `src/sjtu_mpv_sub/mpv_controller.py`: `mpv` EDL builder/writer, process launcher, Windows Named Pipe JSON IPC (`MpvIpcClient`).
- `src/sjtu_mpv_sub/asr.py`: Pure `httpx` RESTful client (`QwenAudioFiletransASR`) for Bailian temp OSS (`GET /uploads?action=getPolicy` -> PostObject -> `oss://...`) + async filetrans.
- `src/sjtu_mpv_sub/translator.py`: Time-windowed PPT OCR collector + `LLMTranslator` (OpenAI-compatible `AsyncOpenAI` over `httpx`).
- `src/sjtu_mpv_sub/subtitle.py`: Styled ASS renderer (`render_ass`) and atomic writer (`write_ass_atomic`).
- `src/sjtu_mpv_sub/pipeline.py`: `LecturePipeline` — Priority Chunk 0 -> `mpv` launch -> parallel chunks -> live `sub-reload` -> `mpv` exit watcher -> session cache (`bilingual.ass`, `cues.json`, `meta.json`, `session.edl`).
- `src/sjtu_mpv_sub/server.py`: Ephemeral FastAPI app on `127.0.0.1:18732` with 300s idle auto-exit and `data/logs/server.log` rotating logs.
- `userscript/sjtu-mpv-bilingual.user.js`: Tampermonkey panel, XHR/fetch interceptor, and `sjtu-mpv://wake` trigger. Note: edits to this file require the user to re-paste/update it in Tampermonkey.
- `wake.vbs`, `install_protocol.ps1`, `uninstall_protocol.ps1`: `HKCU:\Software\Classes\sjtu-mpv` protocol handler launching windowless `.venv\Scripts\pythonw.exe`.
- `tests/`: Unit tests and real `ffmpeg`/`mpv.exe` integration tests.

## 2. Everyday Commands

```powershell
# Run full test suite (includes real ffmpeg & mpv IPC test)
uv run pytest -v

# Check local tools (mpv, ffmpeg, ffprobe), sjtu-mpv:// registry, and API key config
uv run sjtu-mpv-sub check

# Syntax-check the Tampermonkey userscript
& 'D:\Program Files\nodejs\node.exe' --check userscript\sjtu-mpv-bilingual.user.js

# Check or gracefully stop an active companion server before live re-tests
Invoke-RestMethod -Uri 'http://127.0.0.1:18732/health' -TimeoutSec 2
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:18732/api/shutdown' -TimeoutSec 2
```

## 3. Protected & Ignored Files

- **NEVER commit or destructively overwrite `config.yaml`** (holds the user's real DashScope `api_key`, `workspace_id`, and custom subtitle font/order preferences) or `courses.yaml` (user's saved views and course glossaries).
- Both files plus `.venv/`, `.pytest_cache/`, and `data/` are listed in `.gitignore`. Keep new config schema fields documented in `config.example.yaml` instead.

## 4. Verified Platform & API Facts (Do Not Regress)

### SJTU 课堂实录 (`v.sjtu.edu.cn`)
- **`viewNum` mapping (verified live on DOM + video frame)**: `viewNum: 1` is the **classroom camera (`camera`)**; `viewNum: 5` is the **PC/PPT screen capture (`ppt`)**. Keep `target_view_num = 5 if preferred_view == "ppt" else 1` in `media.py`.
- Direct MP4 URLs on `live.sjtu.edu.cn:443` require `Referer: https://v.sjtu.edu.cn/jy-application-resourcemanage-ui/` and support HTTP 206 Range seeking.
- **Do NOT integrate SJTU's `/v1/course/ai/translate/<courseId>`**: live evaluation showed severe English ASR corruption, missing translations, and unusable block lengths; keep a single clean Bailian ASR path.
- SJTU recordings begin ~5 minutes before class starts, so Chunk 0 (`0–180s`) and break chunks are often silent room audio.

### Aliyun Bailian (Model Studio)
- **Single ASR backend**: `qwen-audio-3.1-asr-flash-filetrans` (`X-DashScope-Async: enable`, `X-DashScope-OssResourceResolve: enable` for `oss://...`). `parameters` is mandatory (`channel_id: [0]`, `language_hints` <= 4 items, instant `vocabulary` <= 2000 items with non-ASCII <= 15 chars / ASCII <= 7 words, `input.context` <= 400 chars).
- Never forward the DashScope `Authorization` header when downloading signed OSS `transcription_url` results.
- **`ASR_RESPONSE_HAVE_NO_WORDS`** (top-level or subtask-level) means silent/no-speech chunk (unbilled): `asr.py` must log at `INFO` and return `[]` immediately with **0 retries**.
- **LLM translator**: `qwen3.8-flash` defaults to thinking enabled on Bailian; keep `extra_body={"enable_thinking": self.enable_thinking}` with `llm_enable_thinking: false` default.

## 5. Critical Windows & `mpv-lazy` Invariants

1. **Windows `NO_PROXY` IPv6 bracket crash**: Host `NO_PROXY` includes `[::1]`, which triggers `httpx.InvalidURL: Invalid port: ':1]'` under `trust_env=True`. Always set `trust_env=False` on every `httpx.Client` / `httpx.AsyncClient` instance (never rely on `dashscope` SDK).
2. **`mpv` EDL line endings & chapters**: `.edl` files MUST use strict Unix LF (`\n`), never Windows CRLF (`\r\n`). Omit `!no_chapters` on `session.edl` so per-period `title=` chapters work; include `!no_chapters` on `audio.edl`.
3. **`mpv-lazy` portable defaults**: `D:\Program Files\mpv-lazy` enables `sub-codepage = GB18030` and `autocreate-playlist = same`. Keep UTF-8 BOM (`utf-8-sig`) in `write_ass_atomic`, pass `--sub-codepage=utf8 --autocreate-playlist=no`, and wait for non-empty `track-list` in `MpvIpcClient.wait_ready()` before calling `sub-reload`.
4. **Background lifecycle & idle auto-exit**: With `pipeline.stop_on_mpv_exit: true`, closing `mpv` cancels remaining background chunks (`state = "canceled"`) and skips writing `meta.json`. Terminal `GET /api/jobs/{job_id}` polling skips `runtime.touch()` so lingering browser tabs never block `idle_exit_seconds`.
