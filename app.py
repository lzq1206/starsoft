"""Local browser interface for the standalone Star Soft Focus executable."""

from __future__ import annotations

import atexit
import io
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit


def _startup_log_path() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / "StarSoftFocus" / "startup.log"
    return Path(tempfile.gettempdir()) / "StarSoftFocus-startup.log"


def _write_startup_log(message: str) -> Path | None:
    try:
        path = _startup_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(message.rstrip() + "\n", encoding="utf-8")
        return path
    except OSError:
        return None


def _show_startup_message(message: str, title: str = "星点柔焦启动失败") -> None:
    if sys.platform == "darwin":
        escaped_title = title.replace("\\", "\\\\").replace('"', '\\"')
        escaped_message = message.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        script = f'display dialog "{escaped_message}" with title "{escaped_title}" buttons {{"好"}} default button "好"'
        try:
            subprocess.run(
                ["/usr/bin/osascript", "-e", script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=20,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
    elif os.name == "nt":
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(0, message, title, 0x10)
        except Exception:
            pass


def _report_startup_error(details: str) -> None:
    log_path = _write_startup_log(details)
    message = details.strip().splitlines()[-1] if details.strip() else "未知启动错误"
    if log_path:
        message += f"\n\n详细日志：{log_path}"
    _show_startup_message(message)


try:
    from PIL import Image

    from processor import RawInfo, process_raw
    from version import APP_VERSION
except Exception:
    _report_startup_error(traceback.format_exc())
    raise


APP_NAME = "星点柔焦"
PAGES_ORIGIN = "https://lzq1206.github.io"
MAX_UPLOAD = 1_500_000_000
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
TEMP_DIRS: set[str] = set()
TEMP_DIRS_LOCK = threading.Lock()
ACTIVE_CLIENTS: dict[str, float] = {}
ACTIVE_CLIENTS_LOCK = threading.Lock()
ACTIVE_TRANSFERS = 0
ACTIVE_TRANSFERS_LOCK = threading.Lock()
CLIENT_CONNECTED = threading.Event()
SERVER_STOPPING = threading.Event()
SERVER_STOPPING_LOCK = threading.Lock()
CLIENT_STALE_SECONDS = 120
CLIENT_CLOSE_GRACE_SECONDS = 4
STARTUP_IDLE_SECONDS = 300
JOB_RETENTION_SECONDS = 30 * 60
DOWNLOADED_JOB_RETENTION_SECONDS = 10 * 60
MAX_RETAINED_COMPLETED_JOBS = 3
STALE_TEMP_MAX_AGE_SECONDS = 24 * 60 * 60


def _asset_path() -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / "ui" / "index.html"


def _register_temp_dir(path: str) -> None:
    with TEMP_DIRS_LOCK:
        TEMP_DIRS.add(path)


def _remove_temp_dir(path: str | None) -> None:
    if not path:
        return
    with TEMP_DIRS_LOCK:
        TEMP_DIRS.discard(path)
    shutil.rmtree(path, ignore_errors=True)


def _remove_temps() -> None:
    with TEMP_DIRS_LOCK:
        paths = tuple(TEMP_DIRS)
        TEMP_DIRS.clear()
    for path in paths:
        shutil.rmtree(path, ignore_errors=True)


def _cleanup_stale_temp_dirs() -> None:
    try:
        temp_root = Path(tempfile.gettempdir()).resolve()
        candidates = tuple(temp_root.iterdir())
    except OSError:
        return
    cutoff = time.time() - STALE_TEMP_MAX_AGE_SECONDS
    with TEMP_DIRS_LOCK:
        active_paths = {Path(path).resolve() for path in TEMP_DIRS}
    for candidate in candidates:
        if not candidate.name.startswith(("starsoft_", "starsoft-build-", "starsoft-solve-", "starsoft-solver-image-")):
            continue
        try:
            if candidate.is_symlink() or not candidate.is_dir():
                continue
            resolved = candidate.resolve()
            if resolved.parent != temp_root or resolved in active_paths or candidate.stat().st_mtime >= cutoff:
                continue
            shutil.rmtree(candidate, ignore_errors=True)
        except OSError:
            continue


atexit.register(_remove_temps)


def _open_browser(url: str) -> bool:
    if sys.platform == "darwin":
        try:
            opened = subprocess.run(
                ["/usr/bin/open", url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=15,
                check=False,
            )
            if opened.returncode == 0:
                return True
            detail = opened.stderr.strip() or f"open exited with code {opened.returncode}"
            _write_startup_log(f"macOS 'open' failed: {detail}")
        except (OSError, subprocess.TimeoutExpired) as exc:
            _write_startup_log(f"macOS 'open' failed: {exc}")
    try:
        opened = bool(webbrowser.open_new(url))
        if not opened:
            _write_startup_log("macOS and Python browser launch methods both failed.")
        return opened
    except Exception as exc:
        _write_startup_log(f"Python webbrowser failed: {exc}")
        return False


def _safe_filename(name: str) -> str:
    name = Path(name.replace("\\", "/")).name
    name = "".join(char for char in name if char.isprintable() and char not in '<>:"|?*')
    return name[:180] or "night_sky.raw"


def _update_job(job_id: str, **values: object) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is not None:
            if values.get("state") in {"complete", "error"} and job.get("state") == "processing":
                job["finishedAt"] = time.time()
            job.update(values)


def _release_job_directory(job_id: str, *, forget_job: bool = False) -> None:
    temp_dir: str | None = None
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is not None:
            temp_dir = job.pop("tempDir", None)
            if forget_job:
                JOBS.pop(job_id, None)
            else:
                job["tempDir"] = None
                job["outputPath"] = None
    _remove_temp_dir(temp_dir)


def _prune_jobs(now: float | None = None) -> None:
    current_time = time.time() if now is None else now
    remove_ids: set[str] = set()
    with JOBS_LOCK:
        completed: list[tuple[float, str]] = []
        for job_id, job in JOBS.items():
            state = job.get("state")
            if state == "processing":
                continue
            finished_at = float(job.get("finishedAt", job.get("createdAt", current_time)))
            if state == "complete":
                downloaded_at = job.get("downloadedAt")
                reference_time = float(downloaded_at if downloaded_at is not None else finished_at)
                retention = DOWNLOADED_JOB_RETENTION_SECONDS if downloaded_at is not None else JOB_RETENTION_SECONDS
                if current_time - reference_time >= retention:
                    remove_ids.add(job_id)
                else:
                    completed.append((finished_at, job_id))
            elif current_time - finished_at >= JOB_RETENTION_SECONDS:
                remove_ids.add(job_id)

        retained = sorted((item for item in completed if item[1] not in remove_ids))
        for _finished_at, job_id in retained[:-MAX_RETAINED_COMPLETED_JOBS]:
            remove_ids.add(job_id)

        temp_dirs: list[str] = []
        for job_id in remove_ids:
            job = JOBS.pop(job_id, None)
            if job is not None and job.get("tempDir"):
                temp_dirs.append(job["tempDir"])
    for temp_dir in temp_dirs:
        _remove_temp_dir(temp_dir)


def _has_processing_jobs() -> bool:
    with JOBS_LOCK:
        return any(job.get("state") == "processing" for job in JOBS.values())


def _begin_transfer() -> None:
    global ACTIVE_TRANSFERS
    with ACTIVE_TRANSFERS_LOCK:
        ACTIVE_TRANSFERS += 1


def _end_transfer() -> None:
    global ACTIVE_TRANSFERS
    with ACTIVE_TRANSFERS_LOCK:
        ACTIVE_TRANSFERS = max(0, ACTIVE_TRANSFERS - 1)


def _has_active_work() -> bool:
    if _has_processing_jobs():
        return True
    with ACTIVE_TRANSFERS_LOCK:
        return ACTIVE_TRANSFERS > 0


def _expire_stale_clients() -> bool:
    now = time.monotonic()
    with ACTIVE_CLIENTS_LOCK:
        stale = [client_id for client_id, last_seen in ACTIVE_CLIENTS.items() if now - last_seen >= CLIENT_STALE_SECONDS]
        for client_id in stale:
            ACTIVE_CLIENTS.pop(client_id, None)
        return bool(ACTIVE_CLIENTS)


def _request_shutdown(server: ThreadingHTTPServer) -> None:
    with SERVER_STOPPING_LOCK:
        if SERVER_STOPPING.is_set():
            return
        SERVER_STOPPING.set()
    threading.Thread(target=server.shutdown, daemon=True).start()


def _shutdown_after_page_close(server: ThreadingHTTPServer) -> None:
    if SERVER_STOPPING.wait(CLIENT_CLOSE_GRACE_SECONDS):
        return
    _expire_stale_clients()
    with ACTIVE_CLIENTS_LOCK:
        clients_open = bool(ACTIVE_CLIENTS)
    if not clients_open and CLIENT_CONNECTED.is_set() and not _has_active_work():
        _request_shutdown(server)


def _watch_lifecycle(server: ThreadingHTTPServer, started_at: float) -> None:
    while not SERVER_STOPPING.wait(15):
        _expire_stale_clients()
        _prune_jobs()
        with ACTIVE_CLIENTS_LOCK:
            clients_open = bool(ACTIVE_CLIENTS)
        if not clients_open and not _has_active_work():
            if CLIENT_CONNECTED.is_set() or time.monotonic() - started_at >= STARTUP_IDLE_SECONDS:
                _request_shutdown(server)
                return


def _run_job(job_id: str, input_path: Path, output_path: Path, params: dict[str, float | str]) -> None:
    def report(percent: int, message: str) -> None:
        _update_job(job_id, percent=10 + int(percent * 0.9), message=message)

    try:
        def metadata(info: RawInfo) -> None:
            _update_job(
                job_id,
                metadata={
                    "camera": info.camera,
                    "lens": info.lens,
                    "focalLength": info.focal_length,
                    "aperture": info.aperture,
                    "width": info.width,
                    "height": info.height,
                },
            )

        result = process_raw(input_path, output_path, **params, progress=report, metadata_callback=metadata)
        _update_job(job_id, message="正在准备预览…", percent=98)
        with Image.open(output_path) as image:
            image.thumbnail((1600, 1100), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            preview_options = {"format": "JPEG", "quality": 91, "optimize": True}
            if image.mode in {"RGB", "RGBA"} and image.info.get("icc_profile"):
                preview_options["icc_profile"] = image.info["icc_profile"]
            image.convert("RGB").save(buffer, **preview_options)
            preview = buffer.getvalue()
        _update_job(
            job_id,
            state="complete",
            percent=100,
            message="柔焦完成",
            stars=result.star_count,
            candidates=result.candidate_count,
            catalogMatches=result.catalog_match_count,
            recoveredCatalogStars=result.recovered_catalog_star_count,
            brightnessSource=result.brightness_source,
            selectedCount=result.selected_count,
            relativeMagnitudeLimit=round(result.relative_magnitude_limit, 1),
            width=result.width,
            height=result.height,
            inputKind=result.input_kind,
            colorProfile=result.color_profile,
            skyBackgroundLevel=result.sky_background_level,
            skyAdaptationGain=result.sky_adaptation_gain,
            outputName=Path(result.output_path).name,
            preview=preview,
        )
    except Exception as exc:
        _update_job(job_id, state="error", message=str(exc), percent=0)
    finally:
        try:
            input_path.unlink(missing_ok=True)
        except OSError:
            pass
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            failed = job is None or job.get("state") == "error"
        if failed:
            _release_job_directory(job_id)
        _prune_jobs()


def _make_handler(token: str):
    base = f"/{token}"

    class Handler(BaseHTTPRequestHandler):
        server_version = f"StarSoftFocus/{APP_VERSION}"
        sys_version = ""

        def log_message(self, _format: str, *_args: object) -> None:
            return

        def _send(self, status: int, payload: bytes, content_type: str, extra: dict[str, str] | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob: data:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
            origin = self.headers.get("Origin")
            if origin == PAGES_ORIGIN:
                self.send_header("Access-Control-Allow-Origin", PAGES_ORIGIN)
                self.send_header("Access-Control-Allow-Private-Network", "true")
                self.send_header("Vary", "Origin")
            if extra:
                for key, value in extra.items():
                    self.send_header(key, value)
            self.end_headers()
            self.wfile.write(payload)

        def _json(self, status: int, values: dict) -> None:
            self._send(status, json.dumps(values, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _authorized_path(self) -> str | None:
            path = urlsplit(self.path).path
            if path == base:
                return "/"
            if path.startswith(base + "/"):
                return path[len(base) :]
            return None

        def do_GET(self) -> None:
            route = self._authorized_path()
            if route is None:
                self._json(404, {"error": "not found"})
                return
            _prune_jobs()
            if route in ("/", ""):
                try:
                    html = (
                        _asset_path().read_text(encoding="utf-8")
                        .replace("__APP_BASE__", base)
                        .replace("__APP_VERSION__", APP_VERSION)
                    )
                except OSError as exc:
                    self._json(500, {"error": f"无法读取程序界面资源：{exc}"})
                    return
                self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
                return
            parts = route.strip("/").split("/")
            if parts == ["api", "health"]:
                self._json(200, {"ok": True, "version": APP_VERSION})
                return
            if len(parts) == 3 and parts[:2] == ["api", "jobs"]:
                job_id = parts[2]
                with JOBS_LOCK:
                    job = JOBS.get(job_id)
                    if not job:
                        self._json(404, {"error": "任务不存在"})
                        return
                    summary = {key: value for key, value in job.items() if key not in {"preview", "outputPath", "tempDir"}}
                self._json(200, summary)
                return
            if len(parts) == 4 and parts[:2] == ["api", "jobs"]:
                job_id, action = parts[2], parts[3]
                with JOBS_LOCK:
                    job = JOBS.get(job_id)
                    if not job:
                        self._json(404, {"error": "任务不存在"})
                        return
                    if action == "preview":
                        preview = job.get("preview")
                        if not preview:
                            self._json(404, {"error": "预览尚未准备好"})
                            return
                        self._send(200, preview, "image/jpeg")
                        return
                    if action == "download":
                        output_path = Path(job["outputPath"])
                        output_name = str(job["outputName"])
                if action == "download" and output_path.is_file():
                    _begin_transfer()
                    try:
                        size = output_path.stat().st_size
                        self.send_response(200)
                        self.send_header("Content-Type", "image/tiff")
                        self.send_header("Content-Length", str(size))
                        self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{quote(output_name)}")
                        self.send_header("Cache-Control", "no-store")
                        self.send_header("X-Content-Type-Options", "nosniff")
                        if self.headers.get("Origin") == PAGES_ORIGIN:
                            self.send_header("Access-Control-Allow-Origin", PAGES_ORIGIN)
                            self.send_header("Access-Control-Allow-Private-Network", "true")
                            self.send_header("Vary", "Origin")
                        self.end_headers()
                        with output_path.open("rb") as handle:
                            shutil.copyfileobj(handle, self.wfile, length=1024 * 1024)
                    except OSError:
                        return
                    finally:
                        _end_transfer()
                    with JOBS_LOCK:
                        job = JOBS.get(job_id)
                        if job is not None:
                            job["downloadedAt"] = time.time()
                    return
            self._json(404, {"error": "not found"})

        def do_OPTIONS(self) -> None:
            route = self._authorized_path()
            if route is None or self.headers.get("Origin") != PAGES_ORIGIN:
                self._json(403, {"error": "origin not allowed"})
                return
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", PAGES_ORIGIN)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-File-Name")
            self.send_header("Access-Control-Allow-Private-Network", "true")
            self.send_header("Access-Control-Max-Age", "600")
            self.send_header("Vary", "Origin")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self) -> None:
            route = self._authorized_path()
            if route is None:
                self._json(404, {"error": "not found"})
                return
            if route == "/api/shutdown":
                self._json(200, {"ok": True})
                _request_shutdown(self.server)
                return
            if route in {"/api/client/heartbeat", "/api/client/close"}:
                try:
                    content_length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    self._json(400, {"error": "客户端标识无效"})
                    return
                if content_length <= 0 or content_length > 256:
                    self._json(400, {"error": "客户端标识无效"})
                    return
                try:
                    body = self.rfile.read(content_length).decode("utf-8").strip()
                    if self.headers.get_content_type() == "application/x-www-form-urlencoded":
                        client_id = parse_qs(body).get("clientId", [""])[0]
                    else:
                        client_id = body
                except (UnicodeDecodeError, ValueError):
                    self._json(400, {"error": "客户端标识无效"})
                    return
                if not client_id or len(client_id) > 64 or not all(char.isalnum() or char in "-_" for char in client_id):
                    self._json(400, {"error": "客户端标识无效"})
                    return
                if route.endswith("heartbeat"):
                    with ACTIVE_CLIENTS_LOCK:
                        ACTIVE_CLIENTS[client_id] = time.monotonic()
                    CLIENT_CONNECTED.set()
                    self._send(204, b"", "text/plain; charset=utf-8")
                    return
                with ACTIVE_CLIENTS_LOCK:
                    ACTIVE_CLIENTS.pop(client_id, None)
                    no_clients = not ACTIVE_CLIENTS
                self._send(204, b"", "text/plain; charset=utf-8")
                if no_clients:
                    threading.Thread(target=_shutdown_after_page_close, args=(self.server,), daemon=True).start()
                return
            if route != "/api/jobs":
                self._json(404, {"error": "not found"})
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._json(400, {"error": "上传长度无效"})
                return
            if content_length <= 0 or content_length > MAX_UPLOAD:
                self._json(413, {"error": "输入文件为空或超过 1.5 GB 限制"})
                return

            name = _safe_filename(unquote(self.headers.get("X-File-Name", "night_sky.raw")))
            supported_extensions = {
                ".cr3", ".cr2", ".crw", ".nef", ".nrw", ".arw", ".sr2", ".srf", ".dng",
                ".orf", ".rw2", ".raf", ".pef", ".ptx", ".3fr", ".fff", ".iiq", ".kdc",
                ".dcr", ".mos", ".mrw", ".x3f", ".tif", ".tiff", ".jpg", ".jpeg",
            }
            if Path(name).suffix.lower() not in supported_extensions:
                self._json(415, {"error": "请选择相机 RAW、TIFF 或 JPG 文件。"})
                return
            query = parse_qs(urlsplit(self.path).query)
            try:
                params = {
                    "sensitivity": float(query.get("sensitivity", ["4.8"])[0]),
                    "strength": float(query.get("strength", ["10"])[0]),
                    "opacity": float(query.get("opacity", ["30"])[0]),
                    "brightness_source": str(query.get("brightness_source", ["catalog"])[0]),
                    "relative_magnitude_limit": float(query.get("relative_magnitude_limit", ["5.0"])[0]),
                    "min_radius": float(query.get("min_radius", ["3"])[0]),
                    "max_radius": float(query.get("max_radius", ["42"])[0]),
                }
            except ValueError:
                self._json(400, {"error": "柔焦参数无效"})
                return
            params["sensitivity"] = min(max(params["sensitivity"], 2.0), 10.0)
            params["strength"] = min(max(params["strength"], 0.0), 30.0)
            params["opacity"] = min(max(params["opacity"], 0.0), 100.0)
            if params["brightness_source"] not in {"catalog", "image"}:
                raise ValueError("星点亮度来源无效")
            params["relative_magnitude_limit"] = min(max(params["relative_magnitude_limit"], 0.0), 10.0)
            params["min_radius"] = min(max(params["min_radius"], 2.0), 24.0)
            params["max_radius"] = min(max(params["max_radius"], 8.0), 80.0)
            if params["max_radius"] < params["min_radius"]:
                params["min_radius"], params["max_radius"] = params["max_radius"], params["min_radius"]

            _prune_jobs()
            _begin_transfer()
            temp_dir: str | None = None
            try:
                temp_dir = tempfile.mkdtemp(prefix="starsoft_")
                _register_temp_dir(temp_dir)
                input_path = Path(temp_dir) / name
                output_path = Path(temp_dir) / f"{Path(name).stem}_星点柔焦.tif"
                remaining = content_length
                try:
                    with input_path.open("wb") as handle:
                        while remaining:
                            chunk = self.rfile.read(min(1024 * 1024, remaining))
                            if not chunk:
                                raise ConnectionError("文件上传未完成")
                            handle.write(chunk)
                            remaining -= len(chunk)
                except (OSError, ConnectionError) as exc:
                    _remove_temp_dir(temp_dir)
                    temp_dir = None
                    self._json(400, {"error": f"读取上传文件失败：{exc}"})
                    return

                job_id = secrets.token_urlsafe(18)
                with JOBS_LOCK:
                    JOBS[job_id] = {
                        "id": job_id,
                        "state": "processing",
                        "percent": 0,
                        "message": "文件已接收，正在启动识别…",
                        "sourceName": name,
                        "outputName": output_path.name,
                        "outputPath": str(output_path),
                        "tempDir": temp_dir,
                        "createdAt": time.time(),
                    }
                worker = threading.Thread(target=_run_job, args=(job_id, input_path, output_path, params), daemon=True)
                try:
                    worker.start()
                except RuntimeError as exc:
                    _update_job(job_id, state="error", message=f"无法启动处理任务：{exc}", percent=0)
                    _release_job_directory(job_id)
                    self._json(500, {"error": "无法启动图像处理任务"})
                    return
            except Exception:
                if temp_dir is not None:
                    with JOBS_LOCK:
                        registered_job = any(job.get("tempDir") == temp_dir for job in JOBS.values())
                    if not registered_job:
                        _remove_temp_dir(temp_dir)
                raise
            finally:
                _end_transfer()
            self._json(202, {"id": job_id})

    return Handler


def main() -> None:
    CLIENT_CONNECTED.clear()
    SERVER_STOPPING.clear()
    with ACTIVE_CLIENTS_LOCK:
        ACTIVE_CLIENTS.clear()
    threading.Thread(target=_cleanup_stale_temp_dirs, daemon=True).start()
    token = secrets.token_urlsafe(24)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(token))
    server.daemon_threads = True
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}/{token}/"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    watcher = threading.Thread(target=_watch_lifecycle, args=(server, time.monotonic()), daemon=True)
    watcher.start()
    try:
        if not _open_browser(url):
            log_path = _startup_log_path()
            _show_startup_message(
                f"无法自动打开浏览器。请在浏览器中打开以下本机地址：\n{url}"
                + (f"\n\n启动日志：{log_path}" if log_path else ""),
                title="星点柔焦：浏览器未打开",
            )
        thread.join()
    except KeyboardInterrupt:
        return
    finally:
        if thread.is_alive():
            server.shutdown()
        server.server_close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        _report_startup_error(traceback.format_exc())
        raise SystemExit(1)
