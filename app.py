"""Local browser interface for the standalone Star Soft Focus executable."""

from __future__ import annotations

import atexit
import io
import json
import os
import secrets
import shutil
import sys
import tempfile
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit

from PIL import Image

from processor import RawInfo, process_raw
from version import APP_VERSION


APP_NAME = "星点柔焦"
PAGES_ORIGIN = "https://lzq1206.github.io"
MAX_UPLOAD = 1_500_000_000
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
TEMP_DIRS: list[str] = []


def _asset_path() -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / "ui" / "index.html"


def _remove_temps() -> None:
    for path in TEMP_DIRS:
        shutil.rmtree(path, ignore_errors=True)


atexit.register(_remove_temps)


def _safe_filename(name: str) -> str:
    name = Path(name.replace("\\", "/")).name
    name = "".join(char for char in name if char.isprintable() and char not in '<>:"|?*')
    return name[:180] or "night_sky.raw"


def _update_job(job_id: str, **values: object) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is not None:
            job.update(values)


def _run_job(job_id: str, input_path: Path, output_path: Path, params: dict[str, float]) -> None:
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
                threading.Thread(target=self.server.shutdown, daemon=True).start()
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
                    "relative_magnitude_limit": float(query.get("relative_magnitude_limit", ["3.0"])[0]),
                    "min_radius": float(query.get("min_radius", ["3"])[0]),
                    "max_radius": float(query.get("max_radius", ["42"])[0]),
                }
            except ValueError:
                self._json(400, {"error": "柔焦参数无效"})
                return
            params["sensitivity"] = min(max(params["sensitivity"], 2.0), 10.0)
            params["strength"] = min(max(params["strength"], 0.0), 30.0)
            params["opacity"] = min(max(params["opacity"], 0.0), 100.0)
            params["relative_magnitude_limit"] = min(max(params["relative_magnitude_limit"], 0.0), 10.0)
            params["min_radius"] = min(max(params["min_radius"], 2.0), 24.0)
            params["max_radius"] = min(max(params["max_radius"], 8.0), 80.0)
            if params["max_radius"] < params["min_radius"]:
                params["min_radius"], params["max_radius"] = params["max_radius"], params["min_radius"]

            temp_dir = tempfile.mkdtemp(prefix="starsoft_")
            TEMP_DIRS.append(temp_dir)
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
                shutil.rmtree(temp_dir, ignore_errors=True)
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
                }
            worker = threading.Thread(target=_run_job, args=(job_id, input_path, output_path, params), daemon=True)
            worker.start()
            self._json(202, {"id": job_id})

    return Handler


def main() -> None:
    token = secrets.token_urlsafe(24)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(token))
    server.daemon_threads = True
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}/{token}/"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    opened = webbrowser.open_new(url)
    if not opened and os.name == "nt":
        import ctypes

        ctypes.windll.user32.MessageBoxW(0, f"无法自动打开浏览器，请复制此地址：\n{url}", APP_NAME, 0x40)
    try:
        thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if os.name == "nt":
            import ctypes

            ctypes.windll.user32.MessageBoxW(0, f"程序启动失败：\n{exc}", APP_NAME, 0x10)
