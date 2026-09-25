"""Fetch and stage the redistributable ASTAP CLI and its Gaia DR3 star indexes."""
from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import stat
import sys
import tempfile
import time
import urllib.request
import zipfile
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from pathlib import Path, PurePosixPath

ASTAP_VERSION = "2026.09.19"
SOURCES = {
    "windows": f"https://sourceforge.net/projects/astap-program/files/windows_installer/astap_command-line_version_win64.zip/download",
    "macos-x64": f"https://sourceforge.net/projects/astap-program/files/macOS%20installer/astap_command-line_version_macOS_x86_64.zip/download",
    "macos-arm64": f"https://sourceforge.net/projects/astap-program/files/macOS%20installer/astap_command-line_version_macOS_M1.zip/download",
    "d05": "https://sourceforge.net/projects/astap-program/files/star_databases/d05_star_database.zip/download",
    "g05": "https://sourceforge.net/projects/astap-program/files/star_databases/g05_star_database.zip/download",
    "w08": "https://sourceforge.net/projects/astap-program/files/star_databases/w08_star_database_mag08_astap.zip/download",
}
WINDOWS_CLI_SHA256 = "e9b94a44bd0f7b60e6e1beeb75951d955815445b965d04c4a61fc6f4c398a3ca"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, expected_sha256: str | None = None) -> str:
    if destination.is_file() and destination.stat().st_size > 1000:
        digest = file_sha256(destination)
        if expected_sha256 is None or digest == expected_sha256:
            return digest
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "StarSoftFocus-build/1"})
    temporary = destination.with_suffix(destination.suffix + ".partial")
    print(f"Downloading {destination.name} from {url}", flush=True)
    retryable_http_statuses = {408, 425, 429, 500, 502, 503, 504}
    maximum_attempts = 3
    for attempt in range(1, maximum_attempts + 1):
        temporary.unlink(missing_ok=True)
        digest = hashlib.sha256()
        try:
            with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
            file_hash = digest.hexdigest()
            if expected_sha256 and file_hash != expected_sha256:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(f"SHA-256 verification failed for {destination.name}")
            temporary.replace(destination)
            return file_hash
        except HTTPError as error:
            temporary.unlink(missing_ok=True)
            if error.code not in retryable_http_statuses or attempt >= maximum_attempts:
                raise
            failure: Exception = error
        except (URLError, TimeoutError, OSError, HTTPException) as error:
            temporary.unlink(missing_ok=True)
            if attempt >= maximum_attempts:
                raise
            failure = error
        delay = 2 * attempt
        print(
            f"Temporary download error for {destination.name} "
            f"({attempt}/{maximum_attempts}): {failure}; retrying in {delay}s.",
            flush=True,
        )
        time.sleep(delay)
    raise RuntimeError(f"Download attempts exhausted for {destination.name}")


def safe_members(archive: zipfile.ZipFile):
    for entry in archive.infolist():
        relative = PurePosixPath(entry.filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"Unsafe path in vendor archive: {entry.filename}")
        yield entry


def extract_cli(archive_path: Path, target: Path, executable_name: str) -> Path:
    staging = target / "_cli_extract"
    staging.mkdir(parents=True)
    with zipfile.ZipFile(archive_path) as archive:
        for entry in safe_members(archive):
            if entry.is_dir():
                continue
            source_name = Path(entry.filename).name
            destination = staging / source_name
            with archive.open(entry) as source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)
    executables = [path for path in staging.iterdir() if path.name.lower() in {"astap_cli", "astap_cli.exe"}]
    if not executables:
        raise RuntimeError("ASTAP command-line archive does not contain astap_cli")
    executable = executables[0]
    destination = target / executable_name
    shutil.copy2(executable, destination)
    for item in staging.iterdir():
        if item == executable or not item.is_file():
            continue
        shutil.copy2(item, target / item.name)
    shutil.rmtree(staging)
    if os.name != "nt":
        destination.chmod(destination.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return destination


def extract_catalog(archive_path: Path, target: Path) -> list[Path]:
    extracted: list[Path] = []
    with zipfile.ZipFile(archive_path) as archive:
        for entry in safe_members(archive):
            if entry.is_dir():
                continue
            filename = Path(entry.filename).name
            destination = target / filename
            if destination.exists():
                if destination.stat().st_size == entry.file_size:
                    continue
                raise RuntimeError(f"Conflicting catalog file in ASTAP archives: {filename}")
            with archive.open(entry) as source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)
            extracted.append(destination)
    return extracted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--platform", choices=("windows", "macos-x64", "macos-arm64"))
    args = parser.parse_args()
    system = args.platform or ("windows" if os.name == "nt" else None)
    if system is None:
        machine = platform.machine().lower()
        system = "macos-arm64" if machine in {"arm64", "aarch64"} else "macos-x64"
    executable_name = "astap_cli.exe" if system == "windows" else "astap_cli"
    cache = Path(tempfile.gettempdir()) / "starsoft-vendor-cache" / ASTAP_VERSION
    destination = args.output.resolve()
    if destination.exists():
        shutil.rmtree(destination)
    catalogs = destination / "catalogs"
    catalogs.mkdir(parents=True)
    manifest: list[str] = [f"ASTAP CLI {ASTAP_VERSION}"]
    cli_archive = cache / f"astap-{system}.zip"
    cli_hash = download(SOURCES[system], cli_archive,
                        WINDOWS_CLI_SHA256 if system == "windows" else None)
    manifest.append(f"ASTAP CLI SHA-256: {cli_hash}")
    extract_cli(cli_archive, destination, executable_name)

    for name in ("d05", "g05", "w08"):
        archive = cache / f"{name}.zip"
        archive_hash = download(SOURCES[name], archive)
        extracted = extract_catalog(archive, catalogs)
        if not extracted:
            raise RuntimeError(f"ASTAP {name.upper()} archive contained no catalog files")
        manifest.append(f"{name.upper()} Gaia DR3 index archive SHA-256: {archive_hash}")
        print(f"Staged ASTAP {name.upper()} ({len(extracted)} files)", flush=True)

    acknowledgement = next(catalogs.glob("acknowledgement of databases.txt"), None)
    if acknowledgement is None:
        raise RuntimeError("ASTAP Gaia database acknowledgement is missing from catalog archives")
    (destination / "ASTAP_GAIA_DR3_NOTICE.txt").write_text(
        "星点柔焦随包提供 ASTAP 命令行板解算器与 ASTAP Gaia DR3 星表索引。\n"
        "ASTAP solver is distributed under the Mozilla Public License 2.0.\n"
        "The catalog indices are derived from Gaia DR3. Please acknowledge ESA/Gaia/DPAC.\n"
        "See acknowledgement of databases.txt and ASTAP-MPL-2.0.txt for full notices.\n",
        encoding="utf-8",
    )
    shutil.copy2(acknowledgement, destination / acknowledgement.name)
    repository_root = Path(__file__).resolve().parent.parent
    shutil.copy2(repository_root / "licenses" / "ASTAP-MPL-2.0.txt",
                 destination / "ASTAP-MPL-2.0.txt")
    (destination / "ASTAP-SOURCE.txt").write_text(
        "ASTAP CLI source code: https://github.com/CanardConfit/ASTAP\n"
        "ASTAP CLI license: Mozilla Public License 2.0 (see ASTAP-MPL-2.0.txt).\n",
        encoding="utf-8",
    )
    manifest.append(f"Staged executable: {executable_name}")
    manifest.append(f"Catalog files: {len(list(catalogs.iterdir()))}")
    (destination / "SOLVER_VERSION.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(f"Solver payload ready: {destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
