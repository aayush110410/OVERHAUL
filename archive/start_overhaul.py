"""Convenience launcher for the OVERHAUL stack.

Running this script spins up the FastAPI backend (uvicorn) and a lightweight
static server for the OVERHUAL front-end, then opens the browser once both are
ready. Stop with Ctrl+C to tear everything down cleanly.
"""
from __future__ import annotations

import argparse
import contextlib
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent


def wait_for(url: str, timeout: float = 30.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            with contextlib.closing(urllib.request.urlopen(url, timeout=2)):
                return True
        except urllib.error.URLError:
            time.sleep(0.4)
        except Exception:
            time.sleep(0.4)
    return False


def format_cmd(cmd: List[str]) -> str:
    return " ".join(cmd)


def launch_process(cmd: List[str], name: str, cwd: Path) -> subprocess.Popen:
    print(f"[{name}] {format_cmd(cmd)}", flush=True)
    return subprocess.Popen(cmd, cwd=str(cwd))


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch OVERHAUL backend + frontend")
    parser.add_argument("--api-host", default="127.0.0.1", help="Host for FastAPI/uvicorn")
    parser.add_argument("--api-port", type=int, default=8000, help="Port for FastAPI/uvicorn")
    parser.add_argument("--web-host", default="127.0.0.1", help="Host for static web server")
    parser.add_argument("--web-port", type=int, default=4173, help="Port for static web server")
    parser.add_argument("--no-browser", action="store_true", help="Skip auto-opening the browser")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without launching anything")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Start both servers, verify readiness, then shut them down",
    )
    args = parser.parse_args()

    if args.check_only:
        args.no_browser = True

    frontend_entry = ROOT / "OVERHUAL.html"
    if not frontend_entry.exists():
        raise SystemExit("Cannot find OVERHUAL.html. Did you delete it?")

    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app:app",
        "--host",
        args.api_host,
        "--port",
        str(args.api_port),
    ]
    frontend_cmd = [
        sys.executable,
        "-m",
        "http.server",
        str(args.web_port),
        "--bind",
        args.web_host,
    ]

    if args.dry_run:
        print("[dry-run] Backend:", format_cmd(backend_cmd))
        print("[dry-run] Frontend:", format_cmd(frontend_cmd))
        return

    processes: List[subprocess.Popen] = []
    try:
        backend = launch_process(backend_cmd, "backend", ROOT)
        processes.append(backend)
        if not wait_for(f"http://{args.api_host}:{args.api_port}/health", timeout=25):
            raise RuntimeError("Backend did not become healthy in time.")

        frontend = launch_process(frontend_cmd, "frontend", ROOT)
        processes.append(frontend)
        if not wait_for(f"http://{args.web_host}:{args.web_port}/OVERHUAL.html", timeout=10):
            raise RuntimeError("Frontend server did not respond in time.")

        target_url = f"http://{args.web_host}:{args.web_port}/OVERHUAL.html"
        print("\nOVERHAUL is live:")
        print(f"  Frontend → {target_url}")
        print(f"  API      → http://{args.api_host}:{args.api_port}")
        if not args.no_browser:
            webbrowser.open(target_url)

        if args.check_only:
            print("Servers verified successfully. Shutting down (check-only mode).")
            return

        print("\nPress Ctrl+C to stop both servers.")
        while True:
            for proc in list(processes):
                if proc.poll() is not None:
                    name = "backend" if proc is backend else "frontend"
                    raise RuntimeError(
                        f"{name} process exited unexpectedly with code {proc.returncode}."
                    )
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping servers…")
    except Exception as exc:
        print(f"\nError: {exc}")
    finally:
        for proc in processes:
            if proc.poll() is None:
                with contextlib.suppress(Exception):
                    proc.terminate()
        for proc in processes:
            if proc.poll() is None:
                with contextlib.suppress(Exception):
                    proc.kill()


if __name__ == "__main__":
    main()
