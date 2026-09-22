#!/usr/bin/env python3
"""
Модуль для взаимодействия с обёрткой солвера
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

RUN_DIR = Path("./run").resolve()
RUN_DIR.mkdir(exist_ok=True)


def is_pid_alive(pid: int) -> bool:
    """Проверка живости процесса без отправки сигнала (POSIX)."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # процесс есть, но не наш — считаем живым
    return True


def read_status(task_id: str) -> dict:
    status_file = RUN_DIR / f"{task_id}.status"
    if not status_file.exists():
        return {"status": "unknown"}

    try:
        data = json.loads(status_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"status": "unknown"}

    # Перепроверяем живость по PID, потому что файл мог устареть
    pid = data.get("pid")
    if data.get("status") in {"running", "starting", "terminating"} and pid:
        if not is_pid_alive(pid):
            # Обёртка умерла, не успев записать финальный статус
            data["status"] = "dead"
    return data


def read_exitcode(task_id: str) -> int | None:
    f = RUN_DIR / f"{task_id}.exitcode"
    if not f.exists():
        return None
    try:
        return int(f.read_text().strip())
    except ValueError:
        return None


def start_task(task_id: str, command: list[str]) -> None:
    """Запускает launcher.py как независимый процесс."""
    # Если задача уже идёт — не запускаем дубль
    status = read_status(task_id)
    if status.get("status") in {"running", "starting"}:
        st.warning(f"Задача {task_id} уже выполняется (pid={status.get('pid')})")
        return

    # Чистим старые файлы, чтобы не путаться
    for suffix in (".exitcode", ".pid", ".status", ".log"):
        p = RUN_DIR / f"{task_id}{suffix}"
        if p.exists():
            p.unlink()

    launcher = Path(__file__).parent / "launcher.py"
    cmd = [
        sys.executable, str(launcher),
        "--task-id", task_id,
        "--log-dir", str(RUN_DIR),
        "--",
        *command,
    ]

    # Ключевой момент: start_new_session=True отвязывает launcher от Streamlit.
    # stdout/stderr → DEVNULL, потому что launcher сам пишет в файл.
    subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
        cwd=str(Path(__file__).parent),
    )