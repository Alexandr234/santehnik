#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Оркестратор пайплайна «ПТИЦЫ РЕАЛИСТИЧНЫЕ».

Последовательно запускает 5 скриптов и показывает весь их вывод
в одном окне (Tkinter) в реальном времени:

  1) ПРОМПТЫ/doc_prompt_pipeline_universal_ru_de_es_pl.py
  2) ВИЗУАЛ/flow_visual_batch_generator_realistic.py
  3) flow_repair_missing_videos_100percent.py
  4) analyze_voiceover_cutplan.py
  5) МОНТАЖ/video_creator_times_autovenv_no_subs_realistic.py

Запуск:
    python3 orchestrator_birds_pipeline.py            # окно
    python3 orchestrator_birds_pipeline.py --nogui    # только консоль

Возможности:
  * живой лог (stdout белым, stderr красным), автопрокрутка;
  * чекбоксы — какие шаги выполнять;
  * «Стоп» убивает всю группу процессов (включая дочерние ffmpeg и т.п.);
  * поле ввода — если скрипт что-то спрашивает, ответ уходит ему в stdin;
  * каждый прогон пишется в logs/pipeline_YYYYmmdd_HHMMSS.log;
  * по умолчанию при ошибке шага пайплайн останавливается
    (можно переключить «Продолжать при ошибке»).
"""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------
# Настройки: путь к папке проекта и список шагов
# --------------------------------------------------------------------------

BASE = Path("/Users/aleksandrtomilov/Desktop/ПТИЦЫ РЕАЛИСТИЧНЫЕ")

STEPS: list[tuple[str, Path]] = [
    ("1. Промпты (RU/DE/ES/PL)",
     BASE / "ПРОМПТЫ" / "doc_prompt_pipeline_universal_ru_de_es_pl.py"),
    ("2. Визуал (batch generator)",
     BASE / "ВИЗУАЛ" / "flow_visual_batch_generator_realistic.py"),
    ("3. Дозалив недостающих видео (100%)",
     BASE / "flow_repair_missing_videos_100percent.py"),
    ("4. Анализ озвучки / cutplan",
     BASE / "analyze_voiceover_cutplan.py"),
    ("5. Монтаж (video creator)",
     BASE / "МОНТАЖ" / "video_creator_times_autovenv_no_subs_realistic.py"),
]

LOG_DIR = Path(__file__).resolve().parent / "logs"
MAX_LINES = 20000          # сколько строк держать в окне, чтобы не тормозило


def resolve_python(script: Path) -> str:
    """Шаги запускаются тем же Python, которым запущен сам оркестратор —
    ровно как при ручном запуске `python3 script.py`."""
    exe = sys.executable or "python3"
    # pythonw не умеет нормально отдавать вывод дочерних процессов
    return exe.replace("pythonw", "python")


def child_env() -> dict:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"      # вывод шага появляется сразу, а не в конце
    env["PYTHONIOENCODING"] = "utf-8"
    return env


# --------------------------------------------------------------------------
# Движок: запуск шагов в фоновом потоке, вывод — через очередь
# --------------------------------------------------------------------------

class Runner:
    """Гоняет шаги по очереди. Все сообщения кладёт в out_queue как (tag, text)."""

    def __init__(self, steps: list[tuple[str, Path]], out_queue: "queue.Queue",
                 stop_on_error: bool = True, on_finish=None):
        self.steps = steps
        self.q = out_queue
        self.stop_on_error = stop_on_error
        self.on_finish = on_finish
        self.proc: subprocess.Popen | None = None
        self.cancelled = False
        self.results: list[tuple[str, str, float]] = []   # (имя, статус, секунды)
        self._thread: threading.Thread | None = None

    # -- helpers ----------------------------------------------------------
    def emit(self, text: str, tag: str = "info") -> None:
        self.q.put((tag, text))

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_all, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Останавливаем текущий шаг и всю его группу процессов."""
        self.cancelled = True
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        self.emit("\n⏹  Остановка: посылаю сигнал процессу…\n", "warn")
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:
                proc.terminate()
        except (ProcessLookupError, PermissionError, OSError):
            return
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.emit("⏹  Не отвечает — убиваю принудительно (SIGKILL).\n", "warn")
            try:
                if os.name == "posix":
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                else:
                    proc.kill()
            except (ProcessLookupError, PermissionError, OSError):
                pass

    def send_stdin(self, line: str) -> bool:
        """Отправить строку в stdin текущего шага (если он что-то спрашивает)."""
        proc = self.proc
        if proc is None or proc.poll() is not None or proc.stdin is None:
            return False
        try:
            proc.stdin.write(line + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, ValueError, OSError):
            return False
        self.emit(f"» {line}\n", "stdin")
        return True

    # -- основной цикл ----------------------------------------------------
    def _run_all(self) -> None:
        t_all = time.time()
        self.emit(f"▶  Старт пайплайна: {datetime.now():%d.%m.%Y %H:%M:%S}\n"
                  f"   Папка проекта: {BASE}\n"
                  f"   Шагов к выполнению: {len(self.steps)}\n", "head")

        for index, (name, script) in enumerate(self.steps, 1):
            if self.cancelled:
                self.results.append((name, "пропущен (остановка)", 0.0))
                continue

            self.emit(f"\n{'═' * 78}\n{name}\n{script}\n{'═' * 78}\n", "head")

            if not script.is_file():
                self.emit(f"✖  Файл не найден: {script}\n", "err")
                self.results.append((name, "нет файла", 0.0))
                if self.stop_on_error:
                    self.emit("Пайплайн остановлен (включено «стоп при ошибке»).\n", "err")
                    break
                continue

            code, elapsed = self._run_one(script)

            if self.cancelled and code != 0:
                self.results.append((name, "прерван", elapsed))
                self.emit(f"⏹  «{name}» прерван пользователем "
                          f"({fmt_time(elapsed)}).\n", "warn")
                break
            if code == 0:
                self.results.append((name, "ок", elapsed))
                self.emit(f"\n✔  Готово: {name} — {fmt_time(elapsed)}\n", "ok")
            else:
                self.results.append((name, f"ошибка (код {code})", elapsed))
                self.emit(f"\n✖  Ошибка в «{name}»: код возврата {code} "
                          f"({fmt_time(elapsed)})\n", "err")
                if self.stop_on_error:
                    self.emit("Пайплайн остановлен (включено «стоп при ошибке»).\n", "err")
                    break

        self._summary(time.time() - t_all)
        if self.on_finish:
            self.on_finish()

    def _run_one(self, script: Path) -> tuple[int, float]:
        started = time.time()
        cmd = [resolve_python(script), "-u", str(script)]
        self.emit(f"$ {' '.join(cmd)}\n\n", "cmd")

        popen_kwargs: dict = {}
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True   # своя группа -> killpg

        try:
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(script.parent),
                env=child_env(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                **popen_kwargs,
            )
        except OSError as exc:
            self.emit(f"✖  Не удалось запустить: {exc}\n", "err")
            return 1, time.time() - started

        readers = [
            threading.Thread(target=self._pump, args=(self.proc.stdout, "out"), daemon=True),
            threading.Thread(target=self._pump, args=(self.proc.stderr, "err"), daemon=True),
        ]
        for r in readers:
            r.start()

        code = self.proc.wait()
        for r in readers:
            r.join(timeout=5)
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except OSError:
            pass
        self.proc = None
        return code, time.time() - started

    def _pump(self, stream, tag: str) -> None:
        try:
            for line in iter(stream.readline, ""):
                self.q.put((tag, line))
        except (ValueError, OSError):
            pass
        finally:
            try:
                stream.close()
            except OSError:
                pass

    def _summary(self, total: float) -> None:
        lines = [f"\n{'═' * 78}\nИТОГ\n{'═' * 78}\n"]
        for name, status, elapsed in self.results:
            mark = "✔" if status == "ок" else "✖"
            lines.append(f"{mark}  {name:<42} {status:<20} {fmt_time(elapsed)}\n")
        lines.append(f"\nОбщее время: {fmt_time(total)}\n")
        failed = [r for r in self.results if r[1] != "ок"]
        tag = "err" if failed else "ok"
        lines.append("Пайплайн завершён без ошибок.\n" if not failed
                     else f"Шагов с проблемами: {len(failed)}.\n")
        self.emit("".join(lines), tag)


def fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}ч {m:02d}м {s:02d}с" if h else f"{m}м {s:02d}с"


# --------------------------------------------------------------------------
# Окно
# --------------------------------------------------------------------------

def run_gui() -> int:
    import tkinter as tk
    from tkinter import ttk, filedialog

    COLORS = {
        "bg": "#12141a", "fg": "#d6dae4", "head": "#66b3ff", "cmd": "#9aa4b8",
        "ok": "#5ddb8a", "err": "#ff6b6b", "warn": "#ffc857", "stdin": "#c792ea",
        "info": "#d6dae4", "out": "#d6dae4",
    }

    root = tk.Tk()
    root.title("Оркестратор пайплайна — ПТИЦЫ РЕАЛИСТИЧНЫЕ")
    root.geometry("1100x760")
    root.minsize(760, 480)

    state = {"runner": None, "log_file": None, "started": None}
    out_q: "queue.Queue[tuple[str, str]]" = queue.Queue()
    step_vars: list[tk.BooleanVar] = []

    # ---- верх: список шагов -------------------------------------------
    top = ttk.Frame(root, padding=(10, 8, 10, 4))
    top.pack(fill="x")
    ttk.Label(top, text="Шаги пайплайна (снимите галочку, чтобы пропустить):"
              ).pack(anchor="w")

    steps_box = ttk.Frame(top)
    steps_box.pack(fill="x", pady=(4, 0))
    status_labels: list[ttk.Label] = []
    for i, (name, script) in enumerate(STEPS):
        var = tk.BooleanVar(value=True)
        step_vars.append(var)
        row = ttk.Frame(steps_box)
        row.pack(fill="x")
        ttk.Checkbutton(row, text=name, variable=var).pack(side="left")
        lbl = ttk.Label(row, text="" if script.is_file() else "  ⚠ файл не найден",
                        foreground="#b04040")
        lbl.pack(side="left")
        status_labels.append(lbl)

    # ---- панель управления ---------------------------------------------
    bar = ttk.Frame(root, padding=(10, 6))
    bar.pack(fill="x")
    stop_on_error = tk.BooleanVar(value=True)
    autoscroll = tk.BooleanVar(value=True)

    btn_start = ttk.Button(bar, text="▶  Запустить")
    btn_stop = ttk.Button(bar, text="⏹  Стоп", state="disabled")
    btn_clear = ttk.Button(bar, text="Очистить")
    btn_save = ttk.Button(bar, text="Сохранить лог…")
    for b in (btn_start, btn_stop, btn_clear, btn_save):
        b.pack(side="left", padx=(0, 6))
    ttk.Checkbutton(bar, text="Стоп при ошибке", variable=stop_on_error).pack(side="left", padx=8)
    ttk.Checkbutton(bar, text="Автопрокрутка", variable=autoscroll).pack(side="left")
    status_var = tk.StringVar(value="Готов к запуску.")
    ttk.Label(bar, textvariable=status_var).pack(side="right")

    # ---- лог -------------------------------------------------------------
    log_frame = ttk.Frame(root, padding=(10, 0, 10, 6))
    log_frame.pack(fill="both", expand=True)
    text = tk.Text(log_frame, wrap="none", bg=COLORS["bg"], fg=COLORS["fg"],
                   insertbackground=COLORS["fg"], font=("Menlo", 12),
                   relief="flat", padx=8, pady=6)
    yscroll = ttk.Scrollbar(log_frame, orient="vertical", command=text.yview)
    xscroll = ttk.Scrollbar(log_frame, orient="horizontal", command=text.xview)
    text.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
    text.grid(row=0, column=0, sticky="nsew")
    yscroll.grid(row=0, column=1, sticky="ns")
    xscroll.grid(row=1, column=0, sticky="ew")
    log_frame.rowconfigure(0, weight=1)
    log_frame.columnconfigure(0, weight=1)
    for tag, color in COLORS.items():
        text.tag_configure(tag, foreground=color)
    text.tag_configure("head", foreground=COLORS["head"], font=("Menlo", 12, "bold"))
    text.configure(state="disabled")

    # ---- ввод для интерактивных скриптов --------------------------------
    bottom = ttk.Frame(root, padding=(10, 0, 10, 10))
    bottom.pack(fill="x")
    ttk.Label(bottom, text="Ввод в скрипт:").pack(side="left")
    entry = ttk.Entry(bottom)
    entry.pack(side="left", fill="x", expand=True, padx=6)

    def send_input(_event=None):
        runner = state["runner"]
        line = entry.get()
        if runner and runner.send_stdin(line):
            entry.delete(0, "end")
        else:
            status_var.set("Сейчас нет запущенного шага — вводить некуда.")

    entry.bind("<Return>", send_input)
    ttk.Button(bottom, text="Отправить", command=send_input).pack(side="left")

    # ---- вывод в окно ----------------------------------------------------
    def append(tag: str, chunk: str) -> None:
        text.configure(state="normal")
        text.insert("end", chunk, tag)
        # не даём буферу расти бесконечно
        line_count = int(text.index("end-1c").split(".")[0])
        if line_count > MAX_LINES:
            text.delete("1.0", f"{line_count - MAX_LINES}.0")
        if autoscroll.get():
            text.see("end")
        text.configure(state="disabled")
        log = state["log_file"]
        if log:
            try:
                log.write(chunk)
                log.flush()
            except OSError:
                pass

    def drain() -> None:
        batch: dict[str, list[str]] = {}
        order: list[str] = []
        for _ in range(500):                    # порциями, чтобы окно не подвисало
            try:
                tag, chunk = out_q.get_nowait()
            except queue.Empty:
                break
            if tag not in batch:
                batch[tag] = []
                order.append(tag)
            batch[tag].append(chunk)
        for tag in order:
            append(tag, "".join(batch[tag]))
        if state["runner"] and state["started"]:
            status_var.set(f"Выполняется… {fmt_time(time.time() - state['started'])}")
        root.after(100, drain)

    # ---- кнопки ----------------------------------------------------------
    def on_finish() -> None:
        def done():
            state["runner"] = None
            state["started"] = None
            btn_start.configure(state="normal")
            btn_stop.configure(state="disabled")
            status_var.set("Завершено.")
            log = state["log_file"]
            if log:
                try:
                    log.close()
                except OSError:
                    pass
                state["log_file"] = None
        root.after(0, done)

    def start() -> None:
        selected = [s for s, var in zip(STEPS, step_vars) if var.get()]
        if not selected:
            status_var.set("Не выбран ни один шаг.")
            return
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            path = LOG_DIR / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"
            state["log_file"] = open(path, "w", encoding="utf-8")
            out_q.put(("cmd", f"Лог пишется в {path}\n"))
        except OSError as exc:
            out_q.put(("warn", f"Не удалось открыть файл лога: {exc}\n"))
            state["log_file"] = None

        runner = Runner(selected, out_q, stop_on_error=stop_on_error.get(),
                        on_finish=on_finish)
        state["runner"] = runner
        state["started"] = time.time()
        btn_start.configure(state="disabled")
        btn_stop.configure(state="normal")
        status_var.set("Выполняется…")
        runner.start()

    def stop() -> None:
        runner = state["runner"]
        if runner:
            btn_stop.configure(state="disabled")
            status_var.set("Останавливаю…")
            threading.Thread(target=runner.stop, daemon=True).start()

    def clear() -> None:
        text.configure(state="normal")
        text.delete("1.0", "end")
        text.configure(state="disabled")

    def save() -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".log",
            initialfile=f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log",
            filetypes=[("Лог", "*.log"), ("Все файлы", "*.*")])
        if not path:
            return
        try:
            Path(path).write_text(text.get("1.0", "end"), encoding="utf-8")
            status_var.set(f"Лог сохранён: {path}")
        except OSError as exc:
            status_var.set(f"Не удалось сохранить: {exc}")

    btn_start.configure(command=start)
    btn_stop.configure(command=stop)
    btn_clear.configure(command=clear)
    btn_save.configure(command=save)

    def on_close() -> None:
        runner = state["runner"]
        if runner:
            runner.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.after(100, drain)
    out_q.put(("info", "Нажмите «Запустить», чтобы прогнать пайплайн целиком.\n"))
    root.mainloop()
    return 0


# --------------------------------------------------------------------------
# Консольный режим
# --------------------------------------------------------------------------

def run_console() -> int:
    q: "queue.Queue[tuple[str, str]]" = queue.Queue()
    done = threading.Event()
    runner = Runner(STEPS, q, stop_on_error=True, on_finish=done.set)
    runner.start()
    while not done.is_set() or not q.empty():
        try:
            _tag, chunk = q.get(timeout=0.2)
        except queue.Empty:
            continue
        sys.stdout.write(chunk)
        sys.stdout.flush()
    return 0 if all(r[1] == "ок" for r in runner.results) else 1


def main() -> int:
    if "--nogui" in sys.argv:
        return run_console()
    try:
        import tkinter  # noqa: F401
    except ImportError:
        print("Tkinter недоступен в этом Python. Запустите с --nogui "
              "или поставьте python.org-сборку Python с Tk.", file=sys.stderr)
        return run_console()
    return run_gui()


if __name__ == "__main__":
    sys.exit(main())
