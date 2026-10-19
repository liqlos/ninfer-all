import asyncio
import contextlib
import ctypes
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import shutil
import sys
import threading
import time
import uuid
from ctypes import wintypes
from collections import deque
from contextlib import asynccontextmanager
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOG_PATH = ROOT / "logs" / "runtime.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
LOG_STREAM = LOG_PATH.open("a", encoding="utf-8", buffering=1)
if sys.stdout is None:
    sys.stdout = LOG_STREAM
if sys.stderr is None:
    sys.stderr = LOG_STREAM

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
from wsl_docker import DockerRuntimeError, WSLDocker
from batch import BatchManager
from reasoning import apply_reasoning_level, default_output_tokens, normalize_reasoning_level


CONFIG_PATH = Path(os.environ.get("GPU_BROKER_CONFIG", ROOT / "config.json"))
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def gpu_memory_budget(total_bytes, driver_used_bytes, external_bytes, reserve_fraction=0.3):
    if total_bytes is None or driver_used_bytes is None or external_bytes is None:
        return {"known": False, "external_reserve_bytes": None, "target_llm_bytes": None, "available_llm_bytes": None}
    total_bytes = max(0, int(total_bytes))
    driver_used_bytes = max(0, int(driver_used_bytes))
    external_bytes = max(0, int(external_bytes))
    ratio = Fraction(str(reserve_fraction))
    external_reserve_bytes = (external_bytes * ratio.numerator + ratio.denominator - 1) // ratio.denominator
    target_llm_bytes = max(0, total_bytes - external_bytes - external_reserve_bytes)
    llm_and_system_used_bytes = max(0, driver_used_bytes - external_bytes)
    return {
        "known": True,
        "external_reserve_bytes": external_reserve_bytes,
        "target_llm_bytes": target_llm_bytes,
        "available_llm_bytes": max(0, target_llm_bytes - llm_and_system_used_bytes),
        "llm_and_system_used_bytes": llm_and_system_used_bytes,
    }


def retryable_acquisition_error(exc):
    detail = exc.detail if isinstance(exc.detail, dict) else {}
    if isinstance(exc.detail, str) and exc.detail.startswith(("Managed container failed to start:", "Timed out waiting for managed container")):
        return False
    if "retryable" in detail:
        return bool(detail["retryable"])
    return bool(detail.get("retryable")) or exc.status_code == 503


def unprotect_machine_secret(path):
    class DataBlob(ctypes.Structure):
        _fields_ = [("cbData", ctypes.c_uint32), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]

    encrypted = Path(path).read_bytes()
    source_buffer = ctypes.create_string_buffer(encrypted)
    source = DataBlob(len(encrypted), ctypes.cast(source_buffer, ctypes.POINTER(ctypes.c_ubyte)))
    target = DataBlob()
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    if not crypt32.CryptUnprotectData(ctypes.byref(source), None, None, None, None, 0, ctypes.byref(target)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return ctypes.string_at(target.pbData, target.cbData).decode("utf-8")
    finally:
        kernel32.LocalFree(target.pbData)


class LeaseRequest(BaseModel):
    owner: str = Field(min_length=1, max_length=100)
    kind: str = Field(pattern="^(agent|video)$")
    model_name: str | None = Field(default=None, max_length=200)
    requested_agents: int = Field(default=1, ge=1, le=32)
    context_length: int | None = Field(default=None, ge=1, le=1048576)
    ttl_seconds: int = Field(default=300, ge=15, le=21600)


class Renewal(BaseModel):
    ttl_seconds: int = Field(default=300, ge=15, le=21600)


class Broker:
    def __init__(self, config):
        self.config = config
        self.client = None
        self.db = sqlite3.connect(str(ROOT / config["database"]), isolation_level=None, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS leases (
                id TEXT PRIMARY KEY,
                owner TEXT NOT NULL,
                kind TEXT NOT NULL,
                model_name TEXT,
                requested_agents INTEGER NOT NULL,
                context_length INTEGER,
                expires REAL NOT NULL,
                state TEXT NOT NULL,
                created REAL NOT NULL,
                updated REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lease_id TEXT,
                event TEXT NOT NULL,
                payload TEXT NOT NULL,
                created REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS agent_jobs (
                id TEXT PRIMARY KEY,
                owner TEXT NOT NULL,
                workspace TEXT NOT NULL,
                lease_id TEXT,
                pid INTEGER,
                state TEXT NOT NULL,
                progress TEXT,
                result TEXT,
                error TEXT,
                worktree_path TEXT,
                branch TEXT,
                created REAL NOT NULL,
                updated REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS inference_calls (
                id TEXT PRIMARY KEY,
                lease_id TEXT,
                model_name TEXT,
                api TEXT,
                started REAL,
                elapsed REAL,
                status TEXT,
                http_status INTEGER,
                input_tokens INTEGER,
                output_tokens INTEGER
            );
            CREATE INDEX IF NOT EXISTS inference_calls_started ON inference_calls(started DESC);
            CREATE TABLE IF NOT EXISTS model_kv_cache (
                container_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                capacity_tokens INTEGER NOT NULL,
                measured REAL NOT NULL
            );
        """)
        columns = {row["name"] for row in self.db.execute("PRAGMA table_info(agent_jobs)").fetchall()}
        for name in ("worktree_path", "branch"):
            if name not in columns:
                self.db.execute(f"ALTER TABLE agent_jobs ADD COLUMN {name} TEXT")
        for name, data_type in (("idempotency_key", "TEXT"), ("request_hash", "TEXT"), ("context_length", "INTEGER"), ("task_summary", "TEXT")):
            if name not in columns:
                self.db.execute(f"ALTER TABLE agent_jobs ADD COLUMN {name} {data_type}")
        inference_columns = {row["name"] for row in self.db.execute("PRAGMA table_info(inference_calls)").fetchall()}
        if "http_status" not in inference_columns:
            self.db.execute("ALTER TABLE inference_calls ADD COLUMN http_status INTEGER")
        self.db.execute("CREATE UNIQUE INDEX IF NOT EXISTS agent_jobs_idempotency_key ON agent_jobs(idempotency_key) WHERE idempotency_key IS NOT NULL")
        self.lock = asyncio.Lock()
        self.model_start_lock = asyncio.Lock()
        self.calls = {}
        self.inference_reservations = {}
        self.agent_tasks = {}
        self.agent_processes = {}
        self.agent_leases = {}
        self.kv_capacity_by_instance = {}
        self.model_cache = {}
        self.model_cache_ttl_seconds = 10.0
        self.external_snapshot = {"available": False, "external_memory_bytes": 0}
        self.external_lock = threading.Lock()
        self.external_pressure_active = False
        self.external_pressure_leases = set()
        self.pressure_sampled_monotonic = 0.0
        self.pressure_telemetry_available = False
        self.pressure_refresh_lock = asyncio.Lock()
        self.budget_revision = 0
        self.external_sample_revision = 0
        self.model_state = "idle"
        self.model_ready = False
        self.ready_container_id = None
        self.startup_probe_task = None
        self.draining = False
        self.last_model_activity = time.monotonic()
        self.capacity_changed = asyncio.Event()
        self.max_agent_leases = int(config.get("max_agent_leases", 64))
        self.timeout = float(config.get("model_start_timeout_seconds", 300))
        runtime = config.get("container_runtime") or {}
        self.docker = WSLDocker(runtime.get("distribution", "Ubuntu-24.04"), runtime.get("executable", "/usr/bin/docker"), config.get("docker_command_timeout_seconds", 30))
        self.docker_available = False
        self.runtime_error = None
        self.runtime_gpu_baseline = {}

    def uses_wsl_docker(self):
        return getattr(self, "config", {}).get("runtime_backend") == "wsl_docker"

    def container_spec(self, name):
        models = (self.config.get("container_runtime") or {}).get("models") or {}
        if name not in models:
            raise HTTPException(404, f"Managed container is not configured: {name}")
        return models[name]

    async def gpu_memory_snapshot(self):
        try:
            process = await asyncio.create_subprocess_exec(
                "nvidia-smi.exe",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            try:
                output, _ = await asyncio.wait_for(process.communicate(), timeout=8)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise TimeoutError("Windows nvidia-smi timed out")
            text = output.decode("utf-8", "replace").strip()
            if process.returncode:
                raise RuntimeError(text[-1000:])
            rows = [line.strip() for line in text.splitlines() if line.strip()]
            if len(rows) != 1:
                self.runtime_gpu_error = f"Expected one GPU row from nvidia-smi, received {len(rows)}"
                return None
            name, total, used = [item.strip() for item in rows[0].split(",", 2)]
            total_bytes = int(total) * 1024 * 1024
            used_bytes = int(used) * 1024 * 1024
            self.runtime_gpu_error = None
            return {"name": name, "total_bytes": total_bytes, "used_bytes": used_bytes, "free_bytes": max(0, total_bytes - used_bytes)}
        except (OSError, TimeoutError, DockerRuntimeError, ValueError) as exc:
            self.runtime_gpu_error = str(exc)[:1000]
            return None

    async def wait_gpu_release(self, model_name):
        baseline = self.runtime_gpu_baseline.pop(model_name, None)
        if not baseline:
            return
        deadline = time.monotonic() + float(self.config.get("gpu_release_timeout_seconds", 45))
        allowance = int(self.config.get("gpu_release_tolerance_bytes", 536870912))
        while time.monotonic() < deadline:
            snapshot = await self.gpu_memory_snapshot()
            external = await self.observe_external()
            if snapshot and external.get("available"):
                extra_external = max(0, int(external.get("external_memory_bytes") or 0) - baseline["external_memory_bytes"])
                if snapshot["used_bytes"] <= baseline["used_bytes"] + extra_external + allowance:
                    return
            await asyncio.sleep(2)
        snapshot = await self.gpu_memory_snapshot()
        if not snapshot:
            raise HTTPException(503, "Cannot verify GPU memory after managed container stop")
        extra_external = max(0, int(self.external_snapshot.get("external_memory_bytes") or 0) - baseline["external_memory_bytes"])
        if snapshot["used_bytes"] > baseline["used_bytes"] + extra_external + allowance:
            raise HTTPException(504, f"Managed container stopped but GPU memory did not return to its baseline: {snapshot['used_bytes']} bytes used")

    async def start(self):
        self.model_ready = False
        self.ready_container_id = None
        now = time.time()
        rows = self.db.execute("SELECT id,state,pid FROM agent_jobs WHERE state IN ('queued','acquiring','starting','running','cancel_requested')").fetchall()
        for row in rows:
            if self.stop_orphan_agent_process(row["pid"], row["id"]):
                self.job_event(row["id"], "orphan_process_stopped", "broker restarted")
        self.db.execute("UPDATE leases SET state='interrupted',updated=? WHERE state IN ('starting','active','evicting','expiring')", (now,))
        self.db.execute("UPDATE agent_jobs SET state='interrupted',progress='broker restarted',error='Broker restarted; saved worktree retained',updated=? WHERE state IN ('queued','acquiring','starting','running','cancel_requested')", (now,))
        for row in rows:
            self.job_event(row["id"], "interrupted", "broker restarted")
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(60, connect=5), trust_env=False)
        try:
            await self.docker.info()
            self.docker_available = True
            self.runtime_error = None
            name = next(iter(self.config.get("codex_model_names", [])), None)
            model = await self.model(name) if name else None
            self.model_state = "loading" if model and model.get("replicas") else "idle"
            if model and model.get("replicas"):
                self.startup_probe_task = asyncio.create_task(self.reconcile_warm_model(name))
        except (OSError, TimeoutError, DockerRuntimeError) as exc:
            self.docker_available = False
            self.runtime_error = str(exc)[:1000]
            self.model_state = "error"

    async def reconcile_warm_model(self, model_name):
        try:
            await self.wait_model(model_name, True)
        except asyncio.CancelledError:
            raise
        except (HTTPException, OSError, TimeoutError, DockerRuntimeError):
            self.model_state = "ready" if self.model_ready else "error"

    def record_external_sample(self, sample):
        now = time.monotonic()
        with self.external_lock:
            self.external_sample_revision = getattr(self, "external_sample_revision", 0) + 1
            self.external_snapshot = getattr(self, "external_snapshot", {"available": False, "external_memory_bytes": None})
            prior = (self.external_snapshot.get("available"), self.external_snapshot.get("external_memory_bytes"))
            if sample.get("available"):
                self.external_snapshot = {**sample, "sampled_monotonic": now, "sample_revision": self.external_sample_revision}
            else:
                self.external_snapshot = {**sample, "available": False, "sampled_monotonic": now, "sample_revision": self.external_sample_revision}
            current = (self.external_snapshot.get("available"), self.external_snapshot.get("external_memory_bytes"))
            if current != prior:
                self.revise_budget()
        return self.external_snapshot

    async def monitor_external(self):
        interval = float(self.config.get("external_monitor_interval_seconds", 1))
        attribution_interval = float(self.config.get("external_attribution_interval_seconds", 5))
        last_attribution = 0.0
        while True:
            with contextlib.suppress(Exception):
                await self.expire_leases()
            if time.monotonic() - last_attribution >= attribution_interval:
                sample = await asyncio.to_thread(self.windows_gpu)
                self.record_external_sample(sample)
                last_attribution = time.monotonic()
            if not self.docker_available and time.monotonic() - last_attribution >= attribution_interval:
                try:
                    await self.docker.info()
                    self.docker_available = True
                    self.runtime_error = None
                    self.model_state = "idle"
                except (OSError, TimeoutError, DockerRuntimeError) as exc:
                    self.runtime_error = str(exc)[:1000]
            with contextlib.suppress(Exception):
                await self.evaluate_external_pressure()
            if self.uses_wsl_docker() and not self.active_leases() and self.agent_tasks == {} and time.monotonic() - self.last_model_activity >= float(self.config.get("model_idle_unload_seconds", 300)):
                with contextlib.suppress(Exception):
                    await self.unload_model(False, timeout=0)
            await asyncio.sleep(interval)

    async def observe_external(self):
        with self.external_lock:
            stale = not self.external_snapshot.get("sampled_monotonic") or time.monotonic() - self.external_snapshot["sampled_monotonic"] > max(2.5, float(self.config.get("external_attribution_interval_seconds", 5)) * 1.5)
            snapshot = dict(self.external_snapshot)
        if stale:
            snapshot = self.record_external_sample(await asyncio.to_thread(self.windows_gpu))
        return snapshot

    async def close(self):
        probe = getattr(self, "startup_probe_task", None)
        if probe and not probe.done():
            probe.cancel()
            await asyncio.gather(probe, return_exceptions=True)
        shutdown_batches = [batch_id for batch_id, task in batch_manager.tasks.items() if not task.done() and not task.cancelling()]
        tasks = list(batch_manager.tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if shutdown_batches:
            now = time.time()
            interruption = json.dumps({"code": "broker_restarted"})
            for batch_id in shutdown_batches:
                self.db.execute("UPDATE inference_batch_items SET state='interrupted',error=?,updated=? WHERE batch_id=? AND state IN ('queued','running','cancelled')", (interruption, now, batch_id))
                self.db.execute("UPDATE inference_batches SET state='interrupted',updated=? WHERE id=? AND state IN ('running','cancelled')", (now, batch_id))
        processes = list(self.agent_processes.values())
        for process in processes:
            await self.terminate_agent_process(process)
        tasks = list(self.agent_tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for name in self.config.get("codex_model_names", []):
            with contextlib.suppress(Exception):
                model = await self.model(name)
                if model.get("replicas", 0):
                    await self.set_replicas(model, 0)
        if self.client:
            await self.client.aclose()
        self.db.close()

    def event(self, lease_id, name, **payload):
        self.db.execute("INSERT INTO events(lease_id,event,payload,created) VALUES(?,?,?,?)", (lease_id, name, json.dumps(payload), time.time()))

    def revise_budget(self):
        self.budget_revision = getattr(self, "budget_revision", 0) + 1
        changed = getattr(self, "capacity_changed", None)
        if changed:
            changed.set()

    def sweep(self):
        return self.db.execute("SELECT id FROM leases WHERE state IN ('starting','active','expiring') AND expires<=?", (time.time(),)).fetchall()

    async def expire_leases(self):
        async with self.lock:
            rows = self.db.execute("SELECT * FROM leases WHERE state IN ('starting','active','expiring') AND expires<=? ORDER BY created", (time.time(),)).fetchall()
            targets = [dict(row) for row in rows]
            ids = {row["id"] for row in targets}
            for row in targets:
                if row["state"] != "expiring":
                    self.db.execute("UPDATE leases SET state='expiring',updated=? WHERE id=?", (time.time(), row["id"]))
                    self.event(row["id"], "LEASE_EXPIRING", owner=row["owner"])
            for task, call in list(self.calls.items()):
                if call["lease_id"] in ids:
                    task.cancel()
            jobs = self.db.execute("SELECT id,lease_id FROM agent_jobs WHERE lease_id IN ({}) AND state IN ('queued','acquiring','starting','running','cancel_requested')".format(",".join("?" for _ in ids) or "NULL"), tuple(ids)).fetchall() if ids else []
        tasks = []
        for job in jobs:
            await self.terminate_agent_process(self.agent_processes.get(job["id"]))
            task = self.agent_tasks.get(job["id"])
            if task:
                task.cancel()
                tasks.append(task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        deadline = time.monotonic() + float(self.config.get("lease_expiry_timeout_seconds", 20))
        while any(call["lease_id"] in ids for call in self.calls.values()) and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        async with self.lock:
            for row in targets:
                current = self.lease(row["id"])
                if current["state"] == "expiring" and not any(call["lease_id"] == row["id"] for call in self.calls.values()):
                    self.db.execute("UPDATE leases SET state='expired',updated=? WHERE id=?", (time.time(), row["id"]))
                    self.event(row["id"], "LEASE_EXPIRED", owner=row["owner"])
                    self.revise_budget()

    def lease(self, lease_id):
        row = self.db.execute("SELECT * FROM leases WHERE id=?", (lease_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "Unknown GPU lease")
        return dict(row)

    def active_leases(self, kind=None):
        self.sweep()
        if kind:
            rows = self.db.execute("SELECT * FROM leases WHERE state IN ('starting','active','evicting','expiring') AND kind=? ORDER BY created", (kind,)).fetchall()
        else:
            rows = self.db.execute("SELECT * FROM leases WHERE state IN ('starting','active','evicting','expiring') ORDER BY created").fetchall()
        return [dict(row) for row in rows]

    async def model(self, name):
        spec = self.container_spec(name)
        if not hasattr(self, "model_cache"):
            self.model_cache = {}
            self.model_cache_ttl_seconds = 10.0
        cached = self.model_cache.get(name)
        now = time.monotonic()
        if cached and now - cached[0] < self.model_cache_ttl_seconds:
            container = cached[1]
            running = bool(container and (container.get("State") or {}).get("Status") == "running")
            return {"id": name, "name": name, "replicas": int(running), "ready_replicas": int(running and getattr(self, "model_ready", False) and (container or {}).get("Id") == getattr(self, "ready_container_id", None)), "backend_parameters": spec.get("backend_parameters") or [], "container": container}
        try:
            container = await self.docker.inspect(spec["container_name"])
            self.docker_available = True
            self.runtime_error = None
        except (OSError, TimeoutError, DockerRuntimeError) as exc:
            self.model_cache.pop(name, None)
            self.docker_available = False
            self.runtime_error = str(exc)[:1000]
            raise
        previous = self.model_cache.get(name)
        if previous and (previous[1] or {}).get("Id") != (container or {}).get("Id"):
            self.kv_capacity_by_instance.clear()
            self.model_ready = False
            self.ready_container_id = None
        self.model_cache[name] = (time.monotonic(), container)
        running = bool(container and (container.get("State") or {}).get("Status") == "running")
        return {"id": name, "name": name, "replicas": int(running), "ready_replicas": int(running and getattr(self, "model_ready", False) and (container or {}).get("Id") == getattr(self, "ready_container_id", None)), "backend_parameters": spec.get("backend_parameters") or [], "container": container}

    async def route(self, name):
        spec = self.container_spec(name)
        return {"id": name, "base_url": f"http://127.0.0.1:{int(spec['host_port'])}"}

    async def instances(self, model_id=None):
        names = (self.config.get("container_runtime") or {}).get("models", {})
        selected = [model_id] if model_id in names else list(names) if model_id is None else []
        result = []
        for name in selected:
            spec = names[name]
            container = await self.docker.inspect(spec["container_name"])
            if container:
                result.append({"id": name, "model_id": name, "state": (container.get("State") or {}).get("Status"), "worker_ip": "127.0.0.1", "port": spec["host_port"], "container": container})
        return result

    async def set_replicas(self, model, replicas):
        name = model["name"]
        spec = self.container_spec(name)
        if replicas:
            self.model_cache.pop(name, None)
            self.kv_capacity_by_instance.clear()
            self.model_state = "loading"
            self.model_ready = False
            self.ready_container_id = None
            snapshot = await self.gpu_memory_snapshot()
            external = await self.observe_external()
            self.runtime_gpu_baseline[name] = {"used_bytes": snapshot["used_bytes"], "external_memory_bytes": int(external["external_memory_bytes"])} if snapshot and external.get("available") else None
            label = self.config["container_runtime"]["managed_label"]
            container = await self.docker.start_spec(name, spec, label)
            if (container.get("State") or {}).get("Status") != "running":
                self.model_state = "error"
                raise HTTPException(503, {"code": "model_start_failed", "requested": {"model": name}, "available": {"container_state": (container.get("State") or {}).get("Status")}, "limiting_factor": "model_startup", "retryable": False, "alternatives": []})
            return await self.model(name)
        self.model_state = "unloading"
        self.model_cache.pop(name, None)
        self.kv_capacity_by_instance.clear()
        self.model_ready = False
        self.ready_container_id = None
        result = await self.docker.stop_and_remove(spec["container_name"], timeout=int(self.config.get("container_stop_timeout_seconds", 60)))
        self.model_state = "idle"
        return {**model, "replicas": 0, "ready_replicas": 0, "stop_result": result}

    async def wait_model(self, model_name, ready, timeout=None):
        spec = self.container_spec(model_name)
        deadline = time.monotonic() + (timeout or self.timeout)
        url = f"http://127.0.0.1:{int(spec['host_port'])}/health"
        while time.monotonic() < deadline:
            container = await self.docker.inspect(spec["container_name"])
            self.model_cache.pop(model_name, None)
            running = bool(container and (container.get("State") or {}).get("Status") == "running")
            if ready and running:
                try:
                    response = await self.client.get(url, timeout=3)
                    if response.status_code == 200:
                        models_response = await self.client.get(f"http://127.0.0.1:{int(spec['host_port'])}/v1/models", timeout=3)
                        models_response.raise_for_status()
                        if any(item.get("id") == model_name for item in models_response.json().get("data") or []):
                            await self.kv_capacity_tokens(await self.model(model_name))
                            self.model_ready = True
                            self.ready_container_id = container.get("Id")
                            self.model_state = "ready"
                            return await self.model(model_name)
                except (httpx.HTTPError, ValueError, HTTPException):
                    pass
            if not ready and not container:
                await self.wait_gpu_release(model_name)
                self.model_state = "idle"
                return {"name": model_name, "replicas": 0, "ready_replicas": 0}
            if ready and container and not running:
                logs = await self.docker.logs(spec["container_name"], tail=200)
                self.model_state = "error"
                raise HTTPException(503, {"code": "model_start_failed", "requested": {"model": model_name}, "available": {"logs": logs[-12000:]}, "limiting_factor": "model_startup", "retryable": False, "alternatives": []})
            await asyncio.sleep(2)
        container = await self.docker.inspect(spec["container_name"])
        logs = await self.docker.logs(spec["container_name"], tail=200) if container else "container absent"
        self.model_state = "error"
        raise HTTPException(504, {"code": "model_start_timeout", "requested": {"model": model_name}, "available": {"container_state": (container.get("State") or {}).get("Status") if container else "absent", "logs": logs[-12000:]}, "limiting_factor": "model_startup", "retryable": False, "alternatives": []})

    def windows_gpu(self):
        script = r"$m=Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory -ErrorAction SilentlyContinue; $e=Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine -ErrorAction SilentlyContinue; $mem=@($m | ForEach-Object { $n=$_.Name; $p=[int]$_.ProcessID; if(!$p -and $n -match 'pid_(\d+)'){ $p=[int]$Matches[1] }; $g=Get-Process -Id $p -ErrorAction SilentlyContinue; @{name=$n;pid=$p;process=if($g){$g.ProcessName}else{'unknown'};bytes=[int64]$_.DedicatedUsage} }); $eng=@($e | ForEach-Object { $n=$_.Name; $p=0; if($n -match 'pid_(\d+)'){ $p=[int]$Matches[1] }; $g=Get-Process -Id $p -ErrorAction SilentlyContinue; @{name=$n;pid=$p;process=if($g){$g.ProcessName}else{'unknown'};util=[double]$_.UtilizationPercentage} }); $r=@{memory=$mem;engine=$eng}; $r | ConvertTo-Json -Compress -Depth 4"
        try:
            result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True, text=True, timeout=6, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if result.returncode != 0 or not result.stdout.strip():
                return {"available": False, "external_busy": None, "error": result.stderr.strip()[:300]}
            raw = json.loads(result.stdout)
            memory = raw.get("memory") or []
            engine = raw.get("engine") or []
            if isinstance(memory, dict): memory = [memory]
            if isinstance(engine, dict): engine = [engine]
            if not memory:
                return {"available": False, "external_busy": None, "error": "Windows GPU memory counters returned no processes"}
            mem_limit = int(self.config.get("external_process_memory_mb", 64)) * 1024 * 1024
            active_memory = [x for x in memory if x.get("bytes", 0) >= mem_limit and x.get("process", "").lower() not in ("vmwp", "vmmem")]
            active_engine = [x for x in engine if x.get("util", 0) > 0 and x.get("process", "").lower() not in ("vmwp", "vmmem")]
            external_bytes = sum(x.get("bytes", 0) for x in active_memory)
            fraction = float(self.config.get("external_reserve_fraction", 0.3))
            margin = int(external_bytes * fraction)
            return {"available": True, "external_busy": bool(active_memory or active_engine), "external_activity": bool(active_memory or active_engine), "external_memory_bytes": external_bytes, "external_reserve_fraction": fraction, "external_memory_with_margin_bytes": external_bytes + margin, "additional_margin_bytes": margin, "memory_threshold_mb": mem_limit // (1024 * 1024), "process_memory": active_memory, "active_engines": active_engine}
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, TypeError, ValueError) as exc:
            return {"available": False, "external_busy": None, "error": str(exc)[:300]}

    async def evaluate_external_pressure(self):
        external = await self.observe_external()
        if not external.get("available"):
            self.pressure_telemetry_available = False
            return
        memory = await self.gpu_memory_snapshot()
        if not memory:
            self.pressure_telemetry_available = False
            return
        budget = gpu_memory_budget(memory["total_bytes"], memory["used_bytes"], external.get("external_memory_bytes"), self.config.get("external_reserve_fraction", 0.3))
        overcommitted = budget["target_llm_bytes"] < budget["llm_and_system_used_bytes"]
        rows = self.db.execute("SELECT id FROM leases WHERE state IN ('starting','active','evicting','expiring')").fetchall()
        pressured = {row["id"] for row in rows} if overcommitted else set()
        for lease_id in pressured - self.external_pressure_leases:
            self.event(lease_id, "EXTERNAL_GPU_PRESSURE", external_memory_bytes=int(external.get("external_memory_bytes") or 0), target_llm_bytes=budget["target_llm_bytes"], measured_llm_bytes=budget["llm_and_system_used_bytes"])
        for lease_id in self.external_pressure_leases - pressured:
            self.event(lease_id, "EXTERNAL_GPU_PRESSURE_CLEARED", external_memory_bytes=int(external.get("external_memory_bytes") or 0))
        self.external_pressure_leases = pressured
        self.external_pressure_active = overcommitted
        self.pressure_telemetry_available = True
        self.pressure_sampled_monotonic = time.monotonic()

    async def ensure_recent_pressure(self):
        age = time.monotonic() - getattr(self, "pressure_sampled_monotonic", 0)
        if age > max(2.5, float(self.config.get("external_monitor_interval_seconds", 1)) * 2):
            if not hasattr(self, "pressure_refresh_lock"):
                self.pressure_refresh_lock = asyncio.Lock()
            async with self.pressure_refresh_lock:
                age = time.monotonic() - getattr(self, "pressure_sampled_monotonic", 0)
                if age > max(2.5, float(self.config.get("external_monitor_interval_seconds", 1)) * 2):
                    await self.evaluate_external_pressure()
        if not getattr(self, "pressure_telemetry_available", True):
            raise self.capacity_error("telemetry_unavailable", {"operation": "inference"}, {"gpu_memory": "unknown"}, "gpu_memory_telemetry", True, [{"action": "retry_when_telemetry_is_available"}])

    async def status(self):
        await self.evaluate_external_pressure()
        external = await self.observe_external()
        memory = await self.gpu_memory_snapshot()
        model_name = next(iter(self.config.get("codex_model_names", [])), None)
        models = []
        instances = []
        if model_name:
            try:
                models.append(await self.model(model_name))
                instances = await self.instances(model_name)
            except (OSError, TimeoutError, DockerRuntimeError):
                models.append({"id": model_name, "name": model_name, "replicas": 0, "ready_replicas": 0, "error": self.runtime_error})
        capacity = []
        age = max(0.0, time.monotonic() - float(external.get("sampled_monotonic", 0))) if external.get("sampled_monotonic") else None
        if memory and external.get("available"):
            budget = gpu_memory_budget(memory["total_bytes"], memory["used_bytes"], external.get("external_memory_bytes"), self.config.get("external_reserve_fraction", 0.3))
            capacity.append({"gpu": memory["name"], "total_bytes": memory["total_bytes"], "driver_used_bytes": memory["used_bytes"], "external_process_bytes": int(external.get("external_memory_bytes") or 0), "external_reserve_bytes": budget["external_reserve_bytes"], "target_llm_budget_bytes": budget["target_llm_bytes"], "available_llm_budget_bytes": budget["available_llm_bytes"], "llm_and_system_used_bytes": budget["llm_and_system_used_bytes"], "external_reserve_satisfied": budget["target_llm_bytes"] >= budget["llm_and_system_used_bytes"], "measurement": "measured", "sample_age_seconds": age})
        calls = list(self.inference_reservations.values())
        rows = [dict(row) for row in self.db.execute("SELECT id,owner,workspace,lease_id,state,progress,error,context_length,task_summary,created,updated FROM agent_jobs WHERE state NOT IN ('completed','succeeded','failed','cancelled','preempted','interrupted') ORDER BY created").fetchall()]
        agents = []
        for row in rows:
            requests = [item for item in calls if item["lease_id"] == row["lease_id"]]
            agents.append({"id": row["id"], "owner": row["owner"], "task": row["task_summary"], "status": self.public_job_state(row["state"]), "context_tokens": row["context_length"], "updated": row["updated"], "workspace": row["workspace"], "progress": row["progress"], "error": row["error"], "cpu_state": "running" if row["state"] in ("starting", "running", "acquiring") else row["state"], "active_inference_requests": len(requests), "inference_state": "active" if requests else "idle"})
        return {"runtime": {"kind": "wsl_docker", "distribution": self.config["container_runtime"]["distribution"], "available": self.docker_available, "error": self.runtime_error}, "models": models, "instances": instances, "leases": self.active_leases(), "windows_gpu": external, "capacity_estimate": capacity, "runtime_gpu_observation_error": getattr(self, "runtime_gpu_error", None), "external_growth_pressure": {"active": self.external_pressure_active, "affected_lease_ids": sorted(self.external_pressure_leases)}, "budget_revision": getattr(self, "budget_revision", 0), "model_state": self.model_state, "requires_profile_change": False, "agents": agents, "queue": [{"id": row["id"], "owner": row["owner"], "task": row["task_summary"], "status": self.public_job_state(row["state"]), "context_tokens": row["context_length"], "created": row["created"]} for row in rows if row["state"] in ("queued", "acquiring")], "inference": {"active_requests": len(calls), "active_sequences": sum(int(item["choices"]) for item in calls), "reserved_tokens": sum(int(item["reserved_tokens"]) for item in calls)}}

    def backend_limits(self, model):
        parameters = model.get("backend_parameters") or []
        limits = {}
        for parameter in parameters:
            match = re.fullmatch(r"--(max-num-seqs|max-model-len)=(\d+)", str(parameter))
            if match:
                limits[match.group(1)] = int(match.group(2))
        return limits

    async def kv_capacity_tokens(self, model):
        spec = self.container_spec(model["name"])
        container_id = (model.get("container") or {}).get("Id")
        if container_id and container_id in self.kv_capacity_by_instance:
            return self.kv_capacity_by_instance[container_id]
        if container_id:
            row = self.db.execute("SELECT capacity_tokens FROM model_kv_cache WHERE container_id=?", (container_id,)).fetchone()
            if row:
                self.kv_capacity_by_instance[container_id] = int(row["capacity_tokens"])
                return self.kv_capacity_by_instance[container_id]
        logs = await self.docker.logs(spec["container_name"], tail=10000)
        match = re.search(r"GPU KV cache size:\s*([\d,]+) tokens", logs)
        if not match:
            raise HTTPException(503, {"code": "kv_measurement_unavailable", "requested": {"model": model["name"]}, "available": {}, "limiting_factor": "vllm_logs", "retryable": True, "alternatives": [{"action": "retry"}]})
        capacity = int(match.group(1).replace(",", ""))
        if container_id:
            self.kv_capacity_by_instance[container_id] = capacity
            self.db.execute("INSERT OR REPLACE INTO model_kv_cache(container_id,model_name,capacity_tokens,measured) VALUES(?,?,?,?)", (container_id, model["name"], capacity, time.time()))
        return capacity

    async def check_context_capacity(self, model, limits, existing, spec):
        context_length = int(spec.context_length or limits["max-model-len"])
        if context_length > int(limits["max-model-len"]):
            raise self.capacity_error("context_limit_exceeded", {"context_tokens": context_length}, {"max_context_tokens": int(limits["max-model-len"])}, "max_model_len", False, [{"max_context_tokens": int(limits["max-model-len"])}])
        return None

    def capacity_error(self, reason, requested, available, limiting_factor, retryable, alternatives):
        detail = {
            "code": reason,
            "reason": reason,
            "requested": requested,
            "available": available,
            "limiting_factor": limiting_factor,
            "retryable": retryable,
            "alternatives": alternatives,
        }
        return HTTPException(409, detail, headers={"Retry-After": "5"} if retryable else None)

    async def reserve_inference(self, lease_id, body, api, auto_output=False):
        lease = self.lease(lease_id)
        route = await self.route(lease["model_name"])
        if api == "responses":
            input_value = body.get("input")
            if isinstance(input_value, str):
                tokenize_body = {"model": lease["model_name"], "prompt": input_value, "add_generation_prompt": True}
            elif isinstance(input_value, list):
                tokenize_body = {"model": lease["model_name"], "prompt": input_value, "add_generation_prompt": True}
            else:
                raise HTTPException(422, "Responses API input must be text or a structured input list for exact token accounting")
            tokenize_url = route["base_url"] + "/tokenize"
        else:
            messages = body.get("messages")
            if not isinstance(messages, list) or not messages:
                raise HTTPException(422, "Chat inference requires a nonempty messages list for exact token accounting")
            tokenize_body = {"model": lease["model_name"], "messages": messages, "add_generation_prompt": True}
            for key in ("tools", "continue_final_message", "chat_template", "chat_template_kwargs"):
                if key in body:
                    tokenize_body[key] = body[key]
            tokenize_url = route["base_url"] + "/tokenize"
        try:
            token_response = await self.client.post(tokenize_url, json=tokenize_body, timeout=30)
            token_response.raise_for_status()
            prompt_tokens = int(token_response.json()["count"])
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(503, {"reason": "tokenizer_unavailable", "requested": {}, "available": {}, "limiting_factor": "vllm_tokenizer", "retryable": True, "alternatives": [{"action": "retry"}]}) from exc
        try:
            model = await self.model(lease["model_name"])
            kv_capacity = await self.kv_capacity_tokens(model)
        except (DockerRuntimeError, OSError, TimeoutError) as exc:
            raise HTTPException(503, {"code": "runtime_unavailable", "reason": "docker_control_unavailable", "requested": {"operation": "inference"}, "available": {"runtime": "temporarily_unavailable"}, "limiting_factor": "wsl_docker_control", "retryable": True, "alternatives": [{"action": "retry_after_runtime_recovers"}]}) from exc
        limits = self.backend_limits(model)
        lease_limit = int(lease["context_length"] or limits["max-model-len"])
        lease_limit = min(lease_limit, int(limits["max-model-len"]))
        output_value = body.get("max_completion_tokens", body.get("max_tokens"))
        try:
            output_tokens = int(output_value if output_value is not None else self.config.get("default_inference_output_tokens", 1024))
            choices = int(body.get("n", 1) or 1)
        except (TypeError, ValueError) as exc:
            raise HTTPException(422, "Output token count and choice count must be integers") from exc
        if output_tokens < 1 or choices < 1:
            raise HTTPException(422, "Inference output token count and choice count must be positive")
        if auto_output:
            output_tokens = min(output_tokens, max(1, lease_limit - prompt_tokens))
            if body.get("thinking_token_budget"):
                body["thinking_token_budget"] = min(body["thinking_token_budget"], max(1, output_tokens - min(4096, output_tokens // 2)))
        if choices > int(lease["requested_agents"]):
            raise self.capacity_error("lease_sequence_limit_exceeded", {"choices": choices}, {"lease_sequences": lease["requested_agents"]}, "lease_sequence_limit", False, [{"choices": lease["requested_agents"]}])
        request_total = prompt_tokens + output_tokens * choices
        async with self.lock:
            active = list(self.inference_reservations.values())
            reserved = sum(int(item["reserved_tokens"]) for item in active)
            active_sequences = sum(int(item["choices"]) for item in active)
            lease_sequences = sum(int(item["choices"]) for item in active if item["lease_id"] == lease_id)
            max_sequences = int(limits["max-num-seqs"])
            available_kv = max(0, kv_capacity - reserved)
            limiting_factor = None
            if prompt_tokens + output_tokens > lease_limit:
                limiting_factor = "lease_context"
            elif prompt_tokens + output_tokens > int(limits["max-model-len"]):
                limiting_factor = "engine_context"
            elif request_total > available_kv:
                limiting_factor = "measured_kv_capacity"
            elif lease_sequences + choices > int(lease["requested_agents"]):
                limiting_factor = "lease_sequence_limit"
            elif active_sequences + choices > max_sequences:
                limiting_factor = "max_num_seqs"
            if limiting_factor:
                available_output = max(0, min(lease_limit - prompt_tokens, int(limits["max-model-len"]) - prompt_tokens, available_kv - prompt_tokens))
                alternatives = []
                if available_output > 0:
                    alternatives.append({"max_output_tokens": available_output})
                if limiting_factor in ("measured_kv_capacity", "max_num_seqs"):
                    alternatives.append({"action": "retry_when_capacity_is_available"})
                raise self.capacity_error(
                    "inference_capacity_exceeded",
                    {"input_tokens": prompt_tokens, "output_tokens": output_tokens, "choices": choices, "total_tokens": request_total},
                    {"lease_context_tokens": lease_limit, "kv_capacity_tokens": kv_capacity, "reserved_kv_tokens": reserved, "available_kv_tokens": available_kv, "lease_sequences": lease["requested_agents"], "active_lease_sequences": lease_sequences, "max_num_seqs": max_sequences, "active_sequences": active_sequences},
                    limiting_factor,
                    limiting_factor in ("measured_kv_capacity", "max_num_seqs"),
                    alternatives,
                )
            if "max_completion_tokens" in body:
                body["max_completion_tokens"] = output_tokens
            else:
                body["max_tokens"] = output_tokens
            call_id = uuid.uuid4().hex
            self.inference_reservations[call_id] = {"lease_id": lease_id, "input_tokens": prompt_tokens, "output_tokens": output_tokens, "choices": choices, "reserved_tokens": request_total}
            self.revise_budget()
            return call_id, prompt_tokens, output_tokens, request_total

    def release_inference_reservation(self, call_id):
        if self.inference_reservations.pop(call_id, None) is not None:
            self.revise_budget()

    def record_inference_call(self, call_id, lease_id, model_name, api, started, elapsed, status, input_tokens, output_tokens, http_status=None):
        self.db.execute("INSERT OR IGNORE INTO inference_calls(id,lease_id,model_name,api,started,elapsed,status,http_status,input_tokens,output_tokens) VALUES(?,?,?,?,?,?,?,?,?,?)", (call_id, lease_id, model_name, api, started, elapsed, status, http_status, input_tokens, output_tokens))

    def check_model_capacity(self, model, capacity):
        if not capacity.get("windows_gpu", {}).get("available"):
            raise self.capacity_error("telemetry_unavailable", {"model": model["name"]}, {"external_vram": "unknown"}, "external_vram_telemetry", True, [{"action": "retry_when_telemetry_is_available"}])
        claim = int(self.container_spec(model["name"]).get("vram_claim_bytes") or 0)
        item = capacity.get("capacity_estimate", [None])[0]
        available = item.get("available_llm_budget_bytes") if item else None
        if available is None:
            raise self.capacity_error("telemetry_unavailable", {"vram_bytes": claim}, {"available_llm_bytes": None}, "gpu_memory_telemetry", True, [{"action": "retry_when_telemetry_is_available"}])
        if available < claim:
            raise self.capacity_error("insufficient_vram", {"vram_bytes": claim}, {"available_llm_bytes": available, "external_bytes": item["external_process_bytes"]}, "llm_vram_budget", True, [{"action": "wait_for_external_memory_to_be_released"}])

    async def acquire(self, spec):
        if spec.kind == "video":
            raise HTTPException(501, {"code": "unsupported_operation", "operation": "video", "message": "Video and ComfyUI workloads are not supported by this broker."})
        if not self.uses_wsl_docker():
            raise HTTPException(503, {"code": "runtime_unavailable", "requested": {"model": spec.model_name}, "available": {}, "limiting_factor": "runtime_backend", "retryable": False, "alternatives": []})
        return await self.acquire_wsl_docker(spec)

    async def acquire_wsl_docker(self, spec):
        if spec.kind == "video":
            raise HTTPException(501, {"code": "unsupported_operation", "operation": "video", "message": "Video and ComfyUI workloads are not supported by this broker."})
        external = await self.observe_external()
        if not external.get("available"):
            raise self.capacity_error("telemetry_unavailable", {"operation": "agent_start"}, {"external_vram": "unknown"}, "external_vram_telemetry", True, [{"action": "retry_when_telemetry_is_available"}])
        await self.ensure_recent_pressure()
        if self.external_pressure_active:
            raise self.capacity_error("insufficient_vram", {"operation": "agent_start"}, {"available_llm_bytes": 0}, "llm_vram_budget", True, [{"action": "wait_for_external_memory_to_be_released"}])
        model_name = spec.model_name
        codex_models = self.config.get("codex_model_names", [])
        if not model_name or model_name not in self.config.get("managed_models", []) or (codex_models and model_name not in codex_models):
            raise HTTPException(422, {"code": "model_unavailable", "requested": {"model": model_name}, "available": {"models": codex_models}, "limiting_factor": "model_configuration", "retryable": False, "alternatives": []})
        model = await self.model(model_name)
        limits = self.backend_limits(model)
        if "max-num-seqs" not in limits or "max-model-len" not in limits:
            raise self.capacity_error("model_limit_unknown", {"model": model_name}, {"engine_limits": limits}, "vllm_engine_limits", False, [])
        context_length = int(spec.context_length or limits["max-model-len"])
        if context_length > limits["max-model-len"]:
            raise self.capacity_error("context_limit_exceeded", {"context_tokens": context_length}, {"max_context_tokens": limits["max-model-len"]}, "max_model_len", False, [{"max_context_tokens": limits["max-model-len"]}])
        requested_agents = int(spec.requested_agents)
        if requested_agents > int(limits["max-num-seqs"]):
            raise self.capacity_error("sequence_limit_exceeded", {"agents": requested_agents}, {"max_num_seqs": int(limits["max-num-seqs"])}, "max_num_seqs", False, [{"max_agents": int(limits["max-num-seqs"])}])
        async with self.lock:
            if getattr(self, "draining", False):
                raise self.capacity_error("service_draining", {"operation": "agent_start"}, {"service_state": "draining"}, "service_state", True, [{"action": "retry_after_drain"}])
            self.sweep()
            current = self.active_leases("agent")
            if len(current) >= self.max_agent_leases:
                raise self.capacity_error("agent_queue_full", {"agents": 1}, {"active_agents": len(current), "agent_limit": self.max_agent_leases}, "agent_queue", True, [{"action": "retry_after_capacity_is_available"}])
            if any(lease["state"] in ("evicting", "expiring") for lease in current):
                raise self.capacity_error("waiting_for_release", {"agents": 1}, {"releasing_leases": len(current)}, "lease_transition", True, [{"action": "retry_after_release"}])
            if any(lease["state"] == "starting" for lease in current):
                raise self.capacity_error("model_loading", {"model": model_name}, {"model_state": "loading"}, "model_state", True, [{"action": "retry_after_model_is_ready"}])
            if any(lease["model_name"] != model_name for lease in current):
                raise self.capacity_error("model_in_use", {"model": model_name}, {"active_models": sorted({lease["model_name"] for lease in current})}, "single_model_runtime", True, [{"action": "wait_for_active_jobs"}])
            if not current and not model.get("replicas"):
                self.check_model_capacity(model, await self.status())
            lease_id = uuid.uuid4().hex
            now = time.time()
            self.db.execute("INSERT INTO leases VALUES(?,?,?,?,?,?,?,?,?,?)", (lease_id, spec.owner, "agent", model_name, requested_agents, context_length, now + spec.ttl_seconds, "starting", now, now))
            self.event(lease_id, "LEASE_REQUESTED", owner=spec.owner, kind="agent", model=model_name, requested_agents=requested_agents, context_length=context_length)
            self.revise_budget()
        started_model = False
        try:
            async with self.model_start_lock:
                model = await self.model(model_name)
                if not model.get("ready_replicas"):
                    if not model.get("replicas"):
                        started_model = True
                        await self.set_replicas(model, 1)
                    else:
                        self.model_state = "loading"
                    await self.wait_model(model_name, True)
                self.model_state = "ready"
            if model.get("ready_replicas"):
                await self.ensure_recent_pressure()
                if self.external_pressure_active:
                    raise self.capacity_error("insufficient_vram", {"model": model_name}, {"external_pressure": True}, "llm_vram_budget", True, [{"action": "wait_for_external_memory_to_be_released"}])
            else:
                status = await self.status()
                capacity = status["capacity_estimate"][0] if status["capacity_estimate"] else None
                if not capacity or not capacity["external_reserve_satisfied"]:
                    raise self.capacity_error("insufficient_vram", {"model": model_name}, {"gpu_budget": capacity}, "llm_vram_budget", True, [{"action": "wait_for_external_memory_to_be_released"}])
            external = await self.observe_external()
            if not external.get("available"):
                raise self.capacity_error("telemetry_unavailable", {"operation": "agent_start"}, {"external_vram": "unknown"}, "external_vram_telemetry", True, [{"action": "retry_when_telemetry_is_available"}])
            async with self.lock:
                current = self.lease(lease_id)
                if current["state"] != "starting" or current["expires"] <= time.time():
                    raise self.capacity_error("lease_expired_during_start", {"lease_id": lease_id}, {"lease_state": current["state"]}, "lease_state", False, [])
                self.db.execute("UPDATE leases SET state='active',updated=? WHERE id=?", (time.time(), lease_id))
                self.event(lease_id, "LEASE_GRANTED", external_bytes=int(external.get("external_memory_bytes") or 0))
                self.revise_budget()
            self.last_model_activity = time.monotonic()
            return {"lease_id": lease_id, "state": "active", "kind": "agent", "model": model_name, "requested_agents": requested_agents, "requested_context_length": context_length, "backend_limits": limits, "windows_gpu_observation": external, "inference_url": f"{self.config['public_url']}/v1/leases/{lease_id}/chat/completions", "lease_header": "X-GPU-Lease", "expires_at": now + spec.ttl_seconds}
        except BaseException as exc:
            async with self.lock:
                current = self.lease(lease_id)
                if current["state"] in ("starting", "active"):
                    self.db.execute("UPDATE leases SET state='failed',updated=? WHERE id=?", (time.time(), lease_id))
                    self.event(lease_id, "LEASE_FAILED", kind="agent")
                    self.revise_budget()
            if started_model:
                async with self.model_start_lock:
                    if not self.active_leases() and not self.calls:
                        with contextlib.suppress(Exception):
                            container = await self.docker.inspect(self.container_spec(model_name)["container_name"])
                            label = self.config["container_runtime"]["managed_label"]
                            if container and (container.get("Config") or {}).get("Labels", {}).get(label) == "true":
                                await self.set_replicas(await self.model(model_name), 0)
                                await self.wait_model(model_name, False)
            detail = exc.detail if isinstance(exc, HTTPException) and isinstance(exc.detail, dict) else {}
            if detail.get("code") == "model_start_failed":
                self.model_state = "error"
            else:
                with contextlib.suppress(Exception):
                    loaded = await self.model(model_name)
                    self.model_state = "ready" if loaded.get("ready_replicas") else "idle"
            raise

    async def release(self, lease_id):
        async with self.lock:
            row = self.lease(lease_id)
            if row["state"] not in ("active", "starting", "evicting", "expiring"):
                return row
            if any(request["lease_id"] == lease_id for request in self.calls.values()):
                raise HTTPException(409, "GPU lease still has in-flight requests")
            other_model_leases = [lease for lease in self.active_leases() if lease["model_name"] == row["model_name"] and lease["id"] != lease_id and lease["state"] != "expiring"] if row["model_name"] else []
            stop_model = False
            self.db.execute("UPDATE leases SET state='expiring',updated=? WHERE id=?", (time.time(), lease_id))
        async with self.lock:
            self.db.execute("UPDATE leases SET state='released',updated=? WHERE id=?", (time.time(), lease_id))
            self.event(lease_id, "LEASE_RELEASED")
            if not self.active_leases():
                self.last_model_activity = time.monotonic()
            self.revise_budget()
            return self.lease(lease_id)

    async def renew(self, lease_id, seconds):
        async with self.lock:
            row = self.lease(lease_id)
            if row["state"] != "active":
                raise HTTPException(409, "Lease is no longer active")
            expiry = time.time() + seconds
            self.db.execute("UPDATE leases SET expires=?,updated=? WHERE id=?", (expiry, time.time(), lease_id))
            self.event(lease_id, "LEASE_RENEWED", ttl_seconds=seconds)
            return {**self.lease(lease_id), "expires_at": expiry}

    def events_since(self, lease_id, after=0):
        if lease_id:
            rows = self.db.execute("SELECT * FROM events WHERE lease_id=? AND id>? ORDER BY id LIMIT 100", (lease_id, after)).fetchall()
        else:
            rows = self.db.execute("SELECT * FROM events WHERE id>? ORDER BY id LIMIT 100", (after,)).fetchall()
        return [{"id": r["id"], "lease_id": r["lease_id"], "event": r["event"], "payload": json.loads(r["payload"]), "created": r["created"]} for r in rows]

    async def proxy(self, lease_id, body, request, api="chat/completions"):
        row = self.lease(lease_id)
        if row["kind"] != "agent" or row["state"] != "active":
            raise HTTPException(409, "Agent lease is not active for inference")
        started_at = time.time()
        started_mono = time.monotonic()
        task = asyncio.current_task()
        reservation_id = None
        call_id = uuid.uuid4().hex
        self.calls[task] = {"lease_id": lease_id, "call_id": call_id, "reservation_id": reservation_id}
        try:
            external = await self.observe_external()
            if not external.get("available"):
                raise self.capacity_error("telemetry_unavailable", {"operation": "inference"}, {"external_vram": "unknown"}, "external_vram_telemetry", True, [{"action": "retry_when_telemetry_is_available"}])
            await self.ensure_recent_pressure()
            if self.external_pressure_active:
                raise self.capacity_error("insufficient_vram", {"operation": "inference"}, {"available_llm_bytes": 0}, "llm_vram_budget", True, [{"action": "wait_for_external_memory_to_be_released"}])
            if body.get("model") not in (None, row["model_name"]):
                raise HTTPException(403, "The requested model does not match the agent lease")
            body["model"] = row["model_name"]
            auto_output = api == "chat/completions" and "max_tokens" not in body and "max_completion_tokens" not in body
            if auto_output:
                body["max_tokens"] = getattr(self, "agent_output_limits", {}).get(lease_id) or default_output_tokens(self.config.get("default_inference_output_tokens", 8192), row["context_length"])
            reasoning_level = body.pop("reasoning_level", None)
            if reasoning_level is None:
                reasoning_level = getattr(self, "agent_reasoning_levels", {}).get(lease_id)
            if reasoning_level is not None:
                apply_reasoning_level(body, reasoning_level, api)
                if lease_id in getattr(self, "agent_output_limits", {}) and body.get("thinking_token_budget"):
                    body["thinking_token_budget"] = min(body["thinking_token_budget"], 12288)
            reservation_id, input_tokens, output_tokens, reserved_tokens = await self.reserve_inference(lease_id, body, api, auto_output=auto_output and lease_id in getattr(self, "agent_output_limits", {}))
            call_id = reservation_id
            self.calls[task] = {"lease_id": lease_id, "call_id": call_id, "reservation_id": reservation_id}
        except BaseException:
            self.calls.pop(task, None)
            if reservation_id:
                self.release_inference_reservation(reservation_id)
            raise
        try:
            endpoint = await self.route(row["model_name"])
            suffix = "/v1/responses" if api == "responses" else "/v1/chat/completions"
            upstream = await self.client.send(self.client.build_request("POST", endpoint["base_url"] + suffix, json=body, timeout=httpx.Timeout(1800, connect=5)), stream=True)
            if upstream.status_code >= 400:
                content = await upstream.aread()
                await upstream.aclose()
                self.calls.pop(task, None)
                self.release_inference_reservation(reservation_id)
                self.record_inference_call(call_id, lease_id, row["model_name"], api, started_at, time.monotonic() - started_mono, "error", input_tokens, None, upstream.status_code)
                return JSONResponse(status_code=upstream.status_code, content=json.loads(content) if content.startswith(b"{") else {"error": content.decode("utf-8", "replace")[:1000]})

            async def stream():
                stream_task = asyncio.current_task()
                call = self.calls.pop(task, None)
                if call:
                    self.calls[stream_task] = call
                event_buffer = b""
                response_buffer = bytearray()
                actual_input = input_tokens
                actual_output = None
                finished = False
                try:
                    if self.lease(lease_id)["state"] != "active":
                        return
                    async for chunk in upstream.aiter_bytes():
                        if body.get("stream"):
                            event_buffer += chunk
                            if len(event_buffer) > 65536:
                                event_buffer = event_buffer[-65536:]
                            while b"\n\n" in event_buffer:
                                event, event_buffer = event_buffer.split(b"\n\n", 1)
                                for line in event.splitlines():
                                    if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                                        continue
                                    with contextlib.suppress(ValueError, TypeError, KeyError):
                                        usage = json.loads(line[5:]).get("usage")
                                        if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
                                            actual_output = int(usage["completion_tokens"])
                                            actual_input = int(usage.get("prompt_tokens", actual_input))
                        elif len(response_buffer) <= 2097152:
                            response_buffer.extend(chunk)
                        yield chunk
                    finished = True
                finally:
                    try:
                        await upstream.aclose()
                    finally:
                        self.calls.pop(stream_task, None)
                        self.release_inference_reservation(reservation_id)
                        if finished and not body.get("stream") and len(response_buffer) <= 2097152:
                            with contextlib.suppress(ValueError, TypeError, KeyError):
                                usage = json.loads(response_buffer).get("usage")
                                if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
                                    actual_output = int(usage["completion_tokens"])
                                    actual_input = int(usage.get("prompt_tokens", actual_input))
                        try:
                            lease_state = self.lease(lease_id)["state"]
                        except HTTPException:
                            lease_state = "expired"
                        status = "completed" if finished and lease_state == "active" else "interrupted"
                        self.record_inference_call(call_id, lease_id, row["model_name"], api, started_at, time.monotonic() - started_mono, status, actual_input, actual_output, upstream.status_code)

            headers = {key: value for key, value in upstream.headers.items() if key.lower() in ("x-request-id", "x-ratelimit-limit", "x-ratelimit-remaining")}
            return StreamingResponse(stream(), status_code=upstream.status_code, media_type=upstream.headers.get("content-type", "application/json"), headers=headers)
        except BaseException:
            self.calls.pop(task, None)
            self.release_inference_reservation(reservation_id)
            self.record_inference_call(call_id, lease_id, row["model_name"], api, started_at, time.monotonic() - started_mono, "interrupted", input_tokens, None)
            raise

    def agent_job(self, agent_id):
        row = self.db.execute("SELECT * FROM agent_jobs WHERE id=?", (agent_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "Unknown GPU agent")
        return dict(row)

    def update_agent_job(self, agent_id, **fields):
        fields["updated"] = time.time()
        assignments = ",".join(f"{key}=?" for key in fields)
        self.db.execute(f"UPDATE agent_jobs SET {assignments} WHERE id=?", (*fields.values(), agent_id))
        row = self.agent_job(agent_id)
        self.job_event(agent_id, row["state"], row.get("progress"))

    async def start_agent(self, owner, task_prompt, workspace, context_length=None, files=None, test_command=None, idempotency_key=None, reasoning_level=None):
        if getattr(self, "draining", False):
            raise self.capacity_error("service_draining", {"operation": "agent_start"}, {"service_state": "draining"}, "service_state", True, [{"action": "retry_after_drain"}])
        model_name = next(iter(self.config.get("codex_model_names") or []), None)
        if not model_name or model_name not in self.config.get("managed_models", []):
            raise HTTPException(503, {"code": "model_unavailable", "requested": {"model": model_name}, "available": {}, "limiting_factor": "model_configuration", "retryable": False, "alternatives": []})
        model = await self.model(model_name)
        limits = self.backend_limits(model)
        model_context_limit = limits.get("max-model-len")
        if not model_context_limit:
            raise HTTPException(503, {"code": "model_limit_unknown", "requested": {"model": model_name}, "available": {}, "limiting_factor": "max_model_len", "retryable": False, "alternatives": []})
        context_length = int(model_context_limit if context_length is None else context_length)
        if context_length < 2048 or context_length > model_context_limit:
            raise self.capacity_error("context_limit_exceeded", {"context_tokens": context_length}, {"max_context_tokens": model_context_limit}, "max_model_len", False, [{"max_context_tokens": model_context_limit}])
        reasoning_level = normalize_reasoning_level(reasoning_level, self.config.get("default_reasoning_level", "xhigh"))
        if test_command is not None and (not test_command.strip() or len(test_command) > 500 or "\n" in test_command or "\r" in test_command):
            raise HTTPException(422, "test_command must be a single nonempty command of at most 500 characters")
        root = Path(workspace).resolve()
        allowed = [Path(value).resolve() for value in self.config.get("allowed_workspace_roots", [])]
        if not root.is_dir() or not any(root == value or root.is_relative_to(value) for value in allowed):
            raise HTTPException(422, "Workspace is outside the configured agent roots")
        try:
            git_root = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"], capture_output=True, text=True, timeout=8, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise HTTPException(422, "Agent workspace must be an available Git repository") from exc
        if git_root.returncode != 0 or not git_root.stdout.strip():
            raise HTTPException(422, "Agent workspace must be a Git repository")
        repo_root = str(Path(git_root.stdout.strip()).resolve())
        selected_files = []
        for value in files or []:
            candidate = (root / value).resolve()
            if not candidate.is_file() or not candidate.is_relative_to(Path(repo_root)):
                raise HTTPException(422, f"Agent file must exist inside the Git repository: {value}")
            selected_files.append(str(candidate.relative_to(Path(repo_root))))
        if len(selected_files) > 20:
            raise HTTPException(422, "At most 20 agent files may be supplied")
        idempotency_key = str(idempotency_key or uuid.uuid4().hex).strip()
        if not idempotency_key or len(idempotency_key) > 200:
            raise HTTPException(422, "idempotency_key must contain 1 to 200 characters")
        request_hash = hashlib.sha256(json.dumps({"owner": owner, "task": task_prompt.strip(), "workspace": repo_root, "context_length": context_length, "files": selected_files, "test_command": test_command.strip() if test_command else None, "reasoning_level": reasoning_level}, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        worktree_root = Path(self.config.get("agent_worktree_root") or ROOT / "agent-worktrees")
        worktree_root.mkdir(parents=True, exist_ok=True)
        free_bytes = shutil.disk_usage(worktree_root).free
        if free_bytes < 5 * 1073741824:
            raise self.capacity_error("insufficient_workspace_disk", {"minimum_bytes": 5 * 1073741824}, {"free_bytes": free_bytes}, "worktree_storage", False, [])
        now = time.time()
        async with self.lock:
            existing = self.db.execute("SELECT * FROM agent_jobs WHERE idempotency_key=?", (idempotency_key,)).fetchone()
            if existing:
                row = dict(existing)
                if row.get("request_hash") != request_hash:
                    raise HTTPException(409, {"code": "idempotency_key_reused", "requested": {"idempotency_key": idempotency_key}, "available": {"job_id": row["id"]}, "limiting_factor": "idempotency_key", "retryable": False, "alternatives": [{"action": "use_a_new_idempotency_key"}]})
                return {"agent_id": row["id"], "job_id": row["id"], "state": self.public_job_state(row["state"]), "idempotency_key": idempotency_key, "deduplicated": True, "model": model_name, "context_length": row.get("context_length"), "model_context_limit": model_context_limit}
            agent_id = uuid.uuid4().hex
            self.db.execute("INSERT INTO agent_jobs(id,owner,workspace,state,created,updated,idempotency_key,request_hash,context_length,task_summary) VALUES(?,?,?,?,?,?,?,?,?,?)", (agent_id, owner, repo_root, "queued", now, now, idempotency_key, request_hash, context_length, task_prompt.strip()[:240]))
            self.agent_tasks[agent_id] = asyncio.create_task(self.run_agent(agent_id, task_prompt.strip(), repo_root, model_name, context_length, selected_files, test_command.strip() if test_command else None, reasoning_level))
        self.job_event(agent_id, "queued", "accepted")
        return {"agent_id": agent_id, "job_id": agent_id, "state": "queued", "idempotency_key": idempotency_key, "deduplicated": False, "model": model_name, "context_length": context_length, "reasoning_level": reasoning_level, "model_context_limit": model_context_limit, "workspace": repo_root, "task": task_prompt.strip()[:240], "constraints": {"max_context_tokens": model_context_limit, "max_input_files": 20, "uses_isolated_worktree": True}}

    def public_job_state(self, state):
        return {"queued": "waiting_capacity", "acquiring": "waiting_capacity", "starting": "running", "running": "running", "cancel_requested": "cancelling", "completed": "succeeded", "preempted": "interrupted"}.get(state, state)

    def job_event(self, job_id, state, progress=None):
        self.event(None, "AGENT_JOB_UPDATED", job_id=job_id, state=self.public_job_state(state), progress=progress)

    async def capture_agent_stream(self, stream, error_tail):
        async for line in stream:
            error_tail[0] = (error_tail[0] + line.decode("utf-8", "replace"))[-4000:]

    async def terminate_agent_process(self, process):
        if not process or process.returncode is not None:
            return
        try:
            killer = await asyncio.create_subprocess_exec("taskkill.exe", "/PID", str(process.pid), "/T", "/F", stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            await asyncio.wait_for(killer.wait(), timeout=10)
        except (OSError, asyncio.TimeoutError):
            with contextlib.suppress(ProcessLookupError):
                process.kill()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(process.wait(), timeout=5)

    def stop_orphan_agent_process(self, pid, agent_id):
        if not pid or int(pid) <= 0:
            return False
        try:
            result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", f"(Get-CimInstance Win32_Process -Filter 'ProcessId={int(pid)}').CommandLine"], capture_output=True, text=True, timeout=6, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            command_line = result.stdout.strip()
            if result.returncode != 0 or agent_id not in command_line or not (".model-metadata.json" in command_line or "CodexBrokerAgentId" in command_line):
                return False
            stopped = subprocess.run(["taskkill.exe", "/PID", str(int(pid)), "/T", "/F"], capture_output=True, text=True, timeout=10, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            return stopped.returncode == 0
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return False

    async def run_agent(self, agent_id, task_prompt, repo_root, model_name, context_length, selected_files, test_command, reasoning_level):
        lease_id = None
        process = None
        worktree_path = None
        branch = None
        capture_tasks = []
        error_tail = [""]
        lease_ttl = int(self.config.get("agent_lease_ttl_seconds", 300))
        queued_at = time.monotonic()
        try:
            self.update_agent_job(agent_id, state="acquiring", progress="requesting GPU lease")
            while True:
                self.capacity_changed.clear()
                try:
                    lease = await self.acquire(LeaseRequest(owner=f"Codex agent {agent_id[:8]}", kind="agent", model_name=model_name, requested_agents=1, context_length=context_length, ttl_seconds=lease_ttl))
                    break
                except HTTPException as exc:
                    detail = exc.detail if isinstance(exc.detail, dict) else {}
                    retryable = retryable_acquisition_error(exc)
                    queue_limit = float(self.config.get("agent_max_queue_wait_seconds", 21600))
                    if not self.uses_wsl_docker() or not retryable or time.monotonic() - queued_at >= queue_limit:
                        raise
                    progress = detail.get("reason") or "waiting_for_gpu_capacity"
                    self.update_agent_job(agent_id, state="queued", progress=f"waiting for broker capacity: {progress}")
                    try:
                        await asyncio.wait_for(self.capacity_changed.wait(), timeout=min(float(self.config.get("agent_queue_retry_seconds", 5)), max(0.5, queue_limit - (time.monotonic() - queued_at))))
                    except asyncio.TimeoutError:
                        pass
                    self.update_agent_job(agent_id, state="acquiring", progress="retrying GPU lease")
            lease_id = lease["lease_id"]
            if not hasattr(self, "agent_reasoning_levels"):
                self.agent_reasoning_levels = {}
            self.agent_reasoning_levels[lease_id] = reasoning_level
            self.agent_leases[agent_id] = lease_id
            worktree_path = Path(self.config.get("agent_worktree_root") or ROOT / "agent-worktrees") / agent_id
            worktree_path.parent.mkdir(parents=True, exist_ok=True)
            branch = f"codex-agent-{agent_id}"
            self.update_agent_job(agent_id, lease_id=lease_id, state="starting", progress="creating isolated git worktree", worktree_path=str(worktree_path), branch=branch)
            added = await asyncio.create_subprocess_exec("git", "-C", repo_root, "worktree", "add", "-b", branch, str(worktree_path), "HEAD", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            add_stdout, add_stderr = await asyncio.wait_for(added.communicate(), timeout=180)
            if added.returncode != 0:
                raise RuntimeError((add_stderr or add_stdout).decode("utf-8", "replace")[-2000:] or "Could not create agent worktree")
            executable = self.config.get("aider_executable") or shutil.which("aider")
            if not executable or not Path(executable).is_file():
                raise RuntimeError("Aider executable is unavailable")
            result_dir = ROOT / "results"
            result_dir.mkdir(exist_ok=True)
            output_path = result_dir / f"{agent_id}.history.md"
            input_history_path = result_dir / f"{agent_id}.input.history"
            metadata_path = result_dir / f"{agent_id}.model-metadata.json"
            metadata = json.loads((ROOT / "aider-model-metadata.json").read_text(encoding="utf-8"))
            model_metadata = metadata[f"openai/{model_name}"]
            output_limit = min(int(model_metadata["max_output_tokens"]), max(256, context_length // 2))
            model_metadata.update(max_tokens=context_length, max_input_tokens=context_length - output_limit, max_output_tokens=output_limit)
            if not hasattr(self, "agent_output_limits"):
                self.agent_output_limits = {}
            self.agent_output_limits[lease_id] = output_limit
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            base_url = f"{self.config['public_url'].rstrip('/')}/v1/leases/{lease_id}"
            ask_mode = task_prompt.startswith("[ask] ")
            task_text = task_prompt[6:].strip() if ask_mode else task_prompt
            instructions = "You are a local Qwen coding subagent in an isolated Git worktree. Complete only the assigned task. Do not launch another agent or GPU process. Give a concise summary of changes and checks. Task: " + task_text
            args = [executable, "--model", f"openai/{model_name}", "--weak-model", f"openai/{model_name}", "--model-settings-file", str(ROOT / "aider-model-settings.yml"), "--model-metadata-file", str(metadata_path), "--edit-format", "ask" if ask_mode else "diff", "--map-tokens", "0", "--no-auto-commits", "--yes-always", "--no-stream", "--no-pretty", "--no-check-update", "--no-show-model-warnings", "--no-suggest-shell-commands", "--no-gitignore", "--chat-history-file", str(output_path), "--input-history-file", str(input_history_path), "--openai-api-base", base_url, "--openai-api-key", "local-codex-broker-lease", "--message", instructions]
            if test_command:
                args.extend(["--test-cmd", test_command, "--auto-test"])
            for value in selected_files:
                args.extend(["--file", str(worktree_path / value)])
            agent_env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
            process = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=str(worktree_path), env=agent_env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self.agent_processes[agent_id] = process
            self.update_agent_job(agent_id, pid=process.pid, state="running", progress="Aider Qwen agent running")
            capture_tasks = [asyncio.create_task(self.capture_agent_stream(process.stdout, error_tail)), asyncio.create_task(self.capture_agent_stream(process.stderr, error_tail))]
            waiter = asyncio.create_task(process.wait())
            max_runtime = float(self.config.get("agent_max_runtime_seconds", 21600))
            started = time.monotonic()
            while not waiter.done():
                remaining = max_runtime - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("Agent runtime limit reached")
                done, _ = await asyncio.wait({waiter}, timeout=min(float(self.config.get("agent_renew_interval_seconds", 120)), remaining))
                if done:
                    break
                await self.renew(lease_id, lease_ttl)
            return_code = await waiter
            await asyncio.gather(*capture_tasks)
            history = output_path.read_text(encoding="utf-8", errors="replace") if output_path.exists() else ""
            lines = history.rsplit("####", 1)[-1].splitlines()[1:] if "####" in history else []
            while lines and (not lines[0].strip() or lines[0].lstrip().startswith(">")):
                lines.pop(0)
            result = "\n".join(line for line in lines[:next((i for i, line in enumerate(lines) if line.lstrip().startswith("> Tokens:")), len(lines))]).strip()
            has_edit = ask_mode or bool(self.agent_diff(self.agent_job(agent_id)))
            row = self.lease(lease_id)
            test_error = None
            if return_code == 0 and result and has_edit and test_command and row["state"] == "active":
                check = await asyncio.create_subprocess_exec("powershell.exe", "-NoProfile", "-NonInteractive", "-Command", f"$CodexBrokerAgentId='{agent_id}'; & {{ {test_command} }}", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=str(worktree_path), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                process = check
                self.agent_processes[agent_id] = check
                self.update_agent_job(agent_id, pid=check.pid)
                try:
                    check_stdout, check_stderr = await asyncio.wait_for(check.communicate(), timeout=300)
                    output = (check_stdout + check_stderr).decode("utf-8", "replace")[-1500:]
                    test_error = f"Final test failed (exit {check.returncode}): {output}" if check.returncode else None
                    result += f"\n\nFinal test: {'passed' if check.returncode == 0 else 'failed'} — {test_command}"
                except asyncio.TimeoutError:
                    await self.terminate_agent_process(check)
                    test_error = "Final test timed out after 300 seconds"
            if self.agent_job(agent_id)["state"] == "cancel_requested":
                self.update_agent_job(agent_id, state="cancelled", progress="cancelled", result=result, error=None)
            elif return_code == 0 and result and has_edit and not test_error and "litellm." not in history and "has hit a token limit!" not in history:
                self.update_agent_job(agent_id, state="completed", progress="Aider Qwen agent finished", result=result, error=None)
            elif row["state"] in ("evicting", "evicted"):
                self.update_agent_job(agent_id, state="preempted", progress="owner preempted this agent", result="GPU lease was force-preempted by the owner. Stop this task and continue without the local GPU.\n\n" + result)
            else:
                if "has hit a token limit!" in history:
                    reason = f"Aider response reached its configured {output_limit:,}-token output budget" if "possibly exceeded output limit!" in history else f"Aider request reached its configured {context_length - output_limit:,}-token input budget"
                elif "estimated chat context" in history and "exceeds the" in history:
                    reason = f"Aider estimated input above its configured {context_length - output_limit:,}-token budget"
                else:
                    reason = "Aider produced no answer or code changes" if not result or not has_edit else error_tail[0] or f"aider exit code {return_code}"
                reason = test_error or reason
                self.update_agent_job(agent_id, state="failed", progress="Aider Qwen agent exited with an error", result=result, error=reason[:1000])
        except asyncio.CancelledError:
            await self.terminate_agent_process(process)
            row = self.lease(lease_id) if lease_id else None
            state = "preempted" if row and row["state"] in ("evicting", "evicted") else "failed" if row and row["state"] in ("expiring", "expired") else "cancelled"
            message = "GPU lease expired before this Qwen agent finished" if state == "failed" else state
            self.update_agent_job(agent_id, state=state, progress=message, error=message if state == "failed" else None)
            raise
        except BaseException as exc:
            await self.terminate_agent_process(process)
            self.update_agent_job(agent_id, state="failed", progress="Aider Qwen agent failed", error=str(exc)[:10000])
        finally:
            for task in capture_tasks:
                if not task.done():
                    task.cancel()
            if capture_tasks:
                await asyncio.gather(*capture_tasks, return_exceptions=True)
            self.agent_processes.pop(agent_id, None)
            self.agent_leases.pop(agent_id, None)
            if lease_id:
                getattr(self, "agent_reasoning_levels", {}).pop(lease_id, None)
                getattr(self, "agent_output_limits", {}).pop(lease_id, None)
            if lease_id:
                with contextlib.suppress(Exception):
                    row = self.lease(lease_id)
                    if row["state"] in ("active", "starting"):
                        await self.release(lease_id)
            if worktree_path and worktree_path.is_dir() and branch:
                with contextlib.suppress(Exception):
                    status = subprocess.run(["git", "-C", str(worktree_path), "status", "--porcelain"], capture_output=True, text=True, timeout=30, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    if status.returncode == 0 and not status.stdout.strip():
                        removed = subprocess.run(["git", "-C", repo_root, "worktree", "remove", str(worktree_path)], capture_output=True, text=True, timeout=60, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                        if removed.returncode == 0:
                            subprocess.run(["git", "-C", repo_root, "branch", "-d", branch], capture_output=True, text=True, timeout=10, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                            self.update_agent_job(agent_id, worktree_path=None, branch=None)
            self.agent_tasks.pop(agent_id, None)

    async def poll_agent(self, agent_id, wait_seconds=0):
        if wait_seconds < 0 or wait_seconds > 30:
            raise HTTPException(422, "wait_seconds must be between 0 and 30")
        row = self.agent_job(agent_id)
        terminal = ("completed", "failed", "cancelled", "preempted", "interrupted")
        deadline = time.monotonic() + wait_seconds
        while wait_seconds and row["state"] not in terminal and time.monotonic() < deadline:
            await asyncio.sleep(min(1, deadline - time.monotonic()))
            row = self.agent_job(agent_id)
        return {"job_id": row["id"], "agent_id": row["id"], "state": self.public_job_state(row["state"]), "raw_state": row["state"], "progress": row["progress"], "result": row["result"], "diff": self.agent_diff(row), "tests": row["result"] if row["result"] and "Final test:" in row["result"] else None, "error": row["error"], "worktree_path": row["worktree_path"], "branch": row["branch"], "owner": row["owner"], "task": row.get("task_summary"), "context_length": row.get("context_length"), "created": row["created"], "updated": row["updated"]}

    def agent_diff(self, row):
        worktree = row.get("worktree_path")
        if not worktree or not Path(worktree).is_dir():
            return None
        try:
            result = subprocess.run(["git", "-C", worktree, "diff", "--stat"], capture_output=True, text=True, timeout=10, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            return result.stdout[-4000:] if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None

    async def wait_events(self, after_cursor=0, wait_seconds=20):
        if after_cursor < 0 or wait_seconds < 0 or wait_seconds > 30:
            raise HTTPException(422, "after_cursor must be nonnegative and wait_seconds must be between 0 and 30")
        deadline = time.monotonic() + wait_seconds
        while True:
            events = self.events_since(None, after_cursor)
            if events or time.monotonic() >= deadline:
                return {"events": events, "cursor": events[-1]["id"] if events else after_cursor}
            await asyncio.sleep(min(0.2, deadline - time.monotonic()))

    async def unload_model(self, force=False, timeout=300):
        async with self.lock:
            previous_state = self.model_state
            self.draining = True
            self.model_state = "draining"
        try:
            if force:
                rows = self.db.execute("SELECT id FROM agent_jobs WHERE state NOT IN ('completed','failed','cancelled','preempted','interrupted')").fetchall()
                for row in rows:
                    await self.cancel_agent(row["id"])
                for task in list(batch_manager.tasks.values()):
                    task.cancel()
                if batch_manager.tasks:
                    await asyncio.gather(*list(batch_manager.tasks.values()), return_exceptions=True)
            deadline = time.monotonic() + max(0, timeout) if not force else time.monotonic()
            while True:
                rows = self.db.execute("SELECT id FROM agent_jobs WHERE state NOT IN ('completed','failed','cancelled','preempted','interrupted')").fetchall()
                leases = self.active_leases()
                if not rows and not leases and not self.calls and not batch_manager.tasks:
                    break
                if time.monotonic() >= deadline:
                    return {"state": "busy", "unloaded": False, "active_jobs": [row["id"] for row in rows], "active_leases": [lease["id"] for lease in leases], "active_batches": list(batch_manager.tasks), "active_requests": len(self.calls)}
                await asyncio.sleep(0.2)
            name = next(iter(self.config.get("codex_model_names", [])), None)
            if not name or not self.uses_wsl_docker():
                self.model_state = "idle"
                return {"state": "idle", "unloaded": True}
            probe = getattr(self, "startup_probe_task", None)
            if probe and not probe.done():
                probe.cancel()
                await asyncio.gather(probe, return_exceptions=True)
            async with self.model_start_lock:
                model = await self.model(name)
                if model.get("replicas", 0):
                    self.model_state = "unloading"
                    await self.set_replicas(model, 0)
                    await self.wait_model(name, False)
                    self.model_state = "idle"
            self.last_model_activity = time.monotonic()
            self.revise_budget()
            return {"state": "idle", "unloaded": True, "force": force}
        finally:
            self.draining = False
            if self.model_state == "draining":
                self.model_state = previous_state
            elif self.model_state == "unloading":
                self.model_state = "ready" if self.model_ready else "error"

    async def cancel_agent(self, agent_id):
        row = self.agent_job(agent_id)
        if row["state"] in ("completed", "failed", "cancelled", "preempted", "interrupted"):
            return {"agent_id": agent_id, "state": row["state"]}
        self.update_agent_job(agent_id, state="cancel_requested", progress="cancelling")
        lease_id = row["lease_id"]
        if lease_id:
            for request_task, call in list(self.calls.items()):
                if call["lease_id"] == lease_id:
                    request_task.cancel()
        await self.terminate_agent_process(self.agent_processes.get(agent_id))
        deadline = time.monotonic() + 20
        while lease_id and any(call["lease_id"] == lease_id for call in self.calls.values()) and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        task = self.agent_tasks.get(agent_id)
        if task:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        return {"agent_id": agent_id, "state": self.agent_job(agent_id)["state"]}


broker = Broker(CONFIG)
batch_manager = BatchManager(broker, LeaseRequest)
mcp = FastMCP("Codex local AI agents")
mcp.settings.streamable_http_path = "/"


@mcp.tool()
async def gpu_status():
    status = await broker.status()
    name = next(iter(broker.config.get("codex_model_names", [])), None)
    model = next((item for item in status["models"] if item["name"] == name), None)
    limits = broker.backend_limits(model) if model else {}
    active = list(broker.inference_reservations.values())
    reserved_tokens = sum(int(item["reserved_tokens"]) for item in active)
    active_sequences = sum(int(item["choices"]) for item in active)
    kv_capacity = None
    kv_error = None
    if model and model.get("ready_replicas"):
        try:
            kv_capacity = await broker.kv_capacity_tokens(model)
        except HTTPException as exc:
            kv_error = exc.detail
    jobs = [dict(row) for row in broker.db.execute("SELECT id,owner,workspace,lease_id,state,progress,error,context_length,task_summary,created,updated FROM agent_jobs WHERE state NOT IN ('completed','succeeded','failed','cancelled','preempted','interrupted') ORDER BY created").fetchall()]
    capacity = status["capacity_estimate"][0] if status["capacity_estimate"] else None
    model_state = status.get("model_state", "unknown")
    if model and model.get("ready_replicas") and model_state == "idle":
        model_state = "ready"
    if model and not model.get("replicas") and model_state == "ready":
        model_state = "idle"
    agents = []
    for row in jobs:
        requests = [item for item in active if item["lease_id"] == row["lease_id"]]
        agents.append({"id": row["id"], "owner": row["owner"], "task": row["task_summary"], "status": broker.public_job_state(row["state"]), "context_tokens": row["context_length"], "updated": row["updated"], "workspace": row["workspace"], "progress": row["progress"], "error": row["error"], "cpu_state": "running" if row["state"] in ("starting", "running", "acquiring") else row["state"], "active_inference_requests": len(requests), "inference_state": "active" if requests else "idle"})
    return {
        "service_state": "draining" if broker.draining else model_state,
        "mcp": {"available": True, "path": "/mcp"},
        "docker": {"available": status["runtime"]["available"], "runtime": status["runtime"]["distribution"]},
        "model": {"name": name, "state": model_state, "loaded": bool(model and model.get("ready_replicas")), "image": broker.container_spec(name)["image"] if name else None, "max_context_tokens": limits.get("max-model-len"), "active_sequences": active_sequences, "max_sequences": limits.get("max-num-seqs"), "kv_capacity_tokens": kv_capacity, "kv_reserved_tokens": reserved_tokens, "kv_available_tokens": max(0, kv_capacity - reserved_tokens) if kv_capacity is not None else None, "kv_measurement": "measured" if kv_capacity is not None else "unknown", "kv_error": kv_error},
        "gpu": {"name": capacity.get("gpu") if capacity else None, "total_bytes": capacity.get("total_bytes") if capacity else None, "used_bytes": capacity.get("driver_used_bytes") if capacity else None, "external_bytes": capacity.get("external_process_bytes") if capacity else None, "external_reserve_bytes": capacity.get("external_reserve_bytes") if capacity else None, "target_llm_bytes": capacity.get("target_llm_budget_bytes") if capacity else None, "available_llm_bytes": capacity.get("available_llm_budget_bytes") if capacity else None, "measurement": capacity.get("measurement", "unknown") if capacity else "unknown", "sample_age_seconds": capacity.get("sample_age_seconds") if capacity else None, "external_observation_available": status["windows_gpu"].get("available", False), "revision": status["budget_revision"]},
        "agents": agents,
        "queue": [{"id": row["id"], "owner": row["owner"], "task": row["task_summary"], "status": broker.public_job_state(row["state"]), "context_tokens": row["context_length"], "created": row["created"]} for row in jobs if row["state"] in ("queued", "acquiring")],
        "inference": {"active_requests": len(active), "active_sequences": active_sequences, "reserved_tokens": reserved_tokens},
        "external_pressure": status["external_growth_pressure"]["active"],
        "budget_revision": status["budget_revision"],
        "profile_change_required": status["requires_profile_change"],
        "runtime_gpu_observation_error": status.get("runtime_gpu_observation_error"),
    }


@mcp.tool()
async def gpu_agent_start(task: str, workspace: str, context_length: int | None = None, files: list[str] | None = None, test_command: str | None = None, idempotency_key: str | None = None, reasoning_level: str | None = None):
    """Start a local coding agent in an isolated Git worktree. Set reasoning_level to off, low, medium, or xhigh; default is xhigh. high remains an alias for xhigh. Prefix task with [ask] for a read-only answer. Use idempotency_key for safe retries; poll or cancel by job_id."""
    if not task.strip() or len(task) > 30000:
        raise HTTPException(422, "task must contain 1 to 30000 characters")
    return await broker.start_agent("Codex", task, workspace, context_length, files, test_command, idempotency_key, reasoning_level)


@mcp.tool()
async def gpu_agent_poll(agent_id: str, wait_seconds: int = 20):
    """Poll a coding job for progress, diff summary, tests, and its current state."""
    return await broker.poll_agent(agent_id, wait_seconds)


@mcp.tool()
async def gpu_agent_cancel(agent_id: str):
    """Cancel the local Qwen coding agent and release its broker lease."""
    return await broker.cancel_agent(agent_id)


@mcp.tool()
async def gpu_job_get(job_id: str, wait_seconds: int = 0):
    """Get a job's state, progress, worktree diff summary, test result, and errors."""
    return await broker.poll_agent(job_id, wait_seconds)


@mcp.tool()
async def gpu_wait_events(after_cursor: int = 0, wait_seconds: int = 20):
    """Wait for job and service events after a cursor; returns the next cursor."""
    return await broker.wait_events(after_cursor, wait_seconds)


@mcp.tool()
async def gpu_model_unload(force: bool = False, wait_seconds: int = 300):
    """Gracefully stop new admissions, finish active agent work, and unload Qwen. Set force only to cancel broker-owned jobs."""
    if wait_seconds < 0 or wait_seconds > 3600:
        raise HTTPException(422, "wait_seconds must be between 0 and 3600")
    return await broker.unload_model(force, wait_seconds)


@mcp.tool()
async def gpu_batch_submit(items: list[dict], idempotency_key: str | None = None):
    """Submit up to 32 independent inference items without coding worktrees. Each item may set max_tokens up to 32768 and reasoning_level to off, low, medium, or xhigh. The default output budget is 16384 tokens, with xhigh reasoning by default."""
    return await batch_manager.submit("Codex", items, idempotency_key)


@mcp.tool()
async def gpu_batch_get(batch_id: str):
    """Get partial results and individual errors for an inference batch."""
    return batch_manager.snapshot(batch_id)


@mcp.tool()
async def gpu_batch_retry_failed(batch_id: str, item_ids: list[str] | None = None):
    """Retry failed or interrupted batch items. Items truncated by their output budget are retried with twice the token budget, up to 32768."""
    return await batch_manager.retry_failed(batch_id, item_ids)


@asynccontextmanager
async def lifespan(app):
    await broker.start()
    await batch_manager.recover()
    monitor_task = asyncio.create_task(broker.monitor_external())
    try:
        async with mcp.session_manager.run():
            yield
    finally:
        monitor_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await monitor_task
        await broker.close()


app = FastAPI(title="Codex Local AI Agents", version="0.2.0", lifespan=lifespan)
app.mount("/mcp", mcp.streamable_http_app())


@app.middleware("http")
async def loopback_only(request: Request, call_next):
    if request.client and request.client.host not in ("127.0.0.1", "::1", "localhost"):
        return JSONResponse({"detail": "Loopback access only"}, status_code=403)
    return await call_next(request)


@app.get("/healthz")
async def health():
    return {"status": "ok", "pid": os.getpid()}


@app.get("/v1/status")
async def status():
    return await broker.status()


@app.post("/v1/leases")
async def acquire(spec: LeaseRequest):
    try:
        return await broker.acquire(spec)
    except (DockerRuntimeError, OSError, TimeoutError) as exc:
        broker.model_state = "idle"
        raise HTTPException(503, {"code": "runtime_unavailable", "reason": "docker_control_unavailable", "requested": {"operation": "agent_start"}, "available": {"runtime": "temporarily_unavailable"}, "limiting_factor": "wsl_docker_control", "retryable": True, "alternatives": [{"action": "retry_after_runtime_recovers"}]}) from exc


@app.post("/v1/leases/{lease_id}/renew")
async def renew(lease_id: str, spec: Renewal):
    return await broker.renew(lease_id, spec.ttl_seconds)


@app.delete("/v1/leases/{lease_id}")
async def release(lease_id: str):
    return await broker.release(lease_id)


@app.get("/v1/leases/{lease_id}/events")
async def lease_events(lease_id: str, after: int = 0):
    broker.lease(lease_id)
    return {"events": broker.events_since(lease_id, after)}


@app.post("/v1/leases/{lease_id}/chat/completions")
async def chat_completions(lease_id: str, body: dict, request: Request):
    return await broker.proxy(lease_id, body, request)


@app.post("/v1/batches")
async def batch_submit(body: dict):
    items = body.get("items")
    if not isinstance(items, list):
        raise HTTPException(422, {"code": "invalid_batch_items"})
    return await batch_manager.submit("Dashboard", items, body.get("idempotency_key"))


@app.get("/v1/batches/{batch_id}")
async def batch_get(batch_id: str):
    return batch_manager.snapshot(batch_id)


@app.post("/v1/batches/{batch_id}/retry")
async def batch_retry(batch_id: str, body: dict):
    item_ids = body.get("item_ids")
    if item_ids is not None and (not isinstance(item_ids, list) or any(not isinstance(item_id, str) for item_id in item_ids)):
        raise HTTPException(422, {"code": "invalid_batch_item_ids"})
    return await batch_manager.retry_failed(batch_id, item_ids)


@app.get("/v1/events")
async def events(after_cursor: int = 0, wait_seconds: int = 20):
    return await broker.wait_events(after_cursor, wait_seconds)


@app.get("/v1/agent-jobs/{job_id}")
async def agent_job(job_id: str, wait_seconds: int = 0):
    return await broker.poll_agent(job_id, wait_seconds)


@app.post("/v1/model/unload")
async def model_unload(body: dict):
    wait_seconds = int(body.get("wait_seconds", 300))
    if wait_seconds < 0 or wait_seconds > 3600:
        raise HTTPException(422, "wait_seconds must be between 0 and 3600")
    return await broker.unload_model(bool(body.get("force", False)), wait_seconds)


@app.api_route("/v1/leases/{lease_id}/comfyui/{path:path}", methods=["GET", "POST", "DELETE"], include_in_schema=False)
async def unsupported_video_proxy(lease_id: str, path: str):
    raise HTTPException(501, {"code": "unsupported_operation", "operation": "video", "message": "This broker serves coding agents only."})


@app.api_route("/v1/video-queue/{owner}", methods=["GET", "POST", "DELETE"], include_in_schema=False)
async def unsupported_video_queue(owner: str):
    raise HTTPException(501, {"code": "unsupported_operation", "operation": "video_queue", "message": "This broker serves coding agents only."})


@app.get("/v1/inference-calls")
async def recent_inference_calls(limit: int = 50):
    if limit < 1 or limit > 200:
        raise HTTPException(422, "limit must be between 1 and 200")
    rows = broker.db.execute("SELECT * FROM inference_calls ORDER BY started DESC LIMIT ?", (limit,)).fetchall()
    return {"calls": [dict(row) for row in rows]}


@app.get("/v1/agents/{lease_id}/models")
async def agent_models(lease_id: str):
    row = broker.lease(lease_id)
    if row["kind"] != "agent" or row["state"] != "active":
        raise HTTPException(409, "GPU lease is not active for inference")
    return {"object": "list", "data": [{"id": row["model_name"], "object": "model", "created": 0, "owned_by": "local-broker"}]}


@app.post("/v1/agents/{lease_id}/responses")
async def agent_responses(lease_id: str, body: dict, request: Request):
    return await broker.proxy(lease_id, body, request, "responses")


if __name__ == "__main__":
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = (wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR)
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    singleton_mutex = kernel32.CreateMutexW(None, False, "Local\\CodexLocalGpuBroker")
    if not singleton_mutex:
        raise ctypes.WinError(ctypes.get_last_error())
    if ctypes.get_last_error() == 183:
        raise SystemExit("GPU broker is already running")
    uvicorn.run(app, host=CONFIG.get("host", "127.0.0.1"), port=int(CONFIG.get("port", 8096)), access_log=False, log_level="error", timeout_graceful_shutdown=30)
