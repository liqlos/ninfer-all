import asyncio
import hashlib
import json
import time
import uuid

from fastapi import HTTPException
from reasoning import normalize_reasoning_level


MAX_BATCH_OUTPUT_TOKENS = 32768


class BatchManager:
    def __init__(self, broker, lease_request_type):
        self.broker = broker
        self.lease_request_type = lease_request_type
        self.tasks = {}
        self.lock = asyncio.Lock()
        self.broker.db.executescript("""
            CREATE TABLE IF NOT EXISTS inference_batches (
                id TEXT PRIMARY KEY,
                idempotency_key TEXT UNIQUE,
                request_hash TEXT NOT NULL,
                owner TEXT NOT NULL,
                state TEXT NOT NULL,
                created REAL NOT NULL,
                updated REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS inference_batch_items (
                batch_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                prompt TEXT NOT NULL,
                max_tokens INTEGER NOT NULL,
                reasoning_level TEXT NOT NULL DEFAULT 'medium',
                state TEXT NOT NULL,
                result TEXT,
                error TEXT,
                attempts INTEGER NOT NULL DEFAULT 0,
                updated REAL NOT NULL,
                PRIMARY KEY (batch_id, item_id)
            );
        """)
        columns = {row["name"] for row in self.broker.db.execute("PRAGMA table_info(inference_batch_items)")}
        if "reasoning_level" not in columns:
            self.broker.db.execute("ALTER TABLE inference_batch_items ADD COLUMN reasoning_level TEXT NOT NULL DEFAULT 'medium'")

    async def recover(self):
        now = time.time()
        self.broker.db.execute("UPDATE inference_batch_items SET state='interrupted',error=?,updated=? WHERE state IN ('running','queued') AND batch_id IN (SELECT id FROM inference_batches WHERE state='running')", (json.dumps({"code": "broker_restarted"}), now))
        self.broker.db.execute("UPDATE inference_batches SET state='interrupted',updated=? WHERE state='running'", (now,))

    def snapshot(self, batch_id):
        batch = self.broker.db.execute("SELECT * FROM inference_batches WHERE id=?", (batch_id,)).fetchone()
        if batch is None:
            raise HTTPException(404, {"code": "batch_not_found", "batch_id": batch_id})
        rows = self.broker.db.execute("SELECT item_id,max_tokens,state,result,error,attempts,updated,reasoning_level FROM inference_batch_items WHERE batch_id=? ORDER BY rowid", (batch_id,)).fetchall()
        items = [{"item_id": row["item_id"], "max_tokens": row["max_tokens"], "state": row["state"], "result": json.loads(row["result"]) if row["result"] else None, "error": json.loads(row["error"]) if row["error"] else None, "attempts": row["attempts"], "updated": row["updated"], "reasoning_level": row["reasoning_level"]} for row in rows]
        return {"batch_id": batch_id, "owner": batch["owner"], "state": batch["state"], "created": batch["created"], "updated": batch["updated"], "items": items}

    async def submit(self, owner, items, idempotency_key=None):
        if not 1 <= len(items) <= 32:
            raise HTTPException(422, {"code": "invalid_batch_size", "requested": len(items), "available": {"min": 1, "max": 32}})
        normalized = []
        ids = set()
        for item in items:
            item_id = str(item.get("item_id", ""))
            prompt = item.get("prompt")
            max_tokens = item.get("max_tokens", getattr(self.broker, "config", {}).get("default_inference_output_tokens", 16384))
            if not item_id or len(item_id) > 100 or item_id in ids or not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 30000 or not isinstance(max_tokens, int) or not 1 <= max_tokens <= MAX_BATCH_OUTPUT_TOKENS:
                raise HTTPException(422, {"code": "invalid_batch_item", "item_id": item_id})
            ids.add(item_id)
            reasoning_level = normalize_reasoning_level(item.get("reasoning_level"), getattr(self.broker, "config", {}).get("default_reasoning_level", "xhigh"))
            normalized.append({"item_id": item_id, "prompt": prompt, "max_tokens": max_tokens, "reasoning_level": reasoning_level})
        digest = hashlib.sha256(json.dumps({"owner": owner, "items": normalized}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        async with self.lock:
            if idempotency_key:
                existing = self.broker.db.execute("SELECT id,request_hash FROM inference_batches WHERE idempotency_key=?", (idempotency_key,)).fetchone()
                if existing:
                    if existing["request_hash"] != digest:
                        raise HTTPException(409, {"code": "idempotency_conflict", "batch_id": existing["id"]})
                    return self.snapshot(existing["id"])
            batch_id = uuid.uuid4().hex
            now = time.time()
            self.broker.db.execute("INSERT INTO inference_batches VALUES(?,?,?,?,?,?,?)", (batch_id, idempotency_key, digest, owner, "running", now, now))
            self.broker.db.executemany("INSERT INTO inference_batch_items(batch_id,item_id,prompt,max_tokens,reasoning_level,state,updated) VALUES(?,?,?,?,?,?,?)", [(batch_id, item["item_id"], item["prompt"], item["max_tokens"], item["reasoning_level"], "queued", now) for item in normalized])
            task = asyncio.create_task(self.run(batch_id))
            self.tasks[batch_id] = task
            task.add_done_callback(lambda finished: self.tasks.pop(batch_id, None) if self.tasks.get(batch_id) is finished else None)
            return self.snapshot(batch_id)

    async def run(self, batch_id):
        lease_id = None
        heartbeat_task = None
        lane_group = None
        try:
            rows = self.broker.db.execute("SELECT item_id FROM inference_batch_items WHERE batch_id=? AND state='queued' ORDER BY rowid", (batch_id,)).fetchall()
            model_name = self.broker.config["codex_model_names"][0]
            model = await self.broker.model(model_name)
            max_sequences = int(self.broker.backend_limits(model).get("max-num-seqs", 2))
            lane_count = max(1, min(len(rows), max_sequences))
            spec = self.lease_request_type(owner=f"batch:{batch_id}", kind="agent", model_name=model_name, requested_agents=lane_count, context_length=int(self.broker.container_spec(model_name)["environment"]["MAX_LEN"]), ttl_seconds=900)
            deadline = time.monotonic() + 300
            while True:
                try:
                    lease = await self.broker.acquire(spec)
                    lease_id = lease["lease_id"]
                    break
                except HTTPException as exc:
                    if exc.status_code != 409 or time.monotonic() >= deadline or not isinstance(exc.detail, dict) or not exc.detail.get("retryable"):
                        raise
                    await asyncio.sleep(1)
            heartbeat_task = asyncio.create_task(self.lease_heartbeat(lease_id))
            pending = asyncio.Queue()
            for row in rows:
                pending.put_nowait(row["item_id"])
            lane_tasks = [asyncio.create_task(self.run_lane(batch_id, pending, lease_id, model_name)) for _ in range(lane_count)]
            try:
                lane_group = asyncio.gather(*lane_tasks)
                done, _ = await asyncio.wait((lane_group, heartbeat_task), return_when=asyncio.FIRST_COMPLETED)
                if heartbeat_task in done:
                    heartbeat_task.result()
                    raise RuntimeError("batch lease heartbeat stopped unexpectedly")
                await lane_group
            except BaseException:
                for task in lane_tasks:
                    task.cancel()
                await asyncio.gather(*lane_tasks, return_exceptions=True)
                if lane_group is not None:
                    await asyncio.gather(lane_group, return_exceptions=True)
                raise
            states = [row["state"] for row in self.broker.db.execute("SELECT state FROM inference_batch_items WHERE batch_id=?", (batch_id,)).fetchall()]
            state = "succeeded" if all(value == "succeeded" for value in states) else "completed_with_errors"
            self.broker.db.execute("UPDATE inference_batches SET state=?,updated=? WHERE id=? AND state='running'", (state, time.time(), batch_id))
        except asyncio.CancelledError:
            now = time.time()
            self.broker.db.execute("UPDATE inference_batch_items SET state='cancelled',updated=? WHERE batch_id=? AND state='queued'", (now, batch_id))
            self.broker.db.execute("UPDATE inference_batches SET state='cancelled',updated=? WHERE id=?", (now, batch_id))
            raise
        except Exception as exc:
            now = time.time()
            detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
            self.broker.db.execute("UPDATE inference_batch_items SET state='failed',error=?,updated=? WHERE batch_id=? AND state IN ('queued','running')", (json.dumps(detail, ensure_ascii=False), now, batch_id))
            self.broker.db.execute("UPDATE inference_batches SET state='completed_with_errors',updated=? WHERE id=?", (now, batch_id))
        finally:
            if heartbeat_task:
                heartbeat_task.cancel()
                await asyncio.gather(heartbeat_task, return_exceptions=True)
            if lease_id:
                try:
                    await self.broker.release(lease_id)
                except HTTPException:
                    pass

    async def lease_heartbeat(self, lease_id):
        while True:
            await asyncio.sleep(300)
            await self.broker.renew(lease_id, 900)

    async def run_lane(self, batch_id, pending, lease_id, model_name):
        while True:
            try:
                item_id = pending.get_nowait()
            except asyncio.QueueEmpty:
                return
            await self.run_item(batch_id, item_id, lease_id, model_name)

    async def run_item(self, batch_id, item_id, lease_id, model_name):
        row = self.broker.db.execute("SELECT prompt,max_tokens,attempts,reasoning_level FROM inference_batch_items WHERE batch_id=? AND item_id=?", (batch_id, item_id)).fetchone()
        self.broker.db.execute("UPDATE inference_batch_items SET state='running',attempts=attempts+1,updated=? WHERE batch_id=? AND item_id=?", (time.time(), batch_id, item_id))
        try:
            payload = {"model": model_name, "messages": [{"role": "user", "content": row["prompt"]}], "max_tokens": row["max_tokens"], "reasoning_level": row["reasoning_level"], "stream": False}
            wait_deadline = time.monotonic() + float(self.broker.config.get("batch_max_capacity_wait_seconds", self.broker.config.get("agent_max_queue_wait_seconds", 21600)))
            while True:
                try:
                    started = time.monotonic()
                    response = await self.broker.proxy(lease_id, payload.copy(), None)
                    if hasattr(response, "body_iterator"):
                        content = bytearray()
                        async for chunk in response.body_iterator:
                            content.extend(chunk if isinstance(chunk, bytes) else chunk.encode("utf-8"))
                        response_body = bytes(content)
                    else:
                        response_body = response.body
                    elapsed = time.monotonic() - started
                    data = json.loads(response_body)
                    if response.status_code >= 400:
                        raise HTTPException(response.status_code, data)
                    break
                except HTTPException as exc:
                    detail = exc.detail if isinstance(exc.detail, dict) else {}
                    available = detail.get("available") or {}
                    requested = detail.get("requested") or {}
                    impossible = requested.get("total_tokens", 0) > available.get("kv_capacity_tokens", float("inf"))
                    if exc.status_code not in (409, 503) or not detail.get("retryable") or impossible or time.monotonic() >= wait_deadline:
                        raise
                    changed = getattr(self.broker, "capacity_changed", None)
                    if changed is None:
                        await asyncio.sleep(min(1, max(0, wait_deadline - time.monotonic())))
                    else:
                        changed.clear()
                        try:
                            await asyncio.wait_for(changed.wait(), timeout=min(1, max(0.001, wait_deadline - time.monotonic())))
                        except asyncio.TimeoutError:
                            pass
            choice = data["choices"][0]
            content = choice["message"].get("content")
            if not isinstance(content, str) or not content.strip():
                raise HTTPException(422, {"code": "empty_completion", "finish_reason": choice.get("finish_reason"), "usage": data.get("usage")})
            usage = data.get("usage")
            completion_tokens = usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0
            throughput = completion_tokens / elapsed if isinstance(completion_tokens, (int, float)) and completion_tokens > 0 and elapsed > 0 else 0
            finish_reason = choice.get("finish_reason")
            result = {"content": content, "finish_reason": finish_reason, "usage": data.get("usage"), "elapsed_seconds": elapsed, "output_tokens_per_second": throughput}
            if finish_reason == "length":
                error = {"code": "output_truncated", "message": "The output token budget was exhausted. Retry raises the budget up to the broker limit."}
                self.broker.db.execute("UPDATE inference_batch_items SET state='failed',result=?,error=?,updated=? WHERE batch_id=? AND item_id=?", (json.dumps(result, ensure_ascii=False), json.dumps(error), time.time(), batch_id, item_id))
            else:
                self.broker.db.execute("UPDATE inference_batch_items SET state='succeeded',result=?,error=NULL,updated=? WHERE batch_id=? AND item_id=?", (json.dumps(result, ensure_ascii=False), time.time(), batch_id, item_id))
        except asyncio.CancelledError:
            self.broker.db.execute("UPDATE inference_batch_items SET state='cancelled',updated=? WHERE batch_id=? AND item_id=?", (time.time(), batch_id, item_id))
            raise
        except Exception as exc:
            detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
            self.broker.db.execute("UPDATE inference_batch_items SET state='failed',error=?,updated=? WHERE batch_id=? AND item_id=?", (json.dumps(detail, ensure_ascii=False), time.time(), batch_id, item_id))

    async def retry_failed(self, batch_id, item_ids=None):
        async with self.lock:
            batch = self.snapshot(batch_id)
            if batch_id in self.tasks and not self.tasks[batch_id].done():
                raise HTTPException(409, {"code": "batch_running"})
            if batch["state"] == "running":
                raise HTTPException(409, {"code": "batch_running"})
            failed = [item for item in batch["items"] if item["state"] in ("failed", "interrupted") and (item_ids is None or item["item_id"] in item_ids)]
            retryable = []
            for item in failed:
                row = self.broker.db.execute("SELECT max_tokens FROM inference_batch_items WHERE batch_id=? AND item_id=?", (batch_id, item["item_id"])).fetchone()
                max_tokens = int(row["max_tokens"])
                error = item["error"] if isinstance(item["error"], dict) else {}
                if error.get("code") == "output_truncated":
                    next_max_tokens = min(MAX_BATCH_OUTPUT_TOKENS, max_tokens * 2)
                    if next_max_tokens == max_tokens:
                        continue
                    self.broker.db.execute("UPDATE inference_batch_items SET max_tokens=? WHERE batch_id=? AND item_id=?", (next_max_tokens, batch_id, item["item_id"]))
                retryable.append(item["item_id"])
            if not retryable:
                return batch
            now = time.time()
            for item_id in retryable:
                self.broker.db.execute("UPDATE inference_batch_items SET state='queued',result=NULL,error=NULL,updated=? WHERE batch_id=? AND item_id=?", (now, batch_id, item_id))
            self.broker.db.execute("UPDATE inference_batches SET state='running',updated=? WHERE id=?", (now, batch_id))
            task = asyncio.create_task(self.run(batch_id))
            self.tasks[batch_id] = task
            task.add_done_callback(lambda finished: self.tasks.pop(batch_id, None) if self.tasks.get(batch_id) is finished else None)
            return self.snapshot(batch_id)
