import asyncio
import hashlib
import json
import subprocess


class DockerRuntimeError(RuntimeError):
    def __init__(self, args, returncode, output):
        self.args_list = tuple(args)
        self.returncode = returncode
        self.output = output[-4000:]
        super().__init__(f"WSL Docker command failed with exit {returncode}: {' '.join(args)}: {self.output}")


class WSLDocker:
    def __init__(self, distribution="Ubuntu-24.04", executable="/usr/bin/docker", timeout=30):
        self.distribution = distribution
        self.executable = executable
        self.timeout = float(timeout)

    def command(self, *args):
        return ["wsl.exe", "--distribution", self.distribution, "--exec", self.executable, *map(str, args)]

    def host_command(self, executable, *args):
        return ["wsl.exe", "--distribution", self.distribution, "--exec", str(executable), *map(str, args)]

    async def run(self, *args, timeout=None, check=True):
        argv = self.command(*args)
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            output, _ = await asyncio.wait_for(process.communicate(), timeout or self.timeout)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise TimeoutError(f"WSL Docker command timed out: {' '.join(map(str, args))}")
        text = output.decode("utf-8", "replace").strip()
        if check and process.returncode:
            raise DockerRuntimeError(args, process.returncode, text)
        return text

    async def info(self):
        result = await self.run("info", "--format", "{{json .}}", timeout=15)
        return json.loads(result)

    async def host_run(self, executable, *args, timeout=None, check=True):
        argv = self.host_command(executable, *args)
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            output, _ = await asyncio.wait_for(process.communicate(), timeout or self.timeout)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise TimeoutError(f"WSL host command timed out: {executable} {' '.join(map(str, args))}")
        text = output.decode("utf-8", "replace").strip()
        if check and process.returncode:
            raise DockerRuntimeError((executable, *args), process.returncode, text)
        return text

    async def inspect(self, name):
        result = await self.run("inspect", name, check=False)
        if not result:
            return None
        if "no such object:" in result.lower() or "no such container:" in result.lower():
            return None
        try:
            value = json.loads(result)
        except json.JSONDecodeError as exc:
            raise DockerRuntimeError(("inspect", name), 1, result) from exc
        return value[0] if value else None

    async def containers_with_label(self, key, value):
        result = await self.run("ps", "-aq", "--filter", f"label={key}={value}")
        ids = result.splitlines()
        if not ids:
            return []
        inspected = await self.run("inspect", *ids)
        return json.loads(inspected)

    async def start(self, name, image, command, port, mounts=(), environment=None, labels=None, gpu=True, ipc_host=False):
        specification = {
            "image": image,
            "command": list(map(str, command)),
            "port": [int(port[0]), int(port[1])],
            "mounts": list(mounts),
            "environment": environment or {},
            "labels": labels or {},
            "gpu": bool(gpu),
            "ipc_host": bool(ipc_host),
        }
        fingerprint = hashlib.sha256(json.dumps(specification, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        fingerprint_label = "io.codex.gpu-broker.spec-sha256"
        current = await self.inspect(name)
        if current:
            state = (current.get("State") or {}).get("Status")
            if state == "running":
                actual_fingerprint = ((current.get("Config") or {}).get("Labels") or {}).get(fingerprint_label)
                if actual_fingerprint != fingerprint:
                    raise DockerRuntimeError(("start", name), 1, f"Running container {name} has a different or unverified image, environment, command, or runtime configuration. Stop it through the broker and recreate it before applying this profile; current container was left running.")
                image_id = (await self.run("image", "inspect", image, "--format", "{{.Id}}", check=False)).strip()
                if not image_id or current.get("Image") != image_id:
                    raise DockerRuntimeError(("start", name), 1, f"Running container {name} uses a different image ID than {image}. Stop it through the broker and recreate it before applying this profile; current container was left running.")
                return current
            await self.run("rm", "--force", name)
        args = ["run", "--detach", "--name", name, "--restart", "no", "--init"]
        if gpu:
            args.extend(("--gpus", "all"))
        if ipc_host:
            args.extend(("--ipc", "host"))
        args.extend(("--publish", f"127.0.0.1:{int(port[0])}:{int(port[1])}"))
        for item in mounts:
            mount_type = item.get("type", "bind")
            if mount_type not in ("bind", "volume"):
                raise ValueError(f"Unsupported Docker mount type: {mount_type}")
            source_key = "source" if mount_type == "volume" else "src"
            specification = f"type={mount_type},{source_key}={item['source']},dst={item['target']}"
            if item.get("read_only"):
                specification += ",readonly"
            args.extend(("--mount", specification))
        for key, value in (environment or {}).items():
            args.extend(("--env", f"{key}={value}"))
        for key, value in {**(labels or {}), fingerprint_label: fingerprint}.items():
            args.extend(("--label", f"{key}={value}"))
        args.extend((image, *map(str, command)))
        container_id = await self.run(*args, timeout=120)
        current = await self.inspect(container_id)
        if not current:
            raise DockerRuntimeError(tuple(args), 1, "Container was not inspectable after docker run")
        return current

    async def start_spec(self, model_name, specification, managed_label):
        labels = {managed_label: "true", f"{managed_label}.model": model_name}
        return await self.start(
            specification["container_name"],
            specification["image"],
            specification["command"],
            (specification["host_port"], specification["container_port"]),
            mounts=specification.get("mounts", ()),
            environment=specification.get("environment"),
            labels=labels,
            gpu=specification.get("gpu", True),
            ipc_host=specification.get("ipc_host", False),
        )

    async def stop_and_remove(self, name, timeout=30):
        current = await self.inspect(name)
        if not current:
            return {"container": name, "state": "absent"}
        state = (current.get("State") or {}).get("Status")
        if state == "running":
            await self.run("stop", "--time", int(timeout), name, timeout=timeout + 15)
        await self.run("rm", name)
        if await self.inspect(name):
            raise DockerRuntimeError(("rm", name), 1, "Container still exists after removal")
        return {"container": name, "state": "removed"}

    async def logs(self, name, tail=1000):
        return await self.run("logs", "--tail", int(tail), name, check=False)
