param([switch]$Once)

$ErrorActionPreference = 'Stop'
$distribution = 'Ubuntu-24.04'
$brokerStatusUrl = 'http://127.0.0.1:8096/v1/status'
$logPath = 'C:\AI\gpu-broker\logs\wsl-runtime-watchdog.log'
$badWslCount = 0
$badDockerCount = 0
$badBrokerCount = 0
$lastState = ''

function Write-WatchdogLog([string]$message) {
    $parent = Split-Path -Parent $logPath
    if (-not (Test-Path -LiteralPath $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    if ((Test-Path -LiteralPath $logPath) -and (Get-Item -LiteralPath $logPath).Length -gt 4194304) {
        Move-Item -LiteralPath $logPath -Destination ($logPath + '.previous') -Force
    }
    Add-Content -LiteralPath $logPath -Encoding UTF8 -Value ("{0:o} {1}" -f [DateTime]::UtcNow, $message)
}

function Invoke-WslCommand([string[]]$command, [int]$timeoutMilliseconds = 12000) {
    $allArguments = @('-d', $distribution, '--exec') + $command
    $quotedArguments = foreach ($argument in $allArguments) {
        if ($argument -match '\s') { '"' + $argument.Replace('"', '\"') + '"' } else { $argument }
    }
    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = Join-Path $env:WINDIR 'System32\wsl.exe'
    $startInfo.Arguments = [string]::Join(' ', $quotedArguments)
    $startInfo.UseShellExecute = $false
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.CreateNoWindow = $true
    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    if (-not $process.Start()) { return [pscustomobject]@{ ExitCode = -998; Output = 'process start failed' } }
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    if (-not $process.WaitForExit($timeoutMilliseconds)) {
        try { $process.Kill() } catch {}
        try { $process.WaitForExit() } catch {}
        return [pscustomobject]@{ ExitCode = -999; Output = 'command timeout' }
    }
    $output = $stdoutTask.Result + $stderrTask.Result
    return [pscustomobject]@{ ExitCode = $process.ExitCode; Output = $output.Trim() }
}

function Get-GpuUsedMiB {
    $nvidiaSmi = Join-Path $env:WINDIR 'System32\nvidia-smi.exe'
    if (-not (Test-Path -LiteralPath $nvidiaSmi)) { return $null }
    try {
        $values = @(& $nvidiaSmi '--query-gpu=memory.used' '--format=csv,noheader,nounits' 2>$null | ForEach-Object { $_.Trim() } | Where-Object { $_ -match '^\d+$' } | ForEach-Object { [int]$_ })
        if ($LASTEXITCODE -ne 0 -or $values.Count -eq 0) { return $null }
        return [int](($values | Measure-Object -Sum).Sum)
    } catch {
        return $null
    }
}

function Get-BrokerSafetyState {
    try {
        $status = Invoke-RestMethod -Uri $brokerStatusUrl -TimeoutSec 8
        $external = $status.windows_gpu
        $inference = $status.inference
        $safe = @($status.leases).Count -eq 0 -and @($status.queue).Count -eq 0 -and @($status.agents).Count -eq 0 -and [int]$inference.active_requests -eq 0 -and [int]$inference.active_sequences -eq 0 -and [int]$inference.reserved_tokens -eq 0 -and @($external.active_engines).Count -eq 0 -and [int64]$external.external_memory_bytes -lt 1073741824 -and -not [bool]$status.external_growth_pressure.active
        return [pscustomobject]@{ Available = $true; Safe = $safe; Status = $status }
    } catch {
        $usedMiB = Get-GpuUsedMiB
        return [pscustomobject]@{ Available = $usedMiB -ne $null; Safe = $usedMiB -ne $null -and $usedMiB -le 1024; Status = $null; FallbackGpuUsedMiB = $usedMiB }
    }
}

function Test-DockerApi {
    $probe = Invoke-WslCommand @('/usr/bin/docker', 'info', '--format', '{{.ServerVersion}}')
    return $probe.ExitCode -eq 0 -and $probe.Output.Length -gt 0
}

function Invoke-Recovery {
    $safety = Get-BrokerSafetyState
    if (-not $safety.Available -or -not $safety.Safe) {
        if ($script:badWslCount -ge 1 -or $script:badDockerCount -ge 3 -or $script:badBrokerCount -ge 3) {
            Write-WatchdogLog 'recovery deferred: broker safety gate is unavailable or workload is active'
        }
        return
    }
    if ($script:badWslCount -ge 1) {
        Write-WatchdogLog 'WSL exec probe failed; restarting only Ubuntu-24.04 after idle safety gate passed'
        $terminated = Start-Process -FilePath (Join-Path $env:WINDIR 'System32\wsl.exe') -ArgumentList @('--terminate', $distribution) -Wait -PassThru -WindowStyle Hidden
        if ($terminated.ExitCode -ne 0) {
            Write-WatchdogLog ("wsl terminate failed with exit {0}" -f $terminated.ExitCode)
            return
        }
        for ($attempt = 0; $attempt -lt 12; $attempt++) {
            Start-Sleep -Seconds 2
            $probe = Invoke-WslCommand @('/bin/true')
            if ($probe.ExitCode -eq 0) {
                $script:badWslCount = 0
                $script:badDockerCount = 0
                Write-WatchdogLog 'WSL recovered; checking Docker API'
                break
            }
        }
        if ($script:badWslCount -ge 1) {
            Write-WatchdogLog 'WSL did not recover after distro restart; will retry after next probe interval'
            return
        }
    }
    if ($script:badDockerCount -ge 3) {
        Write-WatchdogLog 'Docker API failed repeatedly; asking the in-distro watchdog to recover it'
        $restart = Invoke-WslCommand @('/usr/bin/systemctl', 'restart', 'docker.service') 45000
        if ($restart.ExitCode -eq 0) {
            for ($attempt = 0; $attempt -lt 12; $attempt++) {
                Start-Sleep -Seconds 2
                if (Test-DockerApi) {
                    $script:badDockerCount = 0
                    Write-WatchdogLog 'Docker API recovered after docker.service restart'
                    break
                }
            }
        } else {
            Write-WatchdogLog ("docker.service restart failed: {0}" -f $restart.Output)
        }
    }
    $model = @($safety.Status.models | Where-Object { $_.id -eq 'qwen3.8-27b' } | Select-Object -First 1)[0]
    $containerState = if ($model -and $model.container) { [string]$model.container.State.Status } else { '' }
    if ([string]$safety.Status.model_state -eq 'loading' -and $model -and [int]$model.replicas -eq 0 -and $containerState -in @('exited', 'dead', 'created')) {
        $script:badBrokerCount++
    } else {
        $script:badBrokerCount = 0
    }
    if ($script:badBrokerCount -ge 3) {
        Write-WatchdogLog 'broker remained loading with no running container and no active workload; restarting GPUStack Broker task'
        try {
            Stop-ScheduledTask -TaskName 'GPUStack Broker'
            Start-Sleep -Seconds 2
            Start-ScheduledTask -TaskName 'GPUStack Broker'
            $script:badBrokerCount = 0
            Start-Sleep -Seconds 5
            $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8096/healthz' -TimeoutSec 8
            Write-WatchdogLog ("GPUStack Broker restarted; health PID {0}" -f $health.pid)
        } catch {
            Write-WatchdogLog ("GPUStack Broker task restart failed: {0}" -f $_.Exception.Message)
        }
    }
}

do {
    $wslProbe = Invoke-WslCommand @('/bin/true')
    if ($wslProbe.ExitCode -eq 0) {
        $badWslCount = 0
        $dockerOk = Test-DockerApi
        if ($dockerOk) {
            $badDockerCount = 0
            if ($lastState -ne 'healthy') { Write-WatchdogLog 'WSL exec and Docker API healthy' }
            $lastState = 'healthy'
            Invoke-Recovery
        } else {
            $badDockerCount++
            if ($lastState -ne 'docker-failed') { Write-WatchdogLog 'WSL exec works but Docker API probe failed' }
            $lastState = 'docker-failed'
            Invoke-Recovery
        }
    } else {
        $badWslCount++
        if ($lastState -ne 'wsl-failed') { Write-WatchdogLog ("WSL exec probe failed: {0}" -f $wslProbe.Output) }
        $lastState = 'wsl-failed'
        Invoke-Recovery
    }
    if ($Once) { break }
    Start-Sleep -Seconds 30
} while ($true)
