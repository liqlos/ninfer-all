$ErrorActionPreference = 'Stop'
$root = 'C:\AI\tuning\ninfer-migration-20260927'
$resultPath = Join-Path $root 'step12-boot-probe.json'
$python = 'C:\AI\gpu-broker\.venv\Scripts\python.exe'
$deadline = (Get-Date).AddMinutes(15)
$result = [ordered]@{ started_at = (Get-Date).ToString('o'); boot_time = (Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToString('o'); attempts = 0; passed = $false }
try {
    while ((Get-Date) -lt $deadline) {
        $result.attempts++
        try {
            $status = Invoke-RestMethod 'http://127.0.0.1:8096/v1/status' -TimeoutSec 10
            $ui = Invoke-WebRequest 'http://127.0.0.1:8081/' -UseBasicParsing -TimeoutSec 10
            if ($status.runtime.backend -eq 'ninfer' -and $ui.StatusCode -eq 200) { break }
        } catch {}
        Start-Sleep -Seconds 10
    }
    if (-not $status -or -not $ui -or $status.runtime.backend -ne 'ninfer' -or $ui.StatusCode -ne 200) { throw 'Broker or dashboard did not start within 15 minutes' }
    $mcpRaw = & $python (Join-Path $root 'step12-boot-mcp-check.py') 2>&1
    if ($LASTEXITCODE -ne 0) { throw "MCP probe failed: $mcpRaw" }
    $mcp = ($mcpRaw | Select-Object -Last 1 | ConvertFrom-Json)
    $lease = Invoke-RestMethod 'http://127.0.0.1:8096/v1/leases' -Method Post -ContentType 'application/json' -Body '{"owner":"NInfer boot probe","kind":"agent","model_name":"qwen3.8-27b","requested_agents":1,"context_mode":"fixed","context_length":8192,"ttl_seconds":300}' -TimeoutSec 600
    try {
        $body = @{ model = 'qwen3.8-27b'; messages = @(@{role = 'user'; content = 'Reply with exactly: BOOT_OK'}); max_tokens = 64; temperature = 0; stream = $false } | ConvertTo-Json -Depth 8 -Compress
        $completion = Invoke-RestMethod "http://127.0.0.1:8096/v1/leases/$($lease.lease_id)/chat/completions" -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 180
    } finally {
        Invoke-RestMethod "http://127.0.0.1:8096/v1/leases/$($lease.lease_id)" -Method Delete -TimeoutSec 30 | Out-Null
    }
    $final = Invoke-RestMethod 'http://127.0.0.1:8096/v1/status' -TimeoutSec 10
    $containers = @(wsl.exe -- docker ps --format '{{.Names}}' | Where-Object { $_ -in @('gpu-broker-ninfer', 'gpu-broker-qwen') })
    $result.mcp = $mcp
    $result.completion = $completion.choices[0].message.content
    $result.final_backend = $final.runtime.backend
    $result.final_ready = $final.models[0].ready_replicas
    $result.final_leases = @($final.leases).Count
    $result.containers = $containers
    $result.ui_status = $ui.StatusCode
    $result.gpu_telemetry = [ordered]@{ measured_at = (Get-Date).ToString('o'); gpu = $final.capacity_estimate[0].gpu; total_bytes = $final.capacity_estimate[0].total_bytes; driver_used_bytes = $final.capacity_estimate[0].driver_used_bytes; external_process_bytes = $final.capacity_estimate[0].external_process_bytes; external_reserve_bytes = $final.capacity_estimate[0].external_reserve_bytes; measurement = $final.capacity_estimate[0].measurement }
    $result.passed = $mcp.available -and $mcp.backend -eq 'ninfer' -and $mcp.ready -and $mcp.context -eq 122880 -and $result.completion.Trim() -eq 'BOOT_OK' -and $result.final_backend -eq 'ninfer' -and $result.final_ready -eq 1 -and $result.final_leases -eq 0 -and $containers.Count -eq 1 -and $containers[0] -eq 'gpu-broker-ninfer'
} catch {
    $result.error = $_.Exception.Message
} finally {
    $result.ended_at = (Get-Date).ToString('o')
    $result | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $resultPath -Encoding UTF8
    Unregister-ScheduledTask -TaskName 'NInfer One-Time Boot Probe' -Confirm:$false -ErrorAction SilentlyContinue
}
