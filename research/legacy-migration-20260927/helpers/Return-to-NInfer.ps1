$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$brokerRoot = 'C:\AI\gpu-broker'
$statusUri = 'http://127.0.0.1:8096/v1/status'
$healthUri = 'http://127.0.0.1:8096/healthz'
try { $status = Invoke-RestMethod $statusUri -TimeoutSec 5 } catch { $status = $null }
if ($status -and ($status.leases.Count -gt 0 -or $status.agents.Count -gt 0 -or $status.inference.active_requests -gt 0)) { throw 'NInfer return refused while broker work is active' }
if ($status) {
    Invoke-RestMethod 'http://127.0.0.1:8096/v1/model/unload' -Method Post -ContentType 'application/json' -Body '{"force":false,"wait_seconds":300}' -TimeoutSec 330 | Out-Null
}
Copy-Item -LiteralPath (Join-Path $root 'step12-ninfer-config-before-rollback.json') -Destination (Join-Path $brokerRoot 'config.json') -Force
foreach ($name in @('GPUStack WSL Keepalive','GPUStack Broker')) {
    $xmlPath = Join-Path $root "step12-active-tasks\$name.xml"
    Register-ScheduledTask -TaskName $name -Xml (Get-Content -LiteralPath $xmlPath -Raw) -Force | Out-Null
}
Stop-ScheduledTask -TaskName 'GPUStack Broker' -ErrorAction SilentlyContinue
$deadline = (Get-Date).AddSeconds(30)
while (Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort 8096 -State Listen -ErrorAction SilentlyContinue) {
    if ((Get-Date) -ge $deadline) { throw 'Broker listener did not stop during NInfer return' }
    Start-Sleep -Milliseconds 250
}
$deadline = (Get-Date).AddSeconds(30)
while (Get-CimInstance Win32_Process -Filter "Name = 'pythonw.exe'" | Where-Object { $_.CommandLine -like '*C:\AI\gpu-broker\app.py*' }) {
    if ((Get-Date) -ge $deadline) { throw 'Broker process did not stop during NInfer return' }
    Start-Sleep -Milliseconds 250
}
Start-ScheduledTask -TaskName 'GPUStack WSL Keepalive'
Start-ScheduledTask -TaskName 'GPUStack Broker'
$ready = $false
for ($i = 0; $i -lt 60; $i++) {
    try { Invoke-RestMethod $healthUri -TimeoutSec 2 | Out-Null; $ready = $true; break } catch { Start-Sleep -Seconds 1 }
}
if (-not $ready) { throw 'Broker did not recover after NInfer return' }
$status = Invoke-RestMethod $statusUri -TimeoutSec 10
if ($status.runtime.backend -ne 'ninfer') { throw "Expected NInfer backend, found $($status.runtime.backend)" }
$lease = $null
try {
    $lease = Invoke-RestMethod 'http://127.0.0.1:8096/v1/leases' -Method Post -ContentType 'application/json' -Body (@{owner='NInfer return verification';kind='agent';model_name='qwen3.8-27b';context_length=4096;ttl_seconds=900}|ConvertTo-Json) -TimeoutSec 660
    $reply = Invoke-RestMethod "http://127.0.0.1:8096/v1/leases/$($lease.lease_id)/chat/completions" -Method Post -ContentType 'application/json' -Body (@{model='qwen3.8-27b';messages=@(@{role='user';content='Reply with exactly: NINFER_RETURN_OK'});max_tokens=24;stream=$false}|ConvertTo-Json -Depth 6) -TimeoutSec 180
    if ($reply.choices[0].message.content.Trim() -ne 'NINFER_RETURN_OK') { throw "Unexpected NInfer response: $($reply.choices[0].message.content)" }
    $result = [pscustomobject]@{backend='ninfer';model=$reply.model;response=$reply.choices[0].message.content;lease_id=$lease.lease_id;status='completion_verified';config_hash=(Get-FileHash (Join-Path $brokerRoot 'config.json') -Algorithm SHA256).Hash}
    $result | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $root 'step12-return-ninfer.json') -Encoding UTF8
    $result | ConvertTo-Json -Depth 4
} finally {
    if ($lease -and $lease.lease_id) { Invoke-RestMethod "http://127.0.0.1:8096/v1/leases/$($lease.lease_id)" -Method Delete -TimeoutSec 30 | Out-Null }
}
