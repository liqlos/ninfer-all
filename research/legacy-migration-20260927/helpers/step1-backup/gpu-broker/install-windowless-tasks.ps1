$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonw = Join-Path $root '.venv\Scripts\pythonw.exe'
$app = Join-Path $root 'app.py'
$keepalive = Join-Path $root 'wsl-keepalive.vbs'
if (-not (Test-Path -LiteralPath $pythonw) -or -not (Test-Path -LiteralPath $keepalive)) { throw 'Required launcher is missing' }

$status = $null
try { $status = Invoke-RestMethod 'http://127.0.0.1:8096/v1/status' -TimeoutSec 5 } catch {}
if ($status -and ($status.leases.Count -gt 0 -or $status.agents.Count -gt 0 -or $status.inference.active_requests -gt 0)) {
    throw 'Broker has active work; task restart refused'
}
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType S4U -RunLevel Highest
$bootTrigger = New-ScheduledTaskTrigger -AtStartup
$brokerAction = New-ScheduledTaskAction -Execute $pythonw -Argument "`"$app`"" -WorkingDirectory $root
Set-ScheduledTask -TaskName 'GPUStack Broker' -Action $brokerAction -Principal $principal -Trigger $bootTrigger | Out-Null
$keepaliveAction = New-ScheduledTaskAction -Execute "$env:WINDIR\System32\wscript.exe" -Argument "//B //NoLogo `"$keepalive`"" -WorkingDirectory $root
Set-ScheduledTask -TaskName 'GPUStack WSL Keepalive' -Action $keepaliveAction -Principal $principal -Trigger $bootTrigger | Out-Null

Stop-ScheduledTask -TaskName 'GPUStack WSL Keepalive' -ErrorAction SilentlyContinue
Start-ScheduledTask -TaskName 'GPUStack WSL Keepalive'
Stop-ScheduledTask -TaskName 'GPUStack Broker' -ErrorAction SilentlyContinue
$deadline = (Get-Date).AddSeconds(30)
while ((Get-ScheduledTask -TaskName 'GPUStack Broker').State -ne 'Ready' -or
       (Get-NetTCPConnection -LocalPort 8096 -State Listen -ErrorAction SilentlyContinue)) {
    if ((Get-Date) -ge $deadline) { throw 'Broker did not stop before restart' }
    Start-Sleep -Milliseconds 250
}
Start-ScheduledTask -TaskName 'GPUStack Broker'
$ready = $false
for ($i = 0; $i -lt 40; $i++) {
    try {
        Invoke-RestMethod 'http://127.0.0.1:8096/healthz' -TimeoutSec 2 | Out-Null
        $ready = $true
        break
    } catch {
        Start-Sleep -Seconds 1
    }
}
if (-not $ready) { throw 'Broker did not recover after task restart' }
