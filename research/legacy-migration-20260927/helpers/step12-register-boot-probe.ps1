$ErrorActionPreference = 'Stop'
$script = 'C:\AI\tuning\ninfer-migration-20260927\step12-boot-probe.ps1'
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$script`""
$trigger = New-ScheduledTaskTrigger -AtStartup
$principal = New-ScheduledTaskPrincipal -UserId 'aleksei' -LogonType S4U -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 25) -StartWhenAvailable
Register-ScheduledTask -TaskName 'NInfer One-Time Boot Probe' -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
Get-ScheduledTask -TaskName 'NInfer One-Time Boot Probe' | Select-Object TaskName,State,Triggers,Actions
