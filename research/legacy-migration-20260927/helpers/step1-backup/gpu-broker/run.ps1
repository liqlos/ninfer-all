$ErrorActionPreference = 'Stop'
Set-Location 'C:\AI\gpu-broker'
& 'C:\AI\gpu-broker\.venv\Scripts\python.exe' 'C:\AI\gpu-broker\app.py'
exit $LASTEXITCODE
