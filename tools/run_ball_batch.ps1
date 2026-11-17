# Sequential batch runner for tools/detect_ball_tracks.py.
# Usage: powershell -File tools/run_ball_batch.ps1 intel:NPU "C:\...\a.mp4" "C:\...\b.mp4"
param(
    [Parameter(Mandatory = $true)][string]$Device,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Videos
)

$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo ".venv\Scripts\python.exe"

foreach ($video in $Videos) {
    if (-not $video) { continue }
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($video)
    $outDir = Join-Path $repo "output_videos\$stem\ball"
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    $log = Join-Path $outDir "detect_$($Device -replace ':', '_').log"
    Add-Content -Path $log -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] start $video on $Device"
    & $python (Join-Path $repo "tools\detect_ball_tracks.py") --video $video --device $Device *>> $log
    Add-Content -Path $log -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] finished $video (exit $LASTEXITCODE)"
}
