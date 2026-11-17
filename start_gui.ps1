#requires -Version 5.1

[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8765,
    [switch]$NoBrowser
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
$venvRoot = Join-Path $projectRoot ".venv"
$pythonPath = Join-Path $venvRoot "Scripts\python.exe"
$requirementsPath = Join-Path $projectRoot "requirements.txt"
$guiRequirementsPath = Join-Path $projectRoot "requirements-gui.txt"
$requirementsStampPath = Join-Path $venvRoot ".football-gui-requirements.sha256"
if ($Port -eq 8765) {
    $candidatePorts = @(8765..8775)
}
else {
    $candidatePorts = @($Port)
}

function Write-Step {
    param([string]$Message)
    Write-Host "[Football GUI] $Message" -ForegroundColor Cyan
}

function Test-Python311 {
    if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
        return $false
    }

    try {
        & $pythonPath -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" *> $null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

function Get-UvCommand {
    $command = Get-Command "uv.exe" -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        $command = Get-Command "uv" -ErrorAction SilentlyContinue
    }
    if ($null -eq $command) {
        return $null
    }
    return $command.Source
}

function Get-FileSha256 {
    param([string]$Path)

    $stream = [System.IO.File]::OpenRead($Path)
    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    try {
        return ([System.BitConverter]::ToString($sha256.ComputeHash($stream))).Replace("-", "")
    }
    finally {
        $sha256.Dispose()
        $stream.Dispose()
    }
}

function Get-RequirementsFingerprint {
    foreach ($path in @($requirementsPath, $guiRequirementsPath)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "缺少依赖文件：$path"
        }
    }

    $parts = foreach ($path in @($requirementsPath, $guiRequirementsPath)) {
        $name = [System.IO.Path]::GetFileName($path)
        $hash = Get-FileSha256 -Path $path
        "${name}:${hash}"
    }
    $bytes = [System.Text.Encoding]::UTF8.GetBytes(($parts -join "`n"))
    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    try {
        return ([System.BitConverter]::ToString($sha256.ComputeHash($bytes))).Replace("-", "")
    }
    finally {
        $sha256.Dispose()
    }
}

function Test-RequiredPackages {
    if (-not (Test-Python311)) {
        return $false
    }

    $check = "from importlib.metadata import distribution; [distribution(name) for name in ('Flask','matplotlib','pandas','scikit-learn','scipy','supervision','torch','ultralytics','openvino','roboflow','opencv-python')]"
    try {
        & $pythonPath -c $check *> $null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

function Test-FootballGui {
    param([int]$CandidatePort)

    try {
        $requestParameters = @{
            Uri = "http://127.0.0.1:$CandidatePort/api/catalog"
            Method = "Get"
            TimeoutSec = 1
        }
        $catalog = Invoke-RestMethod @requestParameters
        return (
            $null -ne $catalog -and
            $null -ne $catalog.PSObject.Properties["videos"] -and
            $null -ne $catalog.PSObject.Properties["models"] -and
            $null -ne $catalog.PSObject.Properties["defaults"]
        )
    }
    catch {
        return $false
    }
}

function Test-PortAvailable {
    param([int]$CandidatePort)

    $listener = [System.Net.Sockets.TcpListener]::new(
        [System.Net.IPAddress]::Loopback,
        $CandidatePort
    )
    try {
        $listener.Start()
        return $true
    }
    catch {
        return $false
    }
    finally {
        try {
            $listener.Stop()
        }
        catch {
            # The listener never started, so there is nothing to stop.
        }
    }
}

function Open-Gui {
    param([int]$GuiPort)
    $url = "http://127.0.0.1:$GuiPort"
    Write-Step "检测到已运行的 GUI，正在打开 $url"
    Start-Process $url
}

function Ensure-PythonEnvironment {
    $uvCommand = Get-UvCommand

    if (-not (Test-Python311)) {
        if ($null -eq $uvCommand) {
            throw "本地 .venv 不可用，并且找不到 uv。请先安装 uv：https://docs.astral.sh/uv/"
        }

        Write-Step "正在自动重建 Python 3.11 环境；首次运行可能需要联网下载。"
        & $uvCommand venv --python 3.11 --clear $venvRoot | Out-Host
        if ($LASTEXITCODE -ne 0 -or -not (Test-Python311)) {
            throw "Python 3.11 环境创建失败，请检查网络连接和磁盘空间。"
        }
    }

    $fingerprint = Get-RequirementsFingerprint
    $installedFingerprint = ""
    if (Test-Path -LiteralPath $requirementsStampPath -PathType Leaf) {
        $installedFingerprint = (Get-Content -LiteralPath $requirementsStampPath -Raw).Trim()
    }

    if ($installedFingerprint -eq $fingerprint -and (Test-RequiredPackages)) {
        Write-Step "Python 环境和依赖已经就绪。"
        return
    }

    Write-Step "正在安装或更新项目依赖；首次运行可能耗时较长。"
    if ($null -ne $uvCommand) {
        & $uvCommand pip install --python $pythonPath -r $guiRequirementsPath | Out-Host
    }
    else {
        & $pythonPath -m pip install -r $guiRequirementsPath | Out-Host
    }
    if ($LASTEXITCODE -ne 0 -or -not (Test-RequiredPackages)) {
        throw "依赖安装失败，请检查网络连接、代理设置和可用磁盘空间。"
    }

    Set-Content -LiteralPath $requirementsStampPath -Value $fingerprint -Encoding Ascii -NoNewline
    Write-Step "依赖安装完成。"
}

function Find-AvailablePort {
    foreach ($candidatePort in $candidatePorts) {
        if (Test-PortAvailable -CandidatePort $candidatePort) {
            return $candidatePort
        }
    }
    if ($Port -eq 8765) {
        throw "端口 8765 到 8775 均被占用，无法启动 GUI。"
    }
    throw "端口 $Port 已被占用，无法启动 GUI。"
}

function Start-FootballGui {
    Set-Location -LiteralPath $projectRoot

    foreach ($candidatePort in $candidatePorts) {
        if (Test-FootballGui -CandidatePort $candidatePort) {
            if (-not $NoBrowser) {
                Open-Gui -GuiPort $candidatePort
            }
            return 0
        }
    }

    Ensure-PythonEnvironment

    $selectedPort = Find-AvailablePort
    if ($selectedPort -ne $Port) {
        Write-Step "端口 $Port 已被其他程序占用，改用端口 $selectedPort。"
    }

    $serverPath = Join-Path $projectRoot "gui_server.py"
    if (-not (Test-Path -LiteralPath $serverPath -PathType Leaf)) {
        throw "缺少 GUI 服务文件：$serverPath"
    }

    Write-Step "正在启动 http://127.0.0.1:$selectedPort"
    Write-Host "关闭此窗口或按 Ctrl+C 即可停止 GUI。"
    $arguments = @($serverPath, "--port", [string]$selectedPort)
    if ($NoBrowser) {
        $arguments += "--no-browser"
    }
    & $pythonPath @arguments | Out-Host
    return $LASTEXITCODE
}

function Invoke-FootballGuiLauncher {
    try {
        $exitCode = Start-FootballGui
        if ($exitCode -ne 0) {
            throw "GUI 服务已退出，退出码：$exitCode"
        }
        exit 0
    }
    catch {
        Write-Host ""
        Write-Host "[错误] $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "你可以保留此窗口并把上面的错误信息发给开发者。"
        if ($env:FOOTBALL_GUI_NO_PAUSE -ne "1") {
            [void](Read-Host "按回车键关闭窗口")
        }
        exit 1
    }
}

Invoke-FootballGuiLauncher
