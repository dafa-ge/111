param(
    [int]$Port = 8765,
    [string]$Url = "http://127.0.0.1:8765",
    [int]$MaxSeconds = 60
)

$ErrorActionPreference = "SilentlyContinue"

$logDir = Join-Path (Split-Path $PSScriptRoot -Parent) "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
$log = Join-Path $logDir "panel-open.log"

function Write-OpenLog([string]$text) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $text"
    Add-Content -LiteralPath $log -Value $line -Encoding UTF8
}

$ready = $false
$elapsed = 0.0
$loops = [int]($MaxSeconds / 0.4)

for ($i = 0; $i -lt $loops; $i++) {
    $client = $null
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $async = $client.BeginConnect("127.0.0.1", $Port, $null, $null)
        if ($async.AsyncWaitHandle.WaitOne(800)) {
            $client.EndConnect($async)
            $ready = $true
        }
    } catch { }
    finally {
        if ($client) { $client.Close() }
    }
    $elapsed = $i * 0.4
    if ($ready) { break }
    Start-Sleep -Milliseconds 400
}

if ($ready) {
    Start-Process $Url
    Write-OpenLog ("opened {0} after {1:N1}s" -f $Url, $elapsed)
} else {
    Write-OpenLog ("timeout: port {0} not ready in {1}s" -f $Port, $MaxSeconds)
}