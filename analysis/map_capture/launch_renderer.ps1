param([int]$Port = 9240)
$ErrorActionPreference = 'Stop'
$workDir = Split-Path -Parent $PSScriptRoot
$projectRoot = Split-Path -Parent $workDir
$profilePath = Join-Path $PSScriptRoot 'browser-profile'
$profileFull = [IO.Path]::GetFullPath($profilePath)
if (-not $profileFull.StartsWith([IO.Path]::GetFullPath($PSScriptRoot) + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw 'Renderer profile must stay inside this task directory.' }
$probe = [Net.Sockets.TcpClient]::new()
try { $probe.Connect('127.0.0.1', $Port); $occupied = $true } catch { $occupied = $false } finally { $probe.Dispose() }
if ($occupied) { throw "Port $Port is already in use. This launcher will not attach to an existing browser." }
$edgePath = 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
if (-not (Test-Path -LiteralPath $edgePath)) { $edgePath = 'C:\Program Files\Microsoft\Edge\Application\msedge.exe' }
if (-not (Test-Path -LiteralPath $edgePath)) { throw 'Microsoft Edge executable not found.' }
$arguments = @('--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check', '--remote-debugging-address=127.0.0.1', "--remote-debugging-port=$Port", ('--user-data-dir="{0}"' -f $profileFull), '--remote-allow-origins=*', 'about:blank')
$rendererProcess = Start-Process -FilePath $edgePath -ArgumentList $arguments -WindowStyle Hidden -PassThru
$rendererRecord = [ordered]@{ pid = $rendererProcess.Id; start_ticks_utc = [string]$rendererProcess.StartTime.ToUniversalTime().Ticks; port = $Port; profile_path = $profileFull; project_root = $projectRoot; work_dir = $workDir; purpose = 'apatite-20261022-map-renderer' }
$rendererRecord | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'renderer.json') -Encoding UTF8
Write-Output "Dedicated hidden map renderer started: PID $($rendererProcess.Id), port $Port"
