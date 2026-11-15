$ErrorActionPreference = 'Stop'
$recordPath = Join-Path $PSScriptRoot 'renderer.json'
if (-not (Test-Path -LiteralPath $recordPath)) { Write-Output 'No dedicated renderer record.'; exit 0 }
$record = Get-Content -LiteralPath $recordPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($record.purpose -ne 'apatite-20261022-map-renderer') { throw 'Unrecognized renderer record.' }
$expectedProfile = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot 'browser-profile'))
if (-not ([IO.Path]::GetFullPath($record.profile_path)).Equals($expectedProfile, [StringComparison]::OrdinalIgnoreCase)) { throw 'Renderer profile ownership mismatch.' }
$rendererProcess = Get-Process -Id ([int]$record.pid) -ErrorAction SilentlyContinue
if (-not $rendererProcess) { Write-Output 'Dedicated renderer already stopped.'; exit 0 }
if ($rendererProcess.ProcessName -ne 'msedge' -or [string]$rendererProcess.StartTime.ToUniversalTime().Ticks -ne [string]$record.start_ticks_utc) { throw 'PID was reused or process identity changed; leaving it untouched.' }
Stop-Process -Id $rendererProcess.Id
Write-Output "Stopped dedicated renderer PID $($rendererProcess.Id). No user browser was targeted."
