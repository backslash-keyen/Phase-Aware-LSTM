# Logs free physical RAM every 10 s and stops MATLAB if it falls below 0.5 GB.
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File ram_watch.ps1 [-Minutes 150] [-LogDir .]
# Stops early when <LogDir>\ram_watch.stop exists.
param([int]$Minutes = 45, [string]$LogDir = (Get-Location).Path, [double]$MinFreeGB = 0.5)
$log  = Join-Path $LogDir 'ram_watch.log'
$stop = Join-Path $LogDir 'ram_watch.stop'
$end  = [datetime]::UtcNow.AddMinutes($Minutes)
while ((-not (Test-Path $stop)) -and ([datetime]::UtcNow -lt $end)) {
    $os = Get-CimInstance Win32_OperatingSystem
    $free = [math]::Round($os.FreePhysicalMemory/1MB, 2)
    $m = @(Get-Process -Name MATLAB -ErrorAction SilentlyContinue)
    $line = '{0:u} freeGB={1} matlabProcs={2}' -f [datetime]::UtcNow, $free, $m.Count
    if ($free -lt $MinFreeGB -and $m.Count -gt 0) {
        $m | Stop-Process -Force
        $line += ' LOW-RAM: MATLAB STOPPED'
    }
    Add-Content -Path $log -Value $line
    Start-Sleep -Seconds 10
}
Add-Content -Path $log -Value ('{0:u} watcher exit' -f [datetime]::UtcNow)
