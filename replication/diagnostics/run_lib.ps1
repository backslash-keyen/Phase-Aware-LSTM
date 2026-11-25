# Helpers for queued MATLAB runs (dot-source this file). One MATLAB job at a time,
# >= 2 GB free before starting (waits up to 20 min, then skips), one log line per event.
# Example, one fold per call so RAM and time are checked between folds:
#   . .\replication\diagnostics\run_lib.ps1
#   foreach ($i in 1..10) { if (ReadyToStart "fold $i") { RunMatlab "fold $i" "runReplication('ids',{'cwl_N2_target-domain_DeepRNN_phase-lstm'},'folds',$i,'variant','piInputPhaseLSTM')" "train_f$i.log" | Out-Null } }
param([string]$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path)   # repo root
$sc   = $Root
$nlog = Join-Path $sc 'replication\results\run_queue.log'

function Log($m) { $line = ('{0:u} {1}' -f [datetime]::UtcNow, $m); Add-Content $nlog $line; $line }

function FreeGB { [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 2) }

# Wait until no MATLAB runs and >= 2 GB free; up to 20 min. Returns $true if OK to start.
function ReadyToStart($tag) {
    $t0 = [datetime]::UtcNow
    while ($true) {
        $m = @(Get-Process -Name MATLAB -ErrorAction SilentlyContinue).Count
        $py = @(Get-Process -Name python -ErrorAction SilentlyContinue).Count
        $f = FreeGB
        if ($m -eq 0 -and $f -ge 2) { Log "[$tag] start check OK: freeGB=$f python=$py"; return $true }
        if (([datetime]::UtcNow - $t0).TotalMinutes -ge 20) { Log "[$tag] SKIPPED: freeGB=$f matlab=$m python=$py after 20 min wait"; return $false }
        Start-Sleep -Seconds 30
    }
}

# Runs one MATLAB batch command from the repo root with replication/ on the path.
function RunMatlab($tag, $cmd, $logfile) {
    $t0 = [datetime]::UtcNow
    cmd /c "matlab -batch ""cd('$sc'); addpath('replication'); $cmd"" > $logfile 2>&1"
    $code = $LASTEXITCODE
    Log ("[$tag] matlab exit=$code, {0:N1} min" -f ([datetime]::UtcNow - $t0).TotalMinutes)
    return $code
}
