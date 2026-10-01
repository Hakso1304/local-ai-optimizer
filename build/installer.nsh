; Create the same writable default download folder that the packaged app exposes via hub:dirs.
; Program Files/app.asar is not writable by a standard user; uninstall preserves downloaded models.
!macro customInstall
  CreateDirectory "$APPDATA\local-ai-optimizer\models"
!macroend

; Uninstall: stop the llama-server this app started, and only that one (W4c D11). The pid file records
; {pid, exePath, startedAt}; the process is killed only if its exe path matches and it started within 30 s of the
; record, so a reused pid is never touched. An old plain-number pid file can't be verified and is left alone.
; userData (%APPDATA%\local-ai-optimizer: database, settings, downloaded runtime) is kept on purpose.
; The PowerShell below is base64 (UTF-16LE) so NSIS never interprets its $ signs. Source:
; $f = Join-Path $env:APPDATA 'local-ai-optimizer\llama-server.pid'
; if (Test-Path $f) {
;   try {
;     $j = Get-Content $f -Raw | ConvertFrom-Json
;     $p = Get-Process -Id $j.pid -ErrorAction Stop
;     $dt = [math]::Abs(($p.StartTime.ToUniversalTime() - ([datetime]$j.startedAt).ToUniversalTime()).TotalSeconds)
;     if ($p.Path -and ($p.Path -ieq $j.exePath) -and $dt -le 30) { Stop-Process -Id $j.pid -Force }
;   } catch { }
; }
!macro customUnInstall
  nsExec::Exec 'powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass -EncodedCommand JABmACAAPQAgAEoAbwBpAG4ALQBQAGEAdABoACAAJABlAG4AdgA6AEEAUABQAEQAQQBUAEEAIAAnAGwAbwBjAGEAbAAtAGEAaQAtAG8AcAB0AGkAbQBpAHoAZQByAFwAbABsAGEAbQBhAC0AcwBlAHIAdgBlAHIALgBwAGkAZAAnAAoAaQBmACAAKABUAGUAcwB0AC0AUABhAHQAaAAgACQAZgApACAAewAKACAAIAB0AHIAeQAgAHsACgAgACAAIAAgACQAagAgAD0AIABHAGUAdAAtAEMAbwBuAHQAZQBuAHQAIAAkAGYAIAAtAFIAYQB3ACAAfAAgAEMAbwBuAHYAZQByAHQARgByAG8AbQAtAEoAcwBvAG4ACgAgACAAIAAgACQAcAAgAD0AIABHAGUAdAAtAFAAcgBvAGMAZQBzAHMAIAAtAEkAZAAgACQAagAuAHAAaQBkACAALQBFAHIAcgBvAHIAQQBjAHQAaQBvAG4AIABTAHQAbwBwAAoAIAAgACAAIAAkAGQAdAAgAD0AIABbAG0AYQB0AGgAXQA6ADoAQQBiAHMAKAAoACQAcAAuAFMAdABhAHIAdABUAGkAbQBlAC4AVABvAFUAbgBpAHYAZQByAHMAYQBsAFQAaQBtAGUAKAApACAALQAgACgAWwBkAGEAdABlAHQAaQBtAGUAXQAkAGoALgBzAHQAYQByAHQAZQBkAEEAdAApAC4AVABvAFUAbgBpAHYAZQByAHMAYQBsAFQAaQBtAGUAKAApACkALgBUAG8AdABhAGwAUwBlAGMAbwBuAGQAcwApAAoAIAAgACAAIABpAGYAIAAoACQAcAAuAFAAYQB0AGgAIAAtAGEAbgBkACAAKAAkAHAALgBQAGEAdABoACAALQBpAGUAcQAgACQAagAuAGUAeABlAFAAYQB0AGgAKQAgAC0AYQBuAGQAIAAkAGQAdAAgAC0AbABlACAAMwAwACkAIAB7ACAAUwB0AG8AcAAtAFAAcgBvAGMAZQBzAHMAIAAtAEkAZAAgACQAagAuAHAAaQBkACAALQBGAG8AcgBjAGUAIAB9AAoAIAAgAH0AIABjAGEAdABjAGgAIAB7ACAAfQAKAH0A'
!macroend
