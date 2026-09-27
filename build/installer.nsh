; Uninstall: stop the llama-server this app started (pid file), never other llama-server processes.
; userData (%APPDATA%\local-ai-optimizer: database, settings, downloaded runtime) is kept on purpose.
!macro customUnInstall
  ClearErrors
  FileOpen $0 "$APPDATA\local-ai-optimizer\llama-server.pid" r
  IfErrors lao_no_pid
  FileRead $0 $1
  FileClose $0
  nsExec::Exec 'taskkill /F /FI "PID eq $1" /FI "IMAGENAME eq llama-server.exe"'
  lao_no_pid:
!macroend
