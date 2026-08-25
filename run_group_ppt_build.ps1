$ErrorActionPreference='Continue'
$root=$PSScriptRoot
$log=Join-Path $root 'ppt_build_20260816.log'
try {
  & (Join-Path $root 'make_group_meeting_ppt_20260816.ps1') *>&1 | Out-File -FilePath $log -Encoding utf8
} catch {
  ($_ | Out-String) | Out-File -FilePath $log -Encoding utf8
}
