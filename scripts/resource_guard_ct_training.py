from __future__ import annotations
import ctypes,time
from pathlib import Path
import psutil

LOG=Path(r'D:\国创项目\experiments\auto_continue_ctspine1k_v6\resource_guard.log')
THRESHOLD=1200*1024*1024
kernel32=ctypes.WinDLL('kernel32',use_last_error=True)
psapi=ctypes.WinDLL('psapi',use_last_error=True)
PROCESS_QUERY_INFORMATION=0x0400
PROCESS_SET_QUOTA=0x0100

def training_alive():
    for p in psutil.process_iter(['cmdline']):
        try:
            if 'src.modeling.train' in ' '.join(p.info['cmdline'] or []): return True
        except Exception: pass
    return False

def trim(pid):
    h=kernel32.OpenProcess(PROCESS_QUERY_INFORMATION|PROCESS_SET_QUOTA,False,pid)
    if not h: return False
    try: return bool(psapi.EmptyWorkingSet(h))
    finally: kernel32.CloseHandle(h)

LOG.parent.mkdir(parents=True,exist_ok=True)
with LOG.open('a',encoding='utf-8') as log:
    log.write(f'[{time.strftime("%F %T")}] guard start threshold_mb=1200\n'); log.flush()
    while training_alive():
        for p in psutil.process_iter(['pid','name','cmdline','memory_info']):
            try:
                cmd=' '.join(p.info['cmdline'] or [])
                rss=p.info['memory_info'].rss
                if p.info['name'].lower()=='msedgewebview2.exe' and 'clash-verge' in cmd and '--type=renderer' in cmd and rss>THRESHOLD:
                    ok=trim(p.info['pid'])
                    log.write(f'[{time.strftime("%F %T")}] trim pid={p.info["pid"]} rss_mb={rss/1048576:.1f} ok={ok}\n'); log.flush()
            except (psutil.NoSuchProcess,psutil.AccessDenied): pass
        time.sleep(30)
    log.write(f'[{time.strftime("%F %T")}] guard exit training_not_alive\n'); log.flush()