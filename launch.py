import subprocess,time,webbrowser,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PORT=18765
PYTHON=Path(sys.executable)
try:
    # Fail visibly before spawning a background server if Windows blocks the
    # embedded runtime's native select extension.
    import socket
    import select  # noqa: F401
except ImportError as exc:
    print('Winding Lab did not start: Windows application control blocked the embedded Python select module.')
    print(f'Detail: {exc}')
    print(f'Log: {ROOT / "server.log"}')
    raise SystemExit(1)
def online():
    try:
        with socket.create_connection(('127.0.0.1',PORT),timeout=.3):return True
    except OSError:return False
if not online():
    if not PYTHON.is_file():
        print(f'Winding Lab runtime is missing: {PYTHON}')
        raise SystemExit(1)
    with open(ROOT/'server.log','a',encoding='utf-8') as log:
        subprocess.Popen([str(PYTHON),str(ROOT/'lab.py'),'--port',str(PORT)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    for i in range(40):
        if online():break
        time.sleep(.1)
if not online():
    print(f'Winding Lab server did not start. Check log: {ROOT / "server.log"}')
    raise SystemExit(1)
webbrowser.open(f'http://127.0.0.1:{PORT}')
