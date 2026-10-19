import socket,subprocess,time,webbrowser
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PORT=18765
def online():
    try:
        with socket.create_connection(('127.0.0.1',PORT),timeout=.3):return True
    except OSError:return False
if not online():
    with open(ROOT/'server.log','a',encoding='utf-8') as log:
        subprocess.Popen([str(ROOT/'.venv'/'Scripts'/'python.exe'),str(ROOT/'lab.py'),'--port',str(PORT)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    for i in range(40):
        if online():break
        time.sleep(.1)
webbrowser.open(f'http://127.0.0.1:{PORT}')
