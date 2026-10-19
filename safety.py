"""Offline worker boundary. This module is installed before importing vendor code."""
import socket
import sys
import os
from pathlib import Path

def install_offline_guard(allowed_writes=None):
    writable=Path(allowed_writes).resolve() if allowed_writes else None
    def audit(event, args):
        if event in {'socket.connect','socket.bind','socket.getaddrinfo','subprocess.Popen','os.system'}:
            raise PermissionError('Offline winding worker: network/process launch blocked')
        if writable and event=='open' and not isinstance(args[0],int):
            mode,flags=args[1],args[2]
            if (mode and any(x in mode for x in 'wax+')) or (flags and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)):
                if not Path(args[0]).resolve().is_relative_to(writable):
                    raise PermissionError('Offline worker: writes outside this run are blocked')
        if writable and event in {'os.remove','os.rmdir','os.mkdir','os.chmod','os.utime','os.rename','os.link','os.symlink'}:
            targets=args[:2] if event in {'os.rename','os.link','os.symlink'} else args[:1]
            for target in targets:
                if not isinstance(target,int) and not Path(target).resolve().is_relative_to(writable):
                    raise PermissionError('Offline worker: filesystem changes outside this run are blocked')
    sys.addaudithook(audit)
    class OfflineSocket(socket.socket):
        def connect(self, *args, **kwargs):
            raise PermissionError('Offline worker: connect blocked')
        connect_ex = connect
        def sendto(self, *args, **kwargs):
            raise PermissionError('Offline worker: sendto blocked')
    socket.socket = OfflineSocket
