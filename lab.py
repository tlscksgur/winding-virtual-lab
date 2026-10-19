"""Loopback-only workbench. Never launches VIMS, PLC Agent, SQL or production services."""
from __future__ import annotations
import argparse, ast, csv, hashlib, io, json, mimetypes, os, shutil, subprocess, sys, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import motion_settings
from simulation_scenarios import catalog as scenario_catalog

ROOT=Path(__file__).resolve().parent
ORIGINAL=ROOT/'original'/'source'/'SaigeWindingMonitoring'
WORKING=ROOT/'working'/'SaigeWindingMonitoring'
EDITABLE=['winding/engine/config.py','winding/modules/shift_state.py','winding/modules/shifting.py','winding/modules/score.py','winding/modules/direction.py']
for folder in ['inputs','outputs','backups']:
    (ROOT/folder).mkdir(exist_ok=True)
JOBS={}
RUN_LOCK=threading.Lock()
MANIFEST=json.loads((ROOT/'original'/'manifest.json').read_text(encoding='utf-8'))
for saved in sorted((ROOT/'outputs').glob('*/job.json')):
    try:
        item=json.loads(saved.read_text(encoding='utf-8'))
        if item['status'] not in {'완료','실패','취소됨'}:item['status']='중단됨'
        JOBS[item['id']]=item
    except (ValueError,KeyError):pass

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def verify_original():
    base=ROOT/'original'/'source'
    changed=[name for name,item in MANIFEST['files'].items() if not (base/name).is_file() or digest(base/name)!=item['sha256']]
    return {'ok':not changed,'files':len(MANIFEST['files']),'changed':changed,'commit':MANIFEST['commit']}

def runtime_python():
    settings=ROOT/'runtime.json'
    if settings.exists():
        path=Path(json.loads(settings.read_text(encoding='utf-8'))['python'])
        if path.is_file(): return path
    return ROOT/'.venv'/'Scripts'/'python.exe'

def environment_status():
    report=ROOT/'environment-status.json'
    return json.loads(report.read_text(encoding='utf-8')) if report.exists() else {'ai_ready':False,'reason':'인계 실행 환경 준비 중','python':str(runtime_python())}

def local_input(value):
    if not value: return None
    if str(value).startswith(('\\\\','//')) or '://' in str(value): raise ValueError('로컬 파일만 사용할 수 있습니다')
    p=Path(value).resolve()
    if not p.is_file() or p.suffix.lower() not in {'.mp4','.mov','.avi','.mkv','.jpg','.jpeg','.png','.bmp'}: raise ValueError('영상 또는 이미지 파일을 선택하세요')
    return p

def snapshot(source,dest):
    # Binary inference libraries are copied, never executed by the web server.
    shutil.copytree(source/'winding',dest/'winding',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    hashes={p.relative_to(dest).as_posix():digest(p) for p in dest.rglob('*') if p.is_file()}
    (dest.parent/'source-hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')

def do_run(job, options):
    try:
        job['status']='검증 중'
        checked=verify_original()
        if not checked['ok']: raise RuntimeError('원본 보관본 해시가 다릅니다. 실행을 중단했습니다.')
        mode=options.get('mode','segmentation')
        if mode not in {'preview','segmentation','monitoring','scenario','simulation'}: raise ValueError('지원하지 않는 실행 모드')
        main=local_input(options.get('main')) if mode not in {'scenario','simulation'} else None
        side=local_input(options.get('side')) if mode=='monitoring' else None
        if mode not in {'scenario','simulation'} and main is None: raise ValueError('정면 영상 또는 이미지가 필요합니다')
        if mode=='monitoring' and side is None: raise ValueError('전체 권취 판단에는 동기화된 측면 영상이 필요합니다')
        if mode not in {'preview','scenario','simulation'} and not environment_status().get('ai_ready'): raise RuntimeError(environment_status().get('reason','AI 환경을 확인하세요'))
        variants=['baseline','working'] if options.get('compare') else ['working']
        job['variants']=variants
        for variant in variants:
            if job.get('cancel'): break
            job['status']='실행 중';job['variant']=variant
            folder=ROOT/'outputs'/job['id']/variant;folder.mkdir(parents=True)
            source=ORIGINAL if variant=='baseline' else WORKING
            snapshot(source,folder/'source')
            config={'mode':mode,'site':options.get('site','taping4'),'main':str(main) if main else '', 'side':str(side) if side else '',
                    'interval':max(.1,min(10,float(options.get('interval',1)))),'start':max(0,float(options.get('start',0))),
                    'duration':max(1,min(3600,float(options.get('duration',20)))),'max_frames':max(1,min(3000,int(options.get('max_frames',30)))),
                    'checkpoint':str(source/'assets'/'checkpoint'/'checkpoint_taihan_infer.ckpt'),
                    'device':'cpu','source':str(folder/'source'),'output':str(folder),'variant':variant,'offline':True,
                    'plant':options.get('plant',{})}
            (folder/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
            worker=ROOT/({'scenario':'scenario_worker.py','simulation':'simulation_worker.py'}.get(mode,'worker.py'))
            executable=ROOT/'.venv'/'Scripts'/'python.exe' if mode in {'scenario','preview','simulation'} else runtime_python()
            env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            with open(folder/'console.log','w',encoding='utf-8') as log:
                proc=subprocess.Popen([str(executable),str(worker),str(folder/'config.json')],cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                job['_process']=proc
                while proc.poll() is None:
                    if job.get('cancel'): proc.terminate()
                    progress=folder/'progress.json'
                    if progress.exists():
                        try: job['progress']=json.loads(progress.read_text(encoding='utf-8'))
                        except (OSError,ValueError): pass
                    time.sleep(.3)
                if proc.returncode and not job.get('cancel'):
                    error_file=folder/'error.json'
                    reason=json.loads(error_file.read_text(encoding='utf-8')).get('error') if error_file.exists() else '엔진 실행 실패. 실행 로그를 확인하세요.'
                    raise RuntimeError(reason)
            job.pop('_process',None)
        checked=verify_original();job['original_integrity']=checked
        if not checked['ok']:raise RuntimeError('실행 후 원본 보관본 해시 불일치')
        job['status']='취소됨' if job.get('cancel') else '완료'
    except Exception as exc:
        job['status']='실패';job['error']=str(exc)
    finally:
        job.pop('_process',None);job['finished_at']=time.time()
        folder=ROOT/'outputs'/job['id'];folder.mkdir(exist_ok=True)
        (folder/'job.json').write_text(json.dumps(job,ensure_ascii=False,indent=2),encoding='utf-8')
        RUN_LOCK.release()

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def send_bytes(self,data,kind='application/json; charset=utf-8',code=200):
        self.send_response(code);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)));self.send_header('X-Content-Type-Options','nosniff');self.send_header('Cache-Control','no-store')
        self.send_header('Content-Security-Policy',"default-src 'self' data: blob:; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; frame-ancestors 'none'")
        self.end_headers();self.wfile.write(data)
    def json(self,value,code=200): self.send_bytes(json.dumps(value,ensure_ascii=False,allow_nan=False).encode(),code=code)
    def query(self): return parse_qs(urlparse(self.path).query)
    def do_GET(self):
        try:
            path=urlparse(self.path).path;query=self.query()
            if path=='/api/status': return self.json({'offline':True,'original_commit':MANIFEST['commit'],'source_files':len(MANIFEST['files']),'environment':environment_status(),'editable':EDITABLE,'inputs':str(ROOT/'inputs'),'jobs':[{k:v for k,v in j.items() if not k.startswith('_')} for j in JOBS.values()]})
            if path=='/api/verify': return self.json(verify_original())
            if path=='/api/scenarios':
                from simulation_worker import DEFAULTS
                return self.json(scenario_catalog(DEFAULTS))
            if path=='/api/motion':
                site=query.get('site',['taping4'])[0];rel='winding/engine/config.py'
                return self.json({'site':site,'working':motion_settings.read(WORKING/rel,site),
                                  'original':motion_settings.read(ORIGINAL/rel,site)})
            if path=='/api/source':
                file=query.get('file',[EDITABLE[0]])[0]
                if file not in EDITABLE: raise ValueError('허용되지 않은 파일')
                return self.json({'file':file,'working':(WORKING/file).read_text(encoding='utf-8'),'baseline':(ORIGINAL/file).read_text(encoding='utf-8'),'changed':digest(WORKING/file)!=digest(ORIGINAL/file)})
            if path=='/api/results':
                ident=query.get('id',[''])[0];variant=query.get('variant',['working'])[0]
                if ident not in JOBS or variant not in {'working','baseline'}: raise ValueError('실행을 찾을 수 없습니다')
                folder=ROOT/'outputs'/ident/variant;results=folder/'results.jsonl'
                records=[]
                if results.exists():
                    for line in results.read_text(encoding='utf-8').splitlines():
                        try: records.append(json.loads(line))
                        except ValueError: pass
                summary=folder/'summary.json'
                return self.json({'records':records,'summary':json.loads(summary.read_text(encoding='utf-8')) if summary.exists() else None,'folder':str(folder)})
            if path=='/api/simulation':
                ident=query.get('id',[''])[0];variant=query.get('variant',['working'])[0]
                if ident not in JOBS or variant not in {'working','baseline'}: raise ValueError('실행을 찾을 수 없습니다')
                file=ROOT/'outputs'/ident/variant/'simulation.json'
                if not file.exists():return self.json({'error':'가상 권취 결과가 없습니다'},404)
                return self.send_bytes(file.read_bytes())
            if path=='/api/frame':
                ident=query.get('id',[''])[0];variant=query.get('variant',['working'])[0];name=query.get('name',[''])[0]
                if ident not in JOBS or variant not in {'working','baseline'} or '/' in name or '\\' in name: raise ValueError('잘못된 이미지')
                file=ROOT/'outputs'/ident/variant/'frames'/name
                if file.suffix!='.jpg' or not file.is_file(): return self.json({'error':'이미지 없음'},404)
                return self.send_bytes(file.read_bytes(),'image/jpeg')
            if path=='/api/sample':
                file=ROOT/'inputs'/'sample-main.jpg';return self.json({'main':str(file) if file.is_file() else ''})
            if path=='/reference.jpg':return self.send_bytes((ROOT/'inputs'/'sample-main.jpg').read_bytes(),'image/jpeg')
            files={'/':'sim.html','/sim.js':'sim.js','/sim-motion.js':'sim-motion.js','/sim-scenarios.js':'sim-scenarios.js','/sim.css':'sim.css','/sim-model.js':'sim-model.js',
                   '/three.min.js':'three.min.js','/reference-video.jpg':'reference-video.jpg','/replay':'index.html','/app.js':'app.js','/style.css':'style.css'}
            if path in files:
                p=ROOT/'web'/files[path];return self.send_bytes(p.read_bytes(),mimetypes.guess_type(p)[0] or 'text/plain')
            self.json({'error':'경로 없음'},404)
        except Exception as exc: self.json({'error':str(exc)},400)
    def do_POST(self):
        try:
            host=self.headers.get('Host','')
            if host not in {f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}: raise ValueError('로컬 요청만 허용')
            origin=self.headers.get('Origin')
            if origin and origin not in {f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'}: raise ValueError('다른 사이트 요청 차단')
            path=urlparse(self.path).path;size=int(self.headers.get('Content-Length','0'))
            if path=='/api/upload':
                name=Path(self.query().get('name',['video.mp4'])[0]).name
                if Path(name).suffix.lower() not in {'.mp4','.mov','.avi','.mkv','.jpg','.jpeg','.png','.bmp'} or not 0<size<2*1024**3: raise ValueError('지원하지 않는 파일 또는 크기')
                dest=ROOT/'inputs'/(uuid.uuid4().hex[:8]+'-'+name)
                with open(dest,'wb') as f:
                    remaining=size
                    while remaining:
                        chunk=self.rfile.read(min(1024*1024,remaining))
                        if not chunk: raise ValueError('파일 전송 중단')
                        f.write(chunk);remaining-=len(chunk)
                return self.json({'path':str(dest)})
            if size>2*1024*1024: raise ValueError('요청이 너무 큽니다')
            data=json.loads(self.rfile.read(size) or '{}')
            if path=='/api/motion':
                if RUN_LOCK.locked():raise ValueError('실험 중에는 스윙 설정을 저장할 수 없습니다')
                result=motion_settings.patch(WORKING/'winding/engine/config.py',data.get('site','taping4'),
                    data.get('values',{}),data.get('expected_hash'),ROOT/'backups')
                return self.json(result)
            if path=='/api/source':
                file=data.get('file');content=data.get('content','')
                if file not in EDITABLE or RUN_LOCK.locked(): raise ValueError('실행 중에는 수정할 수 없습니다')
                ast.parse(content,filename=file)
                previous=WORKING/file
                if content.replace('\r\n','\n')==previous.read_text(encoding='utf-8'):
                    return self.json({'saved':True,'working_only':True,'unchanged':True})
                backup=ROOT/'backups'/(str(time.time_ns())+'-'+Path(file).name);shutil.copyfile(previous,backup)
                previous.write_text(content,encoding='utf-8',newline='');return self.json({'saved':True,'working_only':True})
            if path=='/api/run':
                if not RUN_LOCK.acquire(blocking=False): raise ValueError('이미 실행 중입니다')
                ident=time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]
                job={'id':ident,'status':'준비 중','mode':data.get('mode'),'created_at':time.time(),'progress':{},'variants':[]};JOBS[ident]=job
                threading.Thread(target=do_run,args=(job,data),daemon=True).start();return self.json({'id':ident})
            if path=='/api/cancel':
                if data.get('id') in JOBS: JOBS[data['id']]['cancel']=True
                return self.json({'cancel_requested':True})
            self.json({'error':'경로 없음'},404)
        except Exception as exc: self.json({'error':str(exc)},400)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=18765);args=parser.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'Winding Lab: http://127.0.0.1:{args.port} | Offline / original source preserved',flush=True)
    server.serve_forever()
if __name__=='__main__': main()
