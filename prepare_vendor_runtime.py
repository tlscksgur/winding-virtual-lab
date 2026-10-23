from pathlib import Path
import hashlib,json,shutil,zipfile
ROOT=Path(__file__).resolve().parent
candidates=list(Path('C:/Users/example/Downloads').glob('*.crdownload'))+list(Path('C:/Users/example/Downloads').glob('python*.zip'))
candidate=None
for path in sorted(candidates,key=lambda p:p.stat().st_size,reverse=True):
    if not zipfile.is_zipfile(path):continue
    with zipfile.ZipFile(path) as z:
        if 'python/python.exe' in z.namelist() and any('SaigeToolkit/' in n for n in z.namelist()):candidate=path;break
if not candidate:raise SystemExit('Completed vendor Python archive not found')
archive=ROOT/'original'/'vendor-python.zip'
if not archive.exists():shutil.copyfile(candidate,archive)
dest=ROOT/'.runtime'/'vendor'
with zipfile.ZipFile(archive) as z:
    for i,member in enumerate(z.infolist()):
        p=(dest/member.filename).resolve()
        if not p.is_relative_to(dest.resolve()):raise RuntimeError('Unsafe archive path')
        if member.is_dir():p.mkdir(parents=True,exist_ok=True);continue
        if not p.exists():
            p.parent.mkdir(parents=True,exist_ok=True)
            with z.open(member) as inp,open(p,'wb') as out:shutil.copyfileobj(inp,out)
        if i%5000==0:print(f'Runtime extracted {i}/{len(z.infolist())}',flush=True)
(ROOT/'runtime.json').write_text(json.dumps({'python':str(dest/'python'/'python.exe'),'source_archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()},indent=2),encoding='utf-8')
(ROOT/'inputs').mkdir(exist_ok=True)
shutil.copyfile('C:/Users/example/Downloads/main_00155.jpg',ROOT/'inputs'/'sample-main.jpg')
print('Vendor runtime ready')
