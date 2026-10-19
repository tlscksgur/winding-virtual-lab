"""Preserve the downloaded source and create a separate working copy. No remote writes."""
from pathlib import Path
import hashlib, json, shutil, zipfile

ROOT = Path(__file__).resolve().parent
downloads = Path('C:/Users/DXEL01/Downloads')
candidates = sorted(list(downloads.glob('source-main*.zip')) + list(downloads.glob('*.crdownload')), key=lambda p:p.stat().st_mtime, reverse=True)
source = None
for item in candidates:
    if zipfile.is_zipfile(item):
        with zipfile.ZipFile(item) as archive:
            if 'source-main/SaigeWindingMonitoring/winding/engine/api.py' in archive.namelist():
                source = item
                break
if source is None:
    raise SystemExit('Complete source ZIP not found')
(ROOT/'original').mkdir(parents=True, exist_ok=True)
archive_path = ROOT/'original'/'source-main.zip'
if not archive_path.exists():
    shutil.copyfile(source, archive_path)
with zipfile.ZipFile(archive_path) as archive:
    bad = archive.testzip()
    if bad:
        raise SystemExit('ZIP integrity error: '+bad)
    manifest = {}
    for entry in archive.infolist():
        if entry.is_dir():
            continue
        relative = Path(*Path(entry.filename).parts[1:])
        target = (ROOT/'original'/'source'/relative).resolve()
        if not target.is_relative_to((ROOT/'original'/'source').resolve()):
            raise SystemExit('Unsafe archive member')
        content = archive.read(entry)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise SystemExit('Original snapshot differs: '+str(relative))
        manifest[relative.as_posix()] = {'sha256':digest,'bytes':len(content)}
snapshot = {'source_url':'http://192.168.220.130/saige-winding/source', 'commit':'0e4fbf7f646a81749b188e088018b501c4de769b', 'archive_sha256':hashlib.sha256(archive_path.read_bytes()).hexdigest(),'files':manifest}
(ROOT/'original'/'manifest.json').write_text(json.dumps(snapshot,indent=2),encoding='utf-8')
working = ROOT/'working'/'SaigeWindingMonitoring'
if not working.exists():
    shutil.copytree(ROOT/'original'/'source'/'SaigeWindingMonitoring',working)
print(json.dumps({'files':len(manifest),'bytes':sum(x['bytes'] for x in manifest.values()),'archive_sha256':snapshot['archive_sha256'],'working':str(working)}))
