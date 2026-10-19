"""Read/patch only named literal settings in the local working config.

No execution of config code and no production paths. Byte spans preserve all
unrelated code, comments and CRLFs. Factory defaults always come from original.
"""
import ast
import hashlib
import math
import time
from pathlib import Path

FIELDS = {
    'pulse_ms': (1, 10000),
    'range_left': (.1, 5), 'range_right': (.1, 5),
    'start_left': (0, 1440), 'start_right': (0, 1440),
    'hold_left': (0, 1440), 'hold_right': (0, 1440),
    'reset_left': (0, 1440), 'reset_right': (0, 1440),
}


def _nodes(raw, site):
    if site not in ('taping4', 'taping7'):
        raise ValueError('지원하지 않는 호기입니다')
    tree = ast.parse(raw.decode('utf-8-sig'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ShiftSettings')
    fields = {n.target.id: n.value for n in cls.body if isinstance(n, ast.AnnAssign)}
    def literal_default(name):
        n = fields[name]
        if isinstance(n, ast.Call):
            factory = next(k.value for k in n.keywords if k.arg == 'default_factory')
            if not isinstance(factory, ast.Lambda):
                raise ValueError('직접 수정된 설정은 코드 편집기에서 확인하세요')
            n = factory.body
        ast.literal_eval(n)  # Reject arbitrary expressions; never execute them.
        return n
    sites = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'SITE_CONFIGS' for t in n.targets))
    site_node = dict(zip((ast.literal_eval(k) for k in sites.keys), sites.values))[site]
    hold = dict(zip((ast.literal_eval(k) for k in site_node.keys), site_node.values))['swing_term_angles']
    return {
        'pulse': literal_default('default_swing_shift_strength'),
        'range': literal_default('swing_range_factors'),
        'start': literal_default('angle_thresholds'),
        'hold': hold,
        'reset': literal_default('reset_delay_angles'),
    }


def read(path, site):
    raw = Path(path).read_bytes()
    nodes = _nodes(raw, site)
    values = {'pulse_ms': ast.literal_eval(nodes['pulse'])}
    for name in ('range', 'start', 'hold', 'reset'):
        pair = ast.literal_eval(nodes[name])
        if len(pair) != 2:
            raise ValueError('좌우 설정은 두 값이어야 합니다')
        values[name+'_left'], values[name+'_right'] = pair
    return {'values': values, 'hash': hashlib.sha256(raw).hexdigest()}


def patch(path, site, values, expected_hash, backups):
    path = Path(path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError('설정 코드가 다른 곳에서 바뀌었습니다. 스윙 설정의 새로 읽기를 누르세요.')
    if set(values) != set(FIELDS):
        raise ValueError('스윙 설정 항목을 모두 입력하세요')
    clean = {}
    for key, (lo, hi) in FIELDS.items():
        v = float(values[key])
        if not math.isfinite(v) or not lo <= v <= hi or (key == 'pulse_ms' and not v.is_integer()):
            raise ValueError(f'{key}: {lo} ~ {hi} 범위로 입력하세요')
        clean[key] = int(v) if key == 'pulse_ms' else v
    current = read(path, site)
    if current['values'] == clean:
        return {**current, 'changed': False}
    nodes = _nodes(raw, site)
    lines = raw.splitlines(keepends=True)
    bom = 3 if raw.startswith(b'\xef\xbb\xbf') else 0
    replacements = []
    for key, n in nodes.items():
        value = clean['pulse_ms'] if key == 'pulse' else [clean[key+'_left'], clean[key+'_right']]
        if ast.literal_eval(n) == value:
            continue
        before = sum(len(s) for s in lines[:n.lineno-1])+n.col_offset+(bom if n.lineno == 1 else 0)
        after = sum(len(s) for s in lines[:n.end_lineno-1])+n.end_col_offset+(bom if n.end_lineno == 1 else 0)
        replacements.append((before, after, repr(value).encode('utf-8')))
    updated = raw
    for begin, end, value in sorted(replacements, reverse=True):
        updated = updated[:begin]+value+updated[end:]
    ast.parse(updated.decode('utf-8-sig'))
    backups = Path(backups)
    backups.mkdir(exist_ok=True, parents=True)
    (backups/(str(time.time_ns())+'-motion-config.py')).write_bytes(raw)
    path.write_bytes(updated)
    return {**read(path, site), 'changed': True}
