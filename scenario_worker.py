"""Unit scenario: execute actual copied state classes with synthetic monitor inputs.

This is explicitly not image inference or a mechanical simulation. Only the
state-transition layer is isolated, as in the vendor's state unit tests.
"""
from __future__ import annotations
import ast, enum, importlib.util, json, sys, types
from collections import deque
from pathlib import Path
from safety import install_offline_guard
from worker import write_json

def load_definitions(path,namespace,only=None):
    tree=ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]
    body.extend(n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and (only is None or n.name in only))
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(path),'exec'),namespace)

def run(cfg):
    import numpy as np
    out=Path(cfg['output']);install_offline_guard(out);src=Path(cfg['source'])/'winding'
    ns={'__name__':'lab_state_test','deque':deque,'np':np,'Enum':enum.Enum,'TYPE_CHECKING':False}
    load_definitions(src/'modules'/'direction.py',ns,{'Direction'})
    load_definitions(src/'modules'/'process_state.py',ns)
    load_definitions(src/'utils'/'timer.py',ns)
    messages=[];ns['logging']=types.SimpleNamespace(log=lambda text:messages.append(str(text)))
    load_definitions(src/'modules'/'shift_state.py',ns)
    spec=importlib.util.spec_from_file_location('lab_engine_settings',src/'engine'/'config.py')
    settings=importlib.util.module_from_spec(spec);sys.modules[spec.name]=settings;spec.loader.exec_module(settings)
    settings.CURRENT_SITE=cfg.get('site','taping4');configuration=settings.MonitoringConfig()
    D,P=ns['Direction'],ns['ProcessState'];records=[]
    class Monitor:
        def __init__(self):
            self.shift_alarm_interval=configuration.shift.shift_alarm_interval;self.shift_strength_second=max(1,round(configuration.common.shift_strength/1000));self.num_swing_alarms=0
            self.shift_config=configuration.shift
            self.tightening_score_calculator=types.SimpleNamespace(side_drum_shift_x_range=(0,10),reset=lambda:None)
            self.direction_to_tightening_score_threshold={D.LEFT_TO_RIGHT:self.shift_config.tightening_score_thresholds[1],D.RIGHT_TO_LEFT:self.shift_config.tightening_score_thresholds[0]}
            self.direction_to_swing_score_threshold={D.LEFT_TO_RIGHT:self.shift_config.swing_score_thresholds[0],D.RIGHT_TO_LEFT:self.shift_config.swing_score_thresholds[1]}
            self.direction_to_swing_term_angle={D.LEFT_TO_RIGHT:self.shift_config.swing_term_angles[0],D.RIGHT_TO_LEFT:self.shift_config.swing_term_angles[1]}
            self.direction_to_reset_delay_angle={D.LEFT_TO_RIGHT:self.shift_config.reset_delay_angles[0],D.RIGHT_TO_LEFT:self.shift_config.reset_delay_angles[1]}
            self.cumulative_rotation_angle=0;self.additional_swing_delay_angle=0;self._auto_swing_skipped=False
            self.process_state=P.WINDING;self.state=ns['WindingState'](5)
        def set_process_state(self,v): self.process_state=v
        def set_state(self,v): self.state=v
        def record_rotation_angle(self,v): self.cumulative_rotation_angle=v
        def set_reference_angle(self,v): self.reference=v
        def reset(self): self.cumulative_rotation_angle=0
        def _check_cable_position_by_avg_error(self,*args): return 0,False
    m=Monitor()
    def step(label,direction,**values):
        context={'cable_moving_direction':direction,'progress_rate':.999,'segmentation_results':object(),'cable_width':10,'distance_ema':14,'tightening_score':.02,'drum_swing_score':.1,'is_first_swing_ready':True,**values}
        events=m.state.handle_transition(m,context)
        records.append({'index':len(records),'time':len(records),'process_state':m.process_state.value,'state_index':3 if events.get('require_direction_change') else 1,'scenario_step':label,'cable_moving_direction':direction.value,'rotation_angle':m.cumulative_rotation_angle,'tightening_score':context['tightening_score'],'drum_swing_score':context['drum_swing_score'],'mode':'scenario','synthetic_input':True,'events':{k:str(v) for k,v in events.items()},'pulse':None})
    for d in [D.LEFT_TO_RIGHT,D.RIGHT_TO_LEFT]:
        m=Monitor();step('끝단 접근',d);m.cumulative_rotation_angle=80;step('밀착 시작 조건',d)
        step('밀착 목표 도달',d);d=d.opposite();step('돌출 판단 시작',d,is_first_swing_ready=False)
        step('돌출 확인',d,is_first_swing_ready=False);step('스윙 준비',d)
        step('1차 스윙',d,drum_swing_score=.6);step('1차 스윙 목표 도달',d)
        for i in range(4): step('이동 종료 대기',d)
        if hasattr(m.state,'target_reached'):m.state.target_reached=True
        step('스윙 목표 확인',d);m.cumulative_rotation_angle=80;step('2차 스윙 시작 조건',d)
        for i in range(10): step('복귀 후 대기',d)
        m.cumulative_rotation_angle=400;step('복귀 후 권취 재개',d)
    with open(out/'results.jsonl','w',encoding='utf-8') as f:
        for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    (out/'state-log.txt').write_text('\n'.join(messages),encoding='utf-8')
    states={r['process_state'] for r in records}
    write_json(out/'summary.json',{'frames':len(records),'mode':'scenario','synthetic_input':True,'ai_inference':False,'offline':True,'states':sorted(states),'requests':[],'source':cfg['variant']})
    write_json(out/'progress.json',{'stage':'완료','completed':len(records),'total':len(records)})

if __name__=='__main__':
    cfg=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    try:run(cfg)
    except Exception as e:
        write_json(Path(cfg['output'])/'error.json',{'error':str(e),'type':type(e).__name__});raise
