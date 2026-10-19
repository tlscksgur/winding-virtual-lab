"""Offline closed loop: copied Python controller -> assumed plant -> ideal sensors.

No production service, camera inference, PLC driver or native Saige module is
imported. Mechanical response and orthographic camera calibration are explicit
assumptions. This is a control experiment, not a validated digital twin.
"""
from __future__ import annotations
import abc, enum, importlib.util, json, math, sys, types
from collections import deque
from pathlib import Path
import numpy as np
from scenario_worker import load_definitions
from safety import install_offline_guard
from worker import write_json
from motion_settings import read as read_motion_settings
from loadcell_model import contact as loadcell_contact

DEFAULTS = dict(flange=2400, core=1200, width=1900, cable=60, shaft=2600,
                guide_height=500, guide_distance=2000, rpm=6, speed=12,
                delay=.7, inertia=.45, response=.05, ppm=.25,
                seconds=2400, start='full', calls=1, loadcell='off')

def parameters(values):
    p={**DEFAULTS,**{k:v for k,v in values.items() if k in DEFAULTS}}
    limits={'flange':(400,6000),'core':(150,5000),'width':(200,5000),
            'cable':(10,180),'shaft':(250,6000),'guide_height':(100,4000),
            'guide_distance':(300,8000),'rpm':(.1,30),'speed':(1,150),
            'delay':(0,10),'inertia':(.02,5),'response':(.05,5),
            'ppm':(.05,1),'seconds':(20,3600),'calls':(.25,5)}
    for k,(lo,hi) in limits.items():
        p[k]=float(p[k])
        if not math.isfinite(p[k]) or not lo<=p[k]<=hi:
            raise ValueError(f'{k}: {lo} ~ {hi} 범위로 입력하세요')
    if p['core']+2*p['cable']>=p['flange']:raise ValueError('날개 외경은 몸통 지름 + 케이블 2개보다 커야 합니다')
    if p['width']<4*p['cable']:raise ValueError('권취 폭은 케이블 외경의 4배 이상으로 입력하세요')
    if p['shaft']<=p['flange']/2+100:raise ValueError('축 높이는 날개 반지름보다 100mm 이상 높아야 합니다')
    if math.hypot(p['shaft']-p['guide_height'],p['guide_distance'])<=p['flange']/2+p['cable']:
        raise ValueError('가이드가 드럼에 너무 가깝습니다. 가이드 거리·높이를 조정하세요')
    if p['start'] not in ('right','left','full'):raise ValueError('시작 위치 오류')
    if p['loadcell'] not in ('off','underroller','robot'):raise ValueError('로드셀 보정 방식 오류')
    return p

class Plant:
    """Reduced-order winding model, mm/s, continuous time, no controller states.

    New cable follows the last turn's natural pitch plus lateral tension toward
    the fixed guide. A flange blocks axial motion; one turn climbs to the next
    layer. Sufficient inward guide pull then reverses the physical lay direction.
    Relay retrigger extends the ON interval; reverse cancels the opposite output.
    """
    def __init__(self,p):
        self.p=p;self.t=0.;self.q=0.;self.v=0.;self.layer=0;self.crossings=0
        self.limit=(p['width']-p['cable'])/2
        self.direction=-1 if p['start']=='left' else 1
        self.c=self.direction*(self.limit-1.8*p['cable']) if p['start']!='full' else -self.limit
        self.x=-self.c;self.initial_c=self.c;self.initial_dir=self.direction
        self.edge_turns=0.;self.climbing=False;self.anchor=self.c
        self.relay=0;self.off=0.;self.queue=[];self.commands=[]
        self.max_angle=0.;self.overlap_turns=0.;self.travel=0.;self.last_c=self.c
        self.max_velocity=0.;self.end_hits=0;self.layer_full=False
        self.edge_layers={-1:0,1:0}
        self.full_passes=0;self.last_edge=-self.direction if p['start']=='full' else None
        self.samples=[];self.deposition=[]
        self.robot_x=0.;self.robot_v=0.;self.lc_drive=0;self.lc_sensed=0;self.lc_queue=[]
        self.lc_hits=0;self.lc_conflict_seconds=0.;self.lc_max_offset=0.


    @property
    def radius(self):
        rise=min(1.,self.edge_turns) if self.climbing else 0.
        return self.p['core']/2+self.p['cable']*(.5+self.layer+rise)

    @property
    def contact_angle(self):
        dy=self.p['guide_height']-self.p['shaft'];dz=self.p['guide_distance']
        # Lower tangent: cable approaches from the guide below the drum.
        # Render with positive X rotation and phase a - q*2*pi.
        return math.atan2(dz,dy)+math.acos(min(.999,self.radius/math.hypot(dy,dz)))

    def command(self,result):
        if not result.get('require_drum_shift'):return
        d=1 if result['drum_moving_direction']=='left_to_right' else -1
        duration=max(0,min(10,float(result.get('shift_strength',3000))/1000))
        self.queue.append((self.t+self.p['delay'],d,duration))
        self.commands.append(dict(t=round(self.t,3),direction=d,duration=duration,
                                  state=result['process_state'],x=round(self.x,3)))

    def loadcell(self):
        return loadcell_contact(self.p,self.radius,self.x,self.c,self.robot_x)

    def step(self,dt):
        for event in self.queue[:]:
            when,d,duration=event
            if self.t+1e-8>=when:
                self.off=max(self.off,self.t+duration) if self.relay==d else self.t+duration
                self.relay=d;self.queue.remove(event)
        if self.t>=self.off:self.relay=0
        lc=self.loadcell();self.lc_max_offset=max(self.lc_max_offset,abs(lc['offset']))
        mode=self.p['loadcell']
        if mode!='off':
            signal=lc['signal']
            if signal!=self.lc_sensed:
                if signal:self.lc_hits+=1
                self.lc_queue.append((self.t+self.p['delay'],signal))
                self.lc_sensed=signal
            while self.lc_queue and self.lc_queue[0][0]<=self.t+1e-8:
                _,self.lc_drive=self.lc_queue.pop(0)
        # Explicit hypothesis: a delayed load-cell correction has priority over
        # the AI relay when it acts on the same underroller drive. Not PLC code.
        pressure_relay=-self.lc_drive if mode=='underroller' else 0
        effective_relay=pressure_relay or self.relay
        if pressure_relay and self.relay and pressure_relay!=self.relay:
            self.lc_conflict_seconds+=dt
        target=effective_relay*self.p['speed']
        robot_target=self.lc_drive*self.p['speed'] if mode=='robot' else 0.
        self.robot_v+=(robot_target-self.robot_v)*(1-math.exp(-dt/self.p['inertia']))
        self.robot_x=max(-self.p['width']/2,min(self.p['width']/2,self.robot_x+self.robot_v*dt))
        self.v+=(target-self.v)*(1-math.exp(-dt/self.p['inertia']))
        self.x+=self.v*dt;self.travel+=abs(self.v)*dt
        self.max_velocity=max(self.max_velocity,abs(self.v))
        dq=self.p['rpm']/60*dt;self.q+=dq
        pull=self.robot_x-self.x-self.c
        if self.climbing:
            self.edge_turns+=dq
            if self.edge_turns>=1:self.edge_layers[self.direction]=self.layer+1
            if self.edge_turns>1:self.overlap_turns+=dq
            if self.edge_turns>=1 and pull*self.direction < -.25*self.p['cable']:
                self.direction*=-1;self.layer+=1;self.crossings+=1
                self.climbing=False;self.edge_turns=0.
                self.layer_full=self.radius+self.p['cable']/2>self.p['flange']/2
        if not self.climbing:
            # Self laying is driven by revolutions. Guide pull acts on cable only
            # while cable is being laid, never by a scripted state transition.
            bias=max(-.8*self.p['cable'],min(.8*self.p['cable'],self.p['response']*pull))
            dc=(self.direction*self.p['cable']+bias)*dq
            proposed=self.c+dc
            if abs(proposed)>=self.limit and proposed*self.direction>0:
                self.c=self.direction*self.limit;self.climbing=True
                self.anchor=self.c;self.edge_turns=0.;self.end_hits+=1
                if self.last_edge==-self.direction:self.full_passes+=1
                self.last_edge=self.direction
            else:self.c=max(-self.limit,min(self.limit,proposed))
        angle=math.degrees(math.atan2(self.x+self.c,math.hypot(self.p['shaft']-self.p['guide_height'],self.p['guide_distance'])))
        self.max_angle=max(self.max_angle,abs(angle));self.t+=dt

    def sample(self,state='',result=None):
        p=self.p;lc=self.loadcell()
        return dict(robot_x=round(self.robot_x,3),lc_offset=round(lc['offset'],3),lc_left=round(lc['left'],4),lc_right=round(lc['right'],4),lc_signal=lc['signal'],lc_drive=self.lc_drive,
                    t=round(self.t,3),q=round(self.q,6),x=round(self.x,3),c=round(self.c,3),
                    r=round(self.radius,3),a=round(self.contact_angle,6),v=round(self.v,3),
                    layer=self.layer,direction=self.direction,relay=self.relay,state=state,
                    edge=round(self.edge_turns,3),angle=round(math.degrees(math.atan2(self.x+self.c,
                          math.hypot(p['shaft']-p['guide_height'],p['guide_distance']))),3),
                    tightening=float((result or {}).get('tightening_score',-1)),
                    swing=float((result or {}).get('drum_swing_score',-1)))

class Controller:
    """Load original class/function definitions without native library imports.

    Only camera geometry extraction is substituted. ShiftMonitor.process_frame,
    actual state classes, timers, score ratios, request directions and durations
    execute from the selected, hashed source snapshot.
    """
    def __init__(self,source,site,plant):
        self.plant=plant;self.messages=[];src=Path(source)/'winding'
        ns={'__name__':'offline_controller','deque':deque,'np':np,'Enum':enum.Enum,
            'ABC':abc.ABC,'abstractmethod':abc.abstractmethod,'TYPE_CHECKING':False,
            'MILLISECONDS_IN_SECOND':1000,'logging':types.SimpleNamespace(
                log=lambda s:self.messages.append(str(s)),log_metrics=lambda **k:None)}
        load_definitions(src/'modules/direction.py',ns,{'Direction'})
        load_definitions(src/'modules/process_state.py',ns)
        load_definitions(src/'utils/timer.py',ns)
        load_definitions(src/'utils/roi.py',ns,{'get_roi_side'})
        load_definitions(src/'utils/geometry.py',ns,{'find_winding_top_x'})
        load_definitions(src/'modules/base.py',ns)
        load_definitions(src/'modules/shift_state.py',ns)
        load_definitions(src/'modules/score.py',ns)
        load_definitions(src/'modules/shifting.py',ns)
        spec=importlib.util.spec_from_file_location('virtual_winding_settings',src/'engine/config.py')
        settings=importlib.util.module_from_spec(spec);sys.modules[spec.name]=settings;spec.loader.exec_module(settings)
        settings.CURRENT_SITE=site;config=settings.MonitoringConfig()
        self.ns=ns;self.m=ns['ShiftMonitor'](config);self.D=ns['Direction']
        self.direction=self.D.RIGHT_TO_LEFT if plant.direction<0 else self.D.LEFT_TO_RIGHT
        # Ideal orthographic camera. All coordinates share one scale and frame.
        self.ppm=plant.p['ppm'];self.origin=math.ceil(plant.p['width']*self.ppm*2+100)
        self.frame_width=2*self.origin+100
        m=self.m
        m.tightening_score_calculator._get_side_drum_current_x=lambda boxes,masks,d:self.pixel(
            plant.x+(plant.p['width']/2 if d.is_left_to_right() else -plant.p['width']/2))
        m.swing_score_calculator._get_drum_body_start_x=lambda masks,d:self.pixel(
            plant.x+plant.anchor+(plant.p['cable']/2 if d.is_right_to_left() else -plant.p['cable']/2))
        # Ideal height sensor: real confirmation window and angle gates still run.
        m.detect_edge_protrusion=lambda seg,roi_side:bool(
            (plant.edge_layers[1 if roi_side=='right' else -1]-
             plant.edge_layers[-1 if roi_side=='right' else 1])*plant.p['cable']*self.ppm>
             m.shift_config.min_pixel_for_protrusion)
        self.initial_box=None
        self.last_t=0.

    def pixel(self,x):return int(round(self.origin+x*self.ppm))

    def feedback(self):
        p=self.plant;scale=self.ppm;d=p.p['cable'];w=max(2,round(d*scale))
        cx=self.pixel(p.x+p.c);left=cx-w//2
        if left<1 or left+w>=self.frame_width-1:
            raise ValueError('가상 카메라 범위를 벗어났습니다. 이동 속도·지연 조건을 낮추세요')
        mask=np.zeros((24,self.frame_width),dtype=np.uint8);mask[1:23,left:left+w+1]=1
        # Use projected flange geometry. The right box's 1:3 point equals its
        # inner edge, preserving the copied distance-EMA computation.
        flange_l=self.pixel(p.x-p.p['width']/2);flange_r=self.pixel(p.x+p.p['width']/2)
        boxes={'left_drum':[flange_l-6,0,8,24], 'right_drum':[flange_r-2,0,8,24],
               'drum_body':[flange_l,0,max(1,flange_r-flange_l),22],
               'winding_cable':[left,1,w,22]}
        seg=types.SimpleNamespace(winding_cable_mask=mask,winding_cable_box=boxes['winding_cable'],
                drum_body_box=boxes['drum_body'],drum_body_mask=None,bounding_boxes=boxes,
                segmentation_masks={'winding_cable':mask})
        if self.initial_box is None:
            self.initial_box=boxes['winding_cable'][:]
            self.m.drum_shift_score_calculator.set_shifting_box(self.initial_box)
        progress=(p.c+p.limit)/(2*p.limit) if self.direction.is_left_to_right() else (p.limit-p.c)/(2*p.limit)
        # Left endpoint progress uses the incoming lay direction, not clock time.
        return seg,dict(cable_moving_direction=self.direction.to_str(),progress_rate=float(max(0,min(1,progress))))

    def tick(self):
        seg,direction=self.feedback()
        angle=self.plant.p['rpm']*6*(self.plant.t-self.last_t);self.last_t=self.plant.t
        result=self.m.process_frame(seg,direction,None,angle)
        if result.get('require_direction_change'):self.direction=self.direction.opposite()
        return result

def run(cfg):
    out=Path(cfg['output']);install_offline_guard(out)
    p=parameters(cfg.get('plant',{}));plant=Plant(p);control=Controller(cfg['source'],cfg.get('site','taping4'),plant)
    records=[];trace=[];deposition=[];events=[];result={};next_call=0.;next_sample=0.;next_ring=0.;last_state=None
    dt=1/60;fault=None;phase_travel={'swing':0.,'return':0.}
    for i in range(int(p['seconds']/dt)+1):
        if plant.t+1e-7>=next_call:
            result=control.tick();plant.command(result)
            record={**plant.sample(result['process_state'],result),**result,'time':round(plant.t,3)}
            records.append(record)
            if last_state!=result['process_state']:
                events.append({'t':round(plant.t,3),'state':result['process_state']})
                last_state=result['process_state']
            next_call+=1/p['calls']
        if plant.t+1e-7>=next_sample:
            trace.append(plant.sample(last_state,result));next_sample+=.1
        if plant.q+1e-8>=next_ring:
            deposition.append([round(plant.q,6),round(plant.c,3),round(plant.radius,3),round(plant.contact_angle,6)])
            next_ring+=1/72
        if plant.layer_full:fault='권취 반지름이 날개 반지름에 도달했습니다';break
        if abs(plant.x)>p['width']*1.5:fault='가상 대차 이동 한계를 넘었습니다';break
        if i==int(p['seconds']/dt):break
        before_x=plant.x;plant.step(dt)
        if last_state in ('first_swing','first_swing_start'):phase_travel['swing']+=abs(plant.x-before_x)
        elif last_state in ('second_swing','second_swing_start'):phase_travel['return']+=abs(plant.x-before_x)
        if i%3600==0:write_json(out/'progress.json',{'stage':'가상 권취 계산','completed':round(plant.t,1),'total':p['seconds']})
    summary={'mode':'simulation','offline':True,'ai_inference':False,'actual_python_controller':True,'site':cfg.get('site','taping4'),
             'assumed_mechanics':True,'frames':len(records),'seconds':round(plant.t,3),
             'turns':round(plant.q,3),'layer_changes':plant.crossings,'round_trips':plant.full_passes//2,
             'full_traverses':plant.full_passes,
             'loadcell_mode':p['loadcell'],'loadcell_contacts':plant.lc_hits,
             'loadcell_assumptions':{'plane_fraction':.22,'side_gap_cable_ratio':.16,
                 'contact_index':'overtravel / cable radius; not calibrated force',
                 'arbitration':'loadcell priority over AI in underroller mode',
                 'polarity':'right contact -> drum left / hand right'},
             'loadcell_conflict_seconds':round(plant.lc_conflict_seconds,3),
             'max_loadcell_offset':round(plant.lc_max_offset,3),
             'swing_travel_mm':round(phase_travel['swing'],2),'return_travel_mm':round(phase_travel['return'],2),
             'commands':len(plant.commands),'max_angle':round(plant.max_angle,3),
             'extra_edge_turns':round(plant.overlap_turns,3),'travel_mm':round(plant.travel,2),
             'max_velocity':round(plant.max_velocity,3),'fault':fault,'source':cfg['variant'],
             'turns_per_layer':round((p['width']-p['cable'])/p['cable'],3),
             'ideal_traverse_speed':round(p['cable']*p['rpm']/60,3),
             'initial_line_speed':round(2*math.pi*(p['core']/2+p['cable']/2)*p['rpm']/1000,3),
             'overall_height':p['shaft']+p['flange']/2,'states':list(dict.fromkeys(r['process_state'] for r in records))}
    with open(out/'results.jsonl','w',encoding='utf-8') as f:
        for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    try:motion=read_motion_settings(Path(cfg['source'])/'winding/engine/config.py',cfg.get('site','taping4'))['values']
    except (ValueError,KeyError,TypeError,StopIteration):motion={}
    write_json(out/'simulation.json',{'parameters':p,'site':cfg.get('site','taping4'),'summary':summary,'trace':trace,
               'deposition':deposition,'commands':plant.commands,'events':events,
               'control_settings':motion,
               'initial':{'c':plant.initial_c,'direction':plant.initial_dir}})
    write_json(out/'summary.json',summary)
    (out/'state-log.txt').write_text('\n'.join(control.messages),encoding='utf-8')
    write_json(out/'progress.json',{'stage':'완료','completed':p['seconds'],'total':p['seconds']})

if __name__=='__main__':
    cfg=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    try:run(cfg)
    except Exception as e:
        write_json(Path(cfg['output'])/'error.json',{'error':str(e),'type':type(e).__name__});raise
