"""Runs the copied Saige engine against local files, with networking disabled."""
from __future__ import annotations
import json, math, os, sys, time, traceback
from pathlib import Path
from safety import install_offline_guard

def write_json(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False),encoding='utf-8');tmp.replace(path)

def clean(value):
    import numpy as np
    if isinstance(value,dict): return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean(v) for v in value]
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return clean(value.item())
    if isinstance(value,float) and not math.isfinite(value): return None
    return value

def encode_frame(image,path):
    import cv2
    h,w=image.shape[:2];scale=min(1,1280/w,960/h)
    if scale<1: image=cv2.resize(image,(round(w*scale),round(h*scale)))
    ok,buf=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,86])
    if not ok: raise RuntimeError('프레임 JPEG 저장 실패')
    path.write_bytes(buf.tobytes())

def run(config):
    folder=Path(config['output']);source=Path(config['source']);sys.path.insert(0,str(source))
    install_offline_guard(folder)
    import cv2
    import numpy as np
    progress=lambda **v: write_json(folder/'progress.json',v)
    seg=None
    if config['mode']!='preview':
        import torch
        torch.set_num_threads(4)
        from winding.engine import config as engine_config
        engine_config.CURRENT_SITE=config['site']
        engine_config.LoggingSettings.log_dir=None
        from winding.modules.segmentation import SegmentationEngine
        from winding.modules.checkpoint_handler import DEFAULT_PASSWORD
        progress(stage='모델 로딩',completed=0,total=config['max_frames'])
        seg=SegmentationEngine.build(checkpoint_path=config['checkpoint'],password=DEFAULT_PASSWORD,device='cpu')
    engine=None
    if config['mode']=='monitoring':
        from winding.engine.monitoring import WindingMonitoringEngine
        from winding.modules import AngleMonitor,DirectionMonitor,SpongeMonitor,AnomalyMonitor,ShiftMonitor
        c=engine_config.MonitoringConfig()
        engine=WindingMonitoringEngine(c,seg,AngleMonitor(c),DirectionMonitor(c),SpongeMonitor(c),AnomalyMonitor(c),ShiftMonitor(c))
    frames=folder/'frames';frames.mkdir(exist_ok=True)
    image_mode=Path(config['main']).suffix.lower() in {'.jpg','.jpeg','.png','.bmp'}
    cap=None if image_mode else cv2.VideoCapture(config['main'])
    side=None
    if cap is not None and not cap.isOpened(): raise RuntimeError('정면 영상 열기 실패')
    if config['mode']=='monitoring':
        if image_mode: raise RuntimeError('전체 권취 판단은 동기화된 정면·측면 영상이 필요합니다')
        side=cv2.VideoCapture(config['side'])
        if not side.isOpened(): raise RuntimeError('측면 영상 열기 실패')
        if abs(cap.get(cv2.CAP_PROP_FPS)-side.get(cv2.CAP_PROP_FPS))>.05: raise RuntimeError('정면·측면 영상 FPS가 다릅니다. 동기화된 영상을 사용하세요.')
    count=0;states={};requests=[];last_state=None
    total=1 if image_mode else min(config['max_frames'],max(1,math.ceil(config['duration']/config['interval'])))
    palette={'left_drum':(68,197,121),'right_drum':(255,167,67),'drum_body':(41,192,244),'winding_cable':(104,91,255)}
    try:
        with open(folder/'results.jsonl','w',encoding='utf-8') as out:
            for i in range(total):
                second=config['start']+i*config['interval'];start=time.perf_counter()
                if image_mode:
                    frame=cv2.imdecode(np.fromfile(config['main'],dtype=np.uint8),cv2.IMREAD_COLOR)
                    if frame is None: raise RuntimeError('정면 이미지 열기 실패')
                else:
                    cap.set(cv2.CAP_PROP_POS_MSEC,second*1000);ok,frame=cap.read()
                    if not ok: break
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                if engine:
                    side.set(cv2.CAP_PROP_POS_MSEC,second*1000);ok,side_frame=side.read()
                    if not ok: raise RuntimeError('측면 영상이 먼저 끝났습니다')
                    with engine.timers['segmentation']:
                        main_result=seg.inference_main_frame(rgb)
                        side_result=seg.inference_side_frame(cv2.cvtColor(side_frame,cv2.COLOR_BGR2RGB))
                    result=engine.monitor(main_result,side_result,cv2.cvtColor(side_frame,cv2.COLOR_BGR2RGB))
                elif seg:
                    main_result=seg.inference_main_frame(rgb)
                    result={'process_state':'segmentation_only','state_index':None,'detected':{k:v is not None for k,v in main_result.segmentation_contours.items()},'segmentation_contours':main_result.segmentation_contours,'is_ng':None}
                else:
                    main_result=None;result={'process_state':'video_preview','state_index':None,'is_ng':None,'ai_inference':False}
                state=result['process_state'];states[state]=states.get(state,0)+1
                pulse=None
                if engine and (result.get('state_index') in (2,3) or state!=last_state):
                    pulse={'time':second,'process_state':state,'state_index':result.get('state_index'),'direction':result.get('drum_moving_direction'),'milliseconds':result.get('shift_strength'),'destination':'LOG_ONLY'}
                    requests.append(pulse)
                last_state=state
                visual=frame.copy();overlay=frame.copy()
                for key,contour in (main_result.segmentation_contours.items() if main_result else []):
                    if contour is None: continue
                    pts=np.asarray(contour,dtype=np.int32).reshape(-1,1,2);color=palette.get(key,(200,200,200))
                    cv2.drawContours(overlay,[pts],-1,color,cv2.FILLED);cv2.drawContours(visual,[pts],-1,color,3)
                visual=cv2.addWeighted(visual,.77,overlay,.23,0)
                raw_name=f'{i:05d}-raw.jpg';overlay_name=f'{i:05d}-overlay.jpg'
                encode_frame(frame,frames/raw_name);encode_frame(visual,frames/overlay_name)
                record=clean({k:v for k,v in result.items() if k not in ('segmentation_contours','ng_boxes')})
                record.update(index=i,time=second,raw=raw_name,overlay=overlay_name,elapsed_ms=round((time.perf_counter()-start)*1000,1),pulse=pulse,mode=config['mode'])
                out.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');out.flush();count+=1
                progress(stage='영상 분석',completed=count,total=total,state=state)
    finally:
        if cap is not None:cap.release()
        if side is not None:side.release()
    if count==0: raise RuntimeError('처리된 프레임이 없습니다. 시작 시간과 영상 길이를 확인하세요.')
    write_json(folder/'summary.json',{'frames':count,'states':states,'requests':requests,'mode':config['mode'],'offline':True,'ai_inference':config['mode']!='preview','source':config['variant']})
    progress(stage='완료',completed=count,total=count)

if __name__=='__main__':
    cfg=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    try: run(cfg)
    except Exception as e:
        write_json(Path(cfg['output'])/'error.json',{'error':str(e),'type':type(e).__name__})
        traceback.print_exc();sys.exit(1)
