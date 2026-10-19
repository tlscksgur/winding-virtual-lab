'use strict';
const $=id=>document.getElementById(id);let currentJob=null,records=[],comparison={},selected=0,playing=false,lastStatus=null;
const labels={winding:'정렬 권취',waiting_tightening:'밀착 대기',tightening_start:'밀착 시작',tightening:'밀착 진행',tightening_end:'밀착 종료',protrusion_detection_start:'돌출 판단 시작',protrusion_detection:'돌출 판단',protrusion_detection_end:'돌출 확인',waiting_for_first_swing:'스윙 대기',first_swing_start:'1차 스윙 시작',first_swing:'1차 스윙',first_swing_end:'1차 스윙 종료',swing_term:'스윙 유지',second_swing_start:'복귀 시작',second_swing:'복귀 진행',second_swing_end:'복귀 완료',auto_swing_skip_reset:'스윙 생략 후 대기',segmentation_only:'정면 분할 확인',video_preview:'영상 검토 · AI 없음'};
const num=n=>typeof n==='number'?n.toFixed(3):'—';const dir=d=>d==='left_to_right'?'좌 → 우':d==='right_to_left'?'우 → 좌':'—';
async function api(path,data){const r=await fetch(path,data===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});const j=await r.json();if(!r.ok)throw Error(j.error||'요청 실패');return j;}
function notice(message=''){$('notice').textContent=message;}
function tab(code){$('results-panel').hidden=code;$('code-panel').hidden=!code;for(const [id,on] of [['result-tab',!code],['code-tab',code]]){$(id).classList.toggle('active',on);$(id).setAttribute('aria-selected',on);}if(code)loadCode();}
$('code-tab').onclick=()=>tab(true);$('result-tab').onclick=()=>tab(false);
function mode(){const m=$('mode').value;$('video-fields').hidden=m==='scenario';$('side-fields').hidden=m!=='monitoring';$('mode-help').textContent=m==='scenario'?'실제 복사된 상태 전이 코드를 합성 입력으로 검사합니다. 영상 AI 결과와 구분됩니다.':m==='monitoring'?'동기화된 두 영상으로 방향·밀착·스윙·복귀를 판단합니다.':m==='preview'?'AI를 실행하지 않고 영상의 프레임과 시각을 확인합니다.':'실제 모델로 드럼·케이블 영역을 검출합니다.';if(m==='preview')$('compare').checked=false;}
$('mode').onchange=mode;
for(const slot of ['main','side'])$(slot+'-file').onchange=async e=>{const file=e.target.files[0];if(!file)return;notice('로컬 테스트 폴더로 파일을 복사하는 중…');try{const r=await fetch('/api/upload?name='+encodeURIComponent(file.name),{method:'POST',body:file});const j=await r.json();if(!r.ok)throw Error(j.error);$(slot+'-path').value=j.path;notice();}catch(e){notice(e.message);}};
$('sample').onclick=async()=>{const j=await api('/api/sample');$('main-path').value=j.main;$('mode').value=lastStatus?.environment?.ai_ready?'segmentation':'preview';mode();notice(j.main?'':'첨부 사진을 찾지 못했습니다. 파일을 직접 선택하세요.');};
$('verify').onclick=async()=>{try{$('verify-status').textContent='검사 중…';const j=await api('/api/verify');$('verify-status').textContent=j.ok?j.files+'개 파일 원본과 일치':'원본 보관본 변경 감지';}catch(e){notice(e.message);}};
async function loadCode(){try{const j=await api('/api/source?file='+encodeURIComponent($('code-file').value));$('editor').value=j.working;$('baseline-code').textContent=j.baseline;$('changed').textContent=j.changed?'수정본 변경 있음':'원본과 동일';}catch(e){notice(e.message);}}
$('code-file').onchange=loadCode;$('editor').oninput=()=>{$('changed').textContent='저장하지 않은 변경';};
$('save').onclick=async()=>{try{await api('/api/source',{file:$('code-file').value,content:$('editor').value});$('changed').textContent='수정용 복사본에 저장됨';notice();}catch(e){notice(e.message);}};
$('run').onclick=async()=>{try{notice();playing=false;const j=await api('/api/run',{mode:$('mode').value,main:$('main-path').value,side:$('side-path').value,site:$('site').value,start:+$('start').value,duration:+$('duration').value,interval:+$('interval').value,max_frames:+$('max-frames').value,compare:$('compare').checked});currentJob=j.id;records=[];selected=0;comparison={};$('empty').hidden=false;$('result-view').hidden=true;$('comparison').hidden=true;$('variant').value='working';tab(false);await poll();}catch(e){notice(e.message);}};
$('cancel').onclick=async()=>{if(currentJob)await api('/api/cancel',{id:currentJob});};
function draw(){if(!records.length)return;selected=Math.max(0,Math.min(selected,records.length-1));const r=records[selected];$('empty').hidden=true;$('result-view').hidden=false;const scenario=r.mode==='scenario',preview=r.mode==='preview';$('image-grid').hidden=scenario;$('image-grid').classList.toggle('preview',preview);$('overlay-frame').parentElement.hidden=preview;$('scenario-note').hidden=!scenario;if(!scenario&&r.raw){const base='/api/frame?id='+encodeURIComponent(currentJob)+'&variant='+$('variant').value+'&name=';$('raw-frame').src=base+r.raw;if(!preview)$('overlay-frame').src=base+r.overlay;}$('frame').max=records.length-1;$('frame').value=selected;$('frame-info').textContent=(selected+1)+' / '+records.length+' · '+(r.mode==='scenario'?(r.index+1)+'회':Number(r.time||0).toFixed(1)+'s');$('state').textContent=labels[r.process_state]||r.process_state||'—';$('state-code').textContent=r.process_state||'';$('tightening').textContent=num(r.tightening_score);$('swing').textContent=num(r.drum_swing_score);$('move').textContent=r.pulse?dir(r.pulse.direction)+' '+(r.pulse.milliseconds??0)+'ms':'—';$('play').textContent=playing?'Ⅱ 일시정지':'▶ 재생';for(const row of $('events').children)row.classList.toggle('selected',+row.dataset.index===selected);}
function table(){document.querySelector('thead th').textContent=records[0]?.mode==='scenario'?'호출':'시각';const body=$('events');body.replaceChildren();records.forEach((r,i)=>{const tr=document.createElement('tr');tr.dataset.index=i;for(const text of [(r.mode==='scenario'?(r.index+1)+'회':Number(r.time||0).toFixed(1)+'s'),labels[r.process_state]||r.process_state,num(r.tightening_score),num(r.drum_swing_score),dir(r.drum_moving_direction)]){const td=document.createElement('td');td.textContent=text;tr.appendChild(td);}tr.onclick=()=>{selected=i;playing=false;draw();};body.appendChild(tr);});$('result-count').textContent=records.length+'개 프레임';}
async function loadResults(){
 if(!currentJob)return;
 const v=$('variant').value,j=await api('/api/results?id='+encodeURIComponent(currentJob)+'&variant='+v);
 const changed=j.records.length!==records.length;records=j.records;if(changed)table();$('output').textContent=j.folder;
 if(j.summary)comparison[v]={...j.summary,records:j.records};
 if(records.length)draw();
 if(comparison.baseline&&comparison.working){
  const a=comparison.baseline,b=comparison.working;
  const fields=['process_state','state_index','shift_strength','tightening_score','drum_swing_score','drum_moving_direction'];
  const differences=b.records.filter((r,i)=>!a.records[i]||fields.some(key=>typeof r[key]==='number'&&typeof a.records[i][key]==='number'?Math.abs(r[key]-a.records[i][key])>1e-6:r[key]!==a.records[i][key])).length+Math.max(0,a.records.length-b.records.length);
  $('comparison').hidden=false;$('comparison').textContent='비교 완료 · 원본 '+a.frames+' / 수정본 '+b.frames+'프레임 · 판단 차이 '+differences+'프레임 · 가상 요청 '+(a.requests?.length||0)+' → '+(b.requests?.length||0)+'회';
 }
}
$('variant').onchange=async()=>{records=[];selected=0;await loadResults();table();draw();};$('frame').oninput=e=>{selected=+e.target.value;playing=false;draw();};$('play').onclick=()=>{playing=!playing;if(playing&&selected===records.length-1)selected=0;draw();};
let playbackClock=0;setInterval(()=>{if(!playing||!records.length)return;playbackClock+=100*+$('speed').value;if(playbackClock>=600){playbackClock=0;selected++;if(selected>=records.length){selected=records.length-1;playing=false;}draw();}},100);
async function poll(){try{
 const j=await api('/api/status');lastStatus=j;
 $('source-info').textContent='원본 '+j.original_commit.slice(0,8)+' · '+j.source_files+'개 파일 보관';
 $('environment').textContent=j.environment.ai_ready?'AI 엔진 준비 완료 · CPU · 외부 연결 차단':'AI 추론 차단 · '+(j.environment.reason||'실행 환경 준비 중');
 if(!$('code-file').options.length)for(const file of j.editable){const o=document.createElement('option');o.value=file;o.textContent=file;$('code-file').appendChild(o);}
 for(const item of j.jobs){if(!item.created_at||Array.from($('history').options).some(o=>o.value===item.id))continue;const o=document.createElement('option');o.value=item.id;o.textContent=item.id+' · '+(item.mode==='scenario'?'상태 검사':item.mode==='preview'?'영상 검토':'AI 분석');$('history').appendChild(o);}
 const job=j.jobs.find(x=>x.id===currentJob);
 if(job){
  const active=!['완료','실패','취소됨','중단됨'].includes(job.status);
  $('run').disabled=active;$('save').disabled=active;$('cancel').hidden=!active;$('job-progress').hidden=false;
  const p=job.progress||{};$('job-status').textContent=job.status+' · '+(job.variant==='baseline'?'원본':'수정본')+' · '+(p.stage||'소스 스냅샷 준비')+(p.total?' '+p.completed+'/'+p.total:'');
  $('progress').value=p.total?100*p.completed/p.total:0;$('run-title').textContent=job.mode==='scenario'?'상태 전이 검사':job.mode==='preview'?'영상 프레임 검토':'프레임별 권취 분석';
  if(job.error)notice(job.error);await loadResults();
  if(job.status==='완료'&&job.variants?.includes('baseline')&&!comparison.baseline){const base=await api('/api/results?id='+currentJob+'&variant=baseline');if(base.summary)comparison.baseline={...base.summary,records:base.records};await loadResults();}
 }
}catch(e){notice(e.message);}}
$('history').onchange=async()=>{if(!$('history').value)return;currentJob=$('history').value;records=[];comparison={};selected=0;playing=false;notice();await poll();table();draw();};
mode();poll();setInterval(poll,1500);
