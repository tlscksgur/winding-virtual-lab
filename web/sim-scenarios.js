(()=>{'use strict';
const $=s=>document.querySelector(s),form=$('#conditions');
let catalog=null,selected=null,reference=null,busy=false,edited=false,applied={},previous={};
const labels={gap:'한 바퀴 사이 여백',coast:'OFF 후 잔류 이동',contact:'로드셀 접촉',conflict:'보정 방향 충돌',edge:'끝단 추가 감김',missed:'짧은 신호 미응답'};
const controls={rpm:['드럼 회전','rpm'],speed:['대차 이동','mm/s'],delay:['신호 지연','초'],inertia:['가감속 응답','초'],response:['케이블 횡응답','배율'],relay_mode:['릴레이 방식',''],loadcell:['로드셀 보정 대상',''],operating:['운영 신호값',''],seconds:['시험 시간','초'],start:['시작 위치',''],hold_left:['왼쪽 복귀 대기각','°'],hold_right:['오른쪽 복귀 대기각','°'],pulse_ms:['스윙·복귀 신호시간','ms'],range_left:['왼쪽 목표 간격 배율','×'],range_right:['오른쪽 목표 간격 배율','×'],start_left:['왼쪽 스윙 시작각','°'],start_right:['오른쪽 스윙 시작각','°'],reset_left:['왼쪽 복귀 후 대기각','°'],reset_right:['오른쪽 복귀 후 대기각','°']};
const effects={
 reference:{watch:'오작동을 넣지 않은 기준 계산입니다. 다른 상황을 실행할 때 이 결과와 차이를 비교합니다.',focus:['gap','coast','missed','conflict','edge']},
 delay:{watch:'신호를 늦추면 드럼이 먼저 감겨 케이블 간격이 벌어지거나 끝단에 더 쌓이는지 봅니다.',focus:['gap','edge']},
 coast:{watch:'이동 OFF 뒤 대차가 관성으로 더 움직이는 거리를 확인합니다. 잔류 이동이 크면 끝 위치를 지나칠 수 있습니다.',focus:['coast']},
 slow:{watch:'대차 속도가 드럼의 이론 정렬 속도보다 낮습니다. 한 바퀴 후 옆 줄로 얼마나 밀리는지 확인합니다.',focus:['gap','edge']},
 fast:{watch:'드럼 회전만 빨라집니다. 회전에 맞춰 대차가 따라가지 못하면 여백 또는 끝단 쏠림이 나타납니다.',focus:['gap','edge']},
 late_return:{watch:'복귀 대기각을 키웁니다. 복귀 시작이 늦어지는지 확인하세요. 변화가 없으면 앞선 자동 복귀 조건이 먼저 작동한 것입니다.',focus:['edge','gap']},
 short_pulse:{watch:'릴레이가 요청을 5초 뒤부터 받는다고 가정합니다. 그보다 짧은 ON 신호가 누락되는지 확인합니다.',focus:['missed']},
 loadcell:{watch:'로드셀 접촉 보정이 언더롤러를 구동한다고 가정합니다. AI 드럼 이동과 반대 명령이 겹치는 시간을 확인합니다.',focus:['contact','conflict']},
 robot:{watch:'같은 접촉 보정을 로봇 손으로 옮긴 대안입니다. 로드셀 접촉은 남고 드럼 이동 충돌이 줄어드는지 비교합니다.',focus:['contact','conflict']}
};
const metric={gap:['max_turn_gap_mm','mm'],coast:['max_coast_mm','mm'],missed:['missed_pulses','회'],conflict:['loadcell_conflict_seconds','초'],contact:['loadcell_contacts','회'],edge:['extra_edge_turns','회전']};
const read=()=>{const o={};for(const [k,v]of new FormData(form))o[k]=v;for(const i of document.querySelectorAll('[data-motion]'))o[i.dataset.motion]=i.value;return o;};
function pretty(k,v){const opts={relay_mode:{queued:'전체 신호 유지',on_delay:'ON 지연'},loadcell:{off:'미포함',underroller:'언더롤러',robot:'로봇 팔'},operating:{code:'코드 기본값',handover:'인계 운영값'},start:{right:'오른쪽 끝단',left:'왼쪽 끝단',full:'빈 드럼'}};if(opts[k])return opts[k][v]||v;const d=controls[k];return d?`${v}${d[1]?` ${d[1]}`:''}`:String(v);}
function announce(){
 const brief=$('#scenario-brief');
 if(!selected){brief.hidden=true;return;}
 brief.hidden=false;$('#scenario-brief-title').textContent=selected.label;
 $('#scenario-description').textContent=selected.description+(edited?' 직접 수정한 값도 포함되어 있습니다.':'');
 $('#scenario-expectation').textContent='이번에 확인할 현상: '+(effects[selected.id]?.watch||selected.description);
 const list=$('#scenario-changes');list.replaceChildren();let changed=0;
 const current=read(),base=edited?applied:previous;
 for(const [k,[label]]of Object.entries(controls)){if(base[k]===undefined||base[k]===null||String(base[k])===String(current[k]))continue;const li=document.createElement('li'),name=document.createElement('span'),value=document.createElement('b');name.textContent=label;value.textContent=`${pretty(k,base[k])} → ${pretty(k,current[k])}`;li.append(name,value);list.appendChild(li);changed++;}
 if(!changed){const li=document.createElement('li');li.className='no-changes';li.textContent='이번 선택으로 바뀐 값 없음 · 현재값이 이미 이 시험 조건과 같습니다.';list.appendChild(li);}
 for(const b of $('#scenario-buttons').querySelectorAll('button'))b.setAttribute('aria-pressed',String(b.dataset.scenario===selected.id));
}
async function choose(item){
 if(busy)return;busy=true;lock(true);$('#run').disabled=true;
 try{
  const before=read();
  await window.MotionSettings.stage(item.motion);
  const settings={...catalog.common,...item.plant,scenario:item.id};
  for(const [k,v]of Object.entries(settings)){const el=form.elements.namedItem(k);if(el)el.value=String(v);}
  selected=item;edited=false;previous=before;applied=read();announce();
  form.dispatchEvent(new Event('input',{bubbles:true}));window.MotionSettings.estimate();
  $('#run-status').textContent=item.label+' 조건을 불러왔습니다. 바뀐 설정을 확인하고 ‘조건 적용하고 실험’을 누르세요.';
 }catch(e){$('#scenario-description').textContent=e.message;}
 finally{busy=false;lock(false);$('#run').disabled=false;}
}
function lock(on){for(const b of $('#scenario-buttons').querySelectorAll('button'))b.disabled=on;}
function restore(p){selected=catalog?.scenarios.find(s=>s.id===p.scenario)||null;edited=false;previous=read();applied=read();form.elements.namedItem('scenario').value=selected?.id||'custom';announce();}
function changed(){if(selected){edited=true;form.elements.namedItem('scenario').value='custom';announce();}}
function comparable(a,b){return a&&b&&a.site===b.site&&['flange','core','width','cable','shaft','guide_height','guide_distance'].every(k=>Number(a.parameters?.[k])===Number(b.parameters?.[k]));}
function fmt(v,unit){return `${Number(v).toFixed(unit==='회'?0:3)}${unit}`;}
function showDiagnosis(data,variant){
 const box=$('#scenario-diagnosis'),id=data.parameters?.scenario||'custom',item=catalog?.scenarios.find(x=>x.id===id),guide=effects[id];
 box.hidden=false;$('#diagnosis-kicker').textContent=item?.label||'사용자 조건';
 const ref=reference?.[variant]||reference?.baseline||reference?.working;
 const canCompare=comparable(data,ref),s=data.summary||{},focus=guide?.focus||['gap','coast','missed','conflict','edge'];
 const cards=$('#diagnosis-metrics');cards.replaceChildren();let changed=false,definite=false;
 for(const key of focus){const def=metric[key];if(!def)continue;const [metricKey,unit]=def,value=Number(s[metricKey]||0),before=canCompare?Number(ref.summary?.[metricKey]||0):null,delta=canCompare?value-before:null;
  if(canCompare&&Math.abs(delta)>.0005)changed=true;
  if(['gap','coast','missed','conflict','edge'].includes(key)&&value>0)definite=true;
  const card=document.createElement('div');card.className='diagnostic-metric'+(value>0?' observed':'');
  const name=document.createElement('span');name.textContent=labels[key];const current=document.createElement('strong');current.textContent=fmt(value,unit);const compare=document.createElement('small');
  compare.textContent=canCompare?`기준 ${fmt(before,unit)} · 차이 ${delta>0?'+':''}${fmt(delta,unit)}`:'기준 계산 후 차이를 표시합니다';card.append(name,current,compare);cards.appendChild(card);
 }
 if(id==='late_return'){
  const eventAt=d=>(d?.events||[]).find(e=>e.state==='second_swing_start')?.t;
  const now=eventAt(data),before=canCompare?eventAt(ref):undefined;
  if(now!==undefined&&before!==undefined&&Math.abs(now-before)>.01)changed=true;
  const card=document.createElement('div');card.className='diagnostic-metric'+(now!==undefined?' observed':'');
  const name=document.createElement('span');name.textContent='복귀 시작 시점';const current=document.createElement('strong');current.textContent=now===undefined?'미발생':`${Number(now).toFixed(2)}초`;const compare=document.createElement('small');compare.textContent=before!==undefined?`기준 ${Number(before).toFixed(2)}초 · 차이 ${(now-before)>0?'+':''}${(now-before).toFixed(2)}초`:'기준 실행 후 시점 비교';card.append(name,current,compare);cards.appendChild(card);
 }
 const status=$('#diagnosis-status'),title=$('#diagnosis-title'),message=$('#diagnosis-message');
 if(id==='reference'){status.textContent='비교 기준';title.textContent='이 결과를 다른 문제 상황과 비교합니다';message.textContent='기준 조건 계산을 보관했습니다. 이제 문제 상황을 골라 실행하면 여백·잔류 이동·신호 누락 등의 차이를 함께 표시합니다.';box.dataset.state='good';}
 else if(id==='robot'&&Number(s.loadcell_contacts)>0&&Number(s.loadcell_conflict_seconds)===0){status.textContent='대안 동작 관찰';title.textContent='접촉은 감지되고, 드럼 충돌은 계산되지 않았습니다';message.textContent=`로드셀 접촉 ${s.loadcell_contacts}회 · 언더롤러와 반대 방향으로 겹친 시간 ${fmt(s.loadcell_conflict_seconds||0,'초')}. 로봇 보정 가정에서의 결과입니다.`;box.dataset.state='good';}
 else if(changed||(!canCompare&&definite)){status.textContent='현상 관찰';title.textContent=canCompare?'기준 조건과 다른 움직임이 확인됐습니다':'문제 상황의 관찰값이 발생했습니다';message.textContent=(guide?.watch||'표시된 값을 확인하세요.')+' 아래 수치의 기준 대비 차이를 보고, 주황색 수치 버튼을 눌러 해당 시점의 3D 동작을 확인하세요.';box.dataset.state='observed';}
 else if(!canCompare){status.textContent='기준 비교 필요';title.textContent='기준 계산이 없어 오작동 여부를 판정할 수 없습니다';message.textContent=(guide?.watch||'선택한 조건을 확인하세요.')+' 먼저 기준 조건을 실행한 뒤 이 상황을 다시 계산하면 미세한 차이도 증감값으로 표시합니다.';box.dataset.state='none';}
 else{status.textContent='차이 미검출';title.textContent='이번 계산에서는 기준과 구분되는 변화가 없었습니다';message.textContent=(guide?.watch||'선택한 조건을 확인하세요.')+' 설정은 적용됐지만 관찰값은 기준과 같습니다. 해당 제어 조건이 이번 운전 구간의 판단에 영향을 주지 않았을 수 있습니다.';box.dataset.state='none';}
 $('#diagnosis-reference').textContent=canCompare?'같은 치수·호기의 기준 조건과 비교했습니다. 기준값이 다르면 별도의 기준 실험을 먼저 실행하세요.':'기준 조건을 먼저 실행하면 각 항목의 기준값과 증감량을 함께 표시합니다. 수치는 보정되지 않은 가상 기계의 결과입니다.';
}
function show(data,jump,variant='baseline'){
 showDiagnosis(data,variant);const box=$('#observations');box.replaceChildren();const s=data.summary,points=data.observation_points||[];
 const heading=document.createElement('strong');heading.textContent='계산된 현상 · 누르면 3D가 해당 시점으로 이동';box.appendChild(heading);
 const keys=effects[data.parameters?.scenario]?.focus||['gap','coast','missed','conflict','contact','edge'];
 for(const key of keys){const def=metric[key];if(!def)continue;const [metricKey,unit]=def,value=s[metricKey]??0,point=points.find(p=>p.kind===key),b=document.createElement('button');
  b.type='button';b.textContent=`${labels[key]} ${fmt(value,unit)}`;b.disabled=!point;b.title=point?'관찰된 시점으로 이동':'이번 계산에서 해당 시점은 기록되지 않음';b.dataset.observed=String(Number(value)>0);
  if(point){b.onclick=()=>jump(point.t);b.dataset.time=point.t;}box.appendChild(b);
 }
 const note=document.createElement('small');note.textContent='여백은 같은 층에서 한 바퀴 전 위치와 비교한 값입니다. 수치는 가상 모델의 관찰값이며 현장 합격 기준은 아닙니다.';box.appendChild(note);
 $('#scenario-result').textContent=(catalog?.scenarios.find(x=>x.id===data.parameters.scenario)?.label||'사용자 조건')+' · '+(data.site==='taping7'?'7호기':'4호기');
}
function remember(results){if(results.baseline?.parameters.scenario==='reference'||results.working?.parameters.scenario==='reference')reference=results;const r=reference?.baseline||reference?.working;$('#reference-status').textContent=r?`기준 계산 보관: ${r.site==='taping7'?'7':'4'}호기 · ${r.parameters.seconds}초. 동일 치수에서 증감량을 비교합니다.`:'먼저 ‘기준 조건’을 실행하면 각 상황의 차이를 수치로 비교합니다.';}
async function init(){
 const r=await fetch('/api/scenarios');catalog=await r.json();if(!r.ok)throw Error(catalog.error||'시험 상황을 읽지 못했습니다');
 for(const item of catalog.scenarios){const b=document.createElement('button');b.type='button';b.dataset.scenario=item.id;b.textContent=item.label;b.title=item.description;b.setAttribute('aria-pressed','false');b.onclick=()=>choose(item);$('#scenario-buttons').appendChild(b);}
 $('#scenario-provenance').textContent=catalog.provenance;announce();
 form.addEventListener('input',e=>{if(e.target!==form)changed();});form.addEventListener('change',e=>{if(e.target!==form)changed();});
}
window.WindingScenarios={init,lock,restore,show,remember,changed,getReference:()=>reference};
})();

// Replay annotations use measured observation points; selecting a preset alone
// never manufactures a fault or changes the simulation's coordinates.
(()=>{'use strict';
const titles={gap:'케이블 사이 여백',coast:'정지 명령 뒤에도 대차 이동',missed:'짧은 이동 신호 미응답',contact:'로봇 손 로드셀 접촉',conflict:'AI 이동과 로드셀 보정 충돌',edge:'끝단에 추가 감김'};
const explain={gap:'화살표 끝은 한 바퀴 전 케이블과 지금 감긴 케이블 사이입니다. 같은 층에서 케이블 외경보다 벌어진 간격을 측정했습니다.',coast:'화살표가 가리키는 언더롤러·대차가 OFF 명령 이후에도 움직였습니다. 감속 응답으로 누적된 잔류 이동입니다.',missed:'이동 신호는 나왔지만 설정한 ON 지연보다 짧아서 대차가 받지 못했습니다. 표시 위치는 명령을 받는 대차입니다.',contact:'화살표 끝은 케이블을 감지하는 손 로드셀입니다. 접촉 자체는 고장이 아니며, 어떤 보정을 유발하는지 확인할 관찰점입니다.',conflict:'손 로드셀의 언더롤러 보정 방향이 같은 시점의 AI 이동 요청과 반대입니다. 접촉 위치를 표시했습니다. 제어 충돌은 가상 보정 가정의 결과입니다.',edge:'드럼 끝단에서 층을 올라가는 1회전을 넘어 추가로 감겼습니다. 화살표가 끝단에 케이블이 쌓이는 위치를 가리킵니다.'};
const focus={delay:['gap','edge'],coast:['coast'],slow:['gap','edge'],fast:['gap','edge'],late_return:['edge','gap'],short_pulse:['missed'],loadcell:['conflict','contact'],robot:['contact','conflict']};
const units={gap:'mm',coast:'mm',missed:'회',contact:'',conflict:'',edge:'회전'};
const scenarioNames={delay:'이동 지연',coast:'정지 후 관성',slow:'이동속도 부족',fast:'회전속도 과다',late_return:'복귀 대기각 확대',short_pulse:'짧은 신호 미응답',loadcell:'로드셀·언더롤러 간섭',robot:'로봇 보정 대안'};
function plan(data){
 const id=data.parameters?.scenario;
 if(id==='reference')return [];
 const allowed=focus[id]||Object.keys(titles),end=Number(data.summary?.seconds??Infinity);
 return (data.observation_points||[]).filter(e=>allowed.includes(e.kind)&&titles[e.kind]&&Number.isFinite(e.t)&&e.t>=0&&e.t<=end&&e.value>0).map(e=>({...e,id:`${e.kind}:${e.t}`,scenarioLabel:scenarioNames[id]||'사용자 조건',title:titles[e.kind],detail:e.kind==='conflict'?'빨간 표시는 접촉을 감지한 로드셀, 파란 표시는 보정을 받는 언더롤러 대차입니다. 파란 표시의 AI 방향과 로드셀 보정 방향이 서로 반대입니다. 가상 보정 가정에서의 충돌입니다.':explain[e.kind],representative:['gap','coast'].includes(e.kind)})).sort((a,b)=>a.t-b.t);
}
function crossed(events,seen,from,to,enabled=true){return enabled&&to>=from?events.find(e=>!seen.has(e.id)&&e.t>=from&&e.t<=to)||null:null;}
let events=[],seen=new Set(),seek=null,now=0,selected=null,bar,card,marker,check,count,staleNote,invalidated=false;
function init(){
 if(bar)return;
 const stage=document.querySelector('#sim-stage');
 staleNote=document.createElement('div');staleNote.className='issue-stale';staleNote.hidden=true;staleNote.textContent='조건 변경 · 이 3D는 이전 계산입니다. 새 상황은 ‘조건 적용하고 실험’ 후 표시됩니다.';stage.appendChild(staleNote);
 bar=document.createElement('div');bar.className='issue-controls';
 const label=document.createElement('label');label.className='check';check=document.createElement('input');check.type='checkbox';check.checked=true;label.append(check,document.createTextNode('문제 관찰 시 자동 정지'));
 const next=document.createElement('button');next.type='button';next.textContent='다음 관찰 위치 →';next.onclick=()=>{const e=events.find(e=>e.t>now+.05)||events[0];if(e){seen.add(e.id);seek(e.t);show(e,true);}};
 count=document.createElement('span');bar.append(label,next,count);document.querySelector('.transport').before(bar);
 card=document.createElement('section');card.className='issue-card';card.hidden=true;card.setAttribute('aria-live','polite');
 const head=document.createElement('div');head.className='issue-card-head';const badge=document.createElement('span');badge.className='issue-badge';const close=document.createElement('button');close.type='button';close.textContent='닫기';close.onclick=()=>{selected=null;card.hidden=true;marker.hidden=true;window.WindingView.setIssue(null);};head.append(badge,close);
 const title=document.createElement('strong');title.className='issue-title';const detail=document.createElement('p');const actions=document.createElement('div');
 const zoom=document.createElement('button');zoom.type='button';zoom.textContent='문제 위치 확대';zoom.onclick=()=>window.WindingView.setIssue(selected,true);
 const rewind=document.createElement('button');rewind.type='button';rewind.textContent='3초 전부터 보기';rewind.onclick=()=>{if(!selected)return;const e=selected;seen.delete(e.id);seek(Math.max(0,e.t-3),true);};actions.append(zoom,rewind);const gap=document.createElement('div');gap.className='issue-gap';gap.hidden=true;const strip1=document.createElement('i'),space=document.createElement('b'),strip2=document.createElement('i'),caption=document.createElement('small');caption.textContent='틈 확대 도식 · 3D 크기는 그대로';gap.append(strip1,space,strip2,caption);card.append(head,title,detail,gap,actions);document.querySelector('#sim-root').after(card);
 marker=document.createElement('span');marker.className='issue-location';marker.textContent='빨간 화살표 · 계산된 관찰 위치';marker.hidden=true;stage.appendChild(marker);
}
function show(e,zoom=false){
 if(!card||!e)return;selected=e;card.hidden=false;marker.hidden=false;
 card.querySelector('.issue-badge').textContent=`${e.scenarioLabel} · ${e.representative?'최대값 관찰':'관찰'} · ${e.t.toFixed(2)}초`;
 card.querySelector('.issue-title').textContent=`${e.title}${units[e.kind]?` · ${Number(e.value).toFixed(e.kind==='missed'?0:3)}${units[e.kind]}`:''}`;
 card.querySelector('p').textContent=e.detail;const gap=card.querySelector('.issue-gap');gap.hidden=e.kind!=='gap';gap.querySelector('b').textContent=e.value.toFixed(3)+' mm';window.WindingView.setIssue(e,zoom);
}
function load(data,jump){init();staleNote.hidden=true;events=plan(data);seek=jump;seen=new Set();selected=null;card.hidden=true;marker.hidden=true;now=0;window.WindingView.setIssue(null);bar.querySelector('button').disabled=!events.length;count.textContent=events.length?`이 결과의 관찰 ${events.length}곳 · 부품명으로 위치 확인`:'이 계산에 표시할 문제 관찰점이 없습니다';if(invalidated)invalidate();}
function update(t){now=t;if(selected&&Math.abs(t-selected.t)>1.5){selected=null;card.hidden=true;marker.hidden=true;window.WindingView.setIssue(null);}if(!selected){const e=events.find(e=>Math.abs(t-e.t)<.15);if(e)show(e);}}
function crossing(from,to){const e=crossed(events,seen,from,to,check?.checked);if(e)seen.add(e.id);return e;}
function at(t){const e=events.find(e=>Math.abs(t-e.t)<.2);if(e){seen.add(e.id);show(e,true);}}
function clear(){if(card){card.hidden=true;marker.hidden=true;}selected=null;window.WindingView.setIssue(null);}
function invalidate(){invalidated=true;clear();events=[];if(bar){bar.querySelector('button').disabled=true;count.textContent='조건 변경 · 다시 실험하면 새 관찰 위치가 표시됩니다';staleNote.hidden=false;}}
window.WindingAlerts={plan,crossed,load,update,crossing,show,at,clear,invalidate,acceptResult:()=>{invalidated=false;},reset:()=>{seen.clear();clear();}};
})();
