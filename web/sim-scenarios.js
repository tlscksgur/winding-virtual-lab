(()=>{'use strict';
const $=s=>document.querySelector(s),form=$('#conditions');
let catalog=null,selected=null,reference=null,busy=false,edited=false;
const labels={gap:'한 바퀴 사이 여백',coast:'OFF 후 잔류 이동',contact:'로드셀 접촉',conflict:'보정 방향 충돌',edge:'끝단 추가 감김',missed:'짧은 신호 미응답'};
function announce(){
 if(!selected){$('#scenario-description').textContent='상황을 선택하면 값을 불러옵니다. 실험 버튼으로 계산하세요.';return;}
 $('#scenario-description').textContent=selected.description+(edited?' · 선택 후 값을 추가 수정했습니다.':'')+' 치수와 호기는 유지합니다. 실험 시 스윙 설정은 작업 복사본에 저장됩니다.';
 for(const b of $('#scenario-buttons').querySelectorAll('button'))b.setAttribute('aria-pressed',String(b.dataset.scenario===selected.id));
}
async function choose(item){
 if(busy)return;busy=true;lock(true);$('#run').disabled=true;
 try{
  await window.MotionSettings.stage(item.motion);
  const settings={...catalog.common,...item.plant,scenario:item.id};
  for(const [k,v] of Object.entries(settings)){const el=form.elements.namedItem(k);if(el)el.value=String(v);}
  selected=item;edited=false;announce();
  form.dispatchEvent(new Event('input',{bubbles:true}));window.MotionSettings.estimate();
  $('#run-status').textContent=item.label+'의 값을 불러왔습니다. ‘조건 적용하고 실험’을 누르세요.';
 }catch(e){$('#scenario-description').textContent=e.message;}
 finally{busy=false;lock(false);$('#run').disabled=false;}
}
function lock(on){for(const b of $('#scenario-buttons').querySelectorAll('button'))b.disabled=on;}
function restore(p){
 selected=catalog?.scenarios.find(s=>s.id===p.scenario)||null;edited=false;
 form.elements.namedItem('scenario').value=selected?.id||'custom';announce();
}
function changed(){if(selected){edited=true;form.elements.namedItem('scenario').value='custom';announce();}}
function show(data,jump){
 const box=$('#observations');box.replaceChildren();
 const s=data.summary,points=data.observation_points||[];
 const heading=document.createElement('strong');heading.textContent='계산에서 관찰된 현상';box.appendChild(heading);
 const metrics=[['gap',s.max_turn_gap_mm,'mm'],['coast',s.max_coast_mm,'mm'],['missed',s.missed_pulses,'회'],['conflict',s.loadcell_conflict_seconds,'초'],['contact',s.loadcell_contacts,'회'],['edge',s.extra_edge_turns,'회전']];
 for(const [key,value,unit] of metrics){
  if(value===undefined)continue;
  const point=points.find(p=>p.kind===key),b=document.createElement('button');
  b.type='button';b.textContent=`${labels[key]} ${Number(value).toFixed(key==='missed'||key==='contact'?0:2)}${unit}`;
  b.disabled=!point;b.title=point?'계산된 관찰 시점으로 이동':'이번 계산에서 관찰 지점 없음';
  if(point){b.onclick=()=>jump(point.t);b.dataset.time=point.t;}box.appendChild(b);
 }
 const note=document.createElement('small');note.textContent='수치는 가상 모델의 관찰값입니다. 여백은 동일 층·같은 회전 위상의 케이블 중심 간격에서 외경을 뺀 값이며, 0이어도 다른 문제가 없다는 뜻은 아닙니다.';box.appendChild(note);
 $('#scenario-result').textContent=(catalog?.scenarios.find(x=>x.id===data.parameters.scenario)?.label||'사용자 조건')+' · '+(data.site==='taping7'?'7호기':'4호기');
}
function remember(results){
 if(results.baseline?.parameters.scenario==='reference')reference=results.baseline;
 $('#reference-status').textContent=reference?`기준 조건 보관: ${reference.site==='taping7'?'7':'4'}호기 · ${reference.parameters.seconds}초. 치수·코드·호기가 다르면 직접 비교에 주의하세요.`:'기준 조건을 먼저 실험하면 이후 결과 표에 함께 표시합니다.';
}
async function init(){
 const r=await fetch('/api/scenarios');catalog=await r.json();if(!r.ok)throw Error(catalog.error||'시험 상황을 읽지 못했습니다');
 for(const item of catalog.scenarios){const b=document.createElement('button');b.type='button';b.dataset.scenario=item.id;b.textContent=item.label;b.setAttribute('aria-pressed','false');b.onclick=()=>choose(item);$('#scenario-buttons').appendChild(b);}
 $('#scenario-provenance').textContent=catalog.provenance;announce();
 form.addEventListener('input',e=>{if(e.target!==form)changed();});
 form.addEventListener('change',e=>{if(e.target!==form)changed();});
}
window.WindingScenarios={init,lock,restore,show,remember,changed,getReference:()=>reference};
})();
