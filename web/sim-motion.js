(()=>{'use strict';
const $=s=>document.querySelector(s),inputs=[...document.querySelectorAll('[data-motion]')];
let loaded=null,site='taping4',dirty=false,available=false,generation=0;
const note=t=>$('#motion-status').textContent=t;
const values=()=>Object.fromEntries(inputs.map(i=>[i.dataset.motion,Number(i.value)]));
function fill(v){for(const i of inputs)i.value=v[i.dataset.motion];estimate();}
function estimate(){
 const ms=Number(inputs.find(i=>i.dataset.motion==='pulse_ms').value),v=Number(document.querySelector('[name="speed"]').value),d=Number(document.querySelector('[name="cable"]').value),p=values();
 $('#motion-estimate').textContent=available?`1회 신호: 정속 가정 ${(v*ms/1000).toFixed(1)}mm · 목표 간격 좌 ${(p.range_left*d).toFixed(1)} / 우 ${(p.range_right*d).toFixed(1)}mm. 실제 이동은 지연·관성·반복 신호에 따라 달라집니다.`:'';
}
async function request(body){
 const url=body?'/api/motion':'/api/motion?site='+encodeURIComponent(site);
 const r=await fetch(url,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{}),d=await r.json();
 if(!r.ok||d.error)throw Error(d.error||'스윙 설정을 읽지 못했습니다');return d;
}
async function load(nextSite='taping4',force=false){
 const keep=dirty&&!force,staged=keep?values():null,serial=++generation;site=nextSite;
 try{
  const record=await request();if(serial!==generation)return;loaded=record;available=true;inputs.forEach(i=>i.disabled=false);$('#motion-original').disabled=false;
  fill(staged||loaded.working.values);dirty=keep;
  $('#motion-site-label').textContent=site==='taping7'?'7호기 · 코드값':'4호기 · 코드값';
  note(keep?'입력 중인 값을 유지했습니다. 선택한 호기의 작업 복사본에 적용됩니다.':'작업 복사본 config.py 값입니다. 수정 후 ‘조건 적용하고 실험’을 누르세요.');
 }catch(e){if(serial!==generation)return;available=false;dirty=false;inputs.forEach(i=>i.disabled=true);$('#motion-original').disabled=true;note(e.message);}
}
async function save(nextSite){
 if(!available){if(dirty)throw Error('스윙 설정을 코드에서 다시 읽어 주세요.');return;}
 if(nextSite!==site)throw Error('호기 설정이 바뀌었습니다. 스윙 설정을 코드에서 새로 읽어 주세요.');
 if(!dirty)return;
 if(inputs.some(i=>!i.value||!i.checkValidity()))throw Error('스윙·복귀 설정을 표시된 범위 안으로 입력하세요.');
 const d=await request({site,values:values(),expected_hash:loaded.working.hash});loaded.working=d;dirty=false;
 note('작업 복사본 코드에 저장했습니다. 원본 보관본은 그대로 비교합니다.');return d.changed;
}
for(const i of inputs)i.addEventListener('input',()=>{dirty=true;estimate();note('미적용 설정입니다. ‘조건 적용하고 실험’을 누르면 복사본 코드에 저장하고 계산합니다.');});
for(const n of ['speed','cable'])document.querySelector(`[name="${n}"]`).addEventListener('input',estimate);
$('#motion-original').onclick=()=>{if(!loaded)return;fill(loaded.original.values);dirty=true;note('원본값을 불러왔습니다. 실험을 누르면 작업 복사본에 적용합니다.');};
$('#motion-reload').onclick=()=>load(document.querySelector('[name="site"]').value,true);
document.querySelector('[name="site"]').addEventListener('change',e=>load(e.target.value));
async function stage(patch={}){
 await load(document.querySelector('[name="site"]').value,true);
 if(!available)throw Error('설정 코드를 읽지 못해 상황을 불러오지 못했습니다.');
 fill({...loaded.original.values,...patch});dirty=true;
 note('상황의 스윙 설정을 불러왔습니다. 실험 시 작업 복사본에 백업 후 저장합니다.');
}
window.MotionSettings={load,save,estimate,stage};
})();
