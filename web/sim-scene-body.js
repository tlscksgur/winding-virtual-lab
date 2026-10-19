// Dimensions in mm. Every reel, guide and cable point uses the same scale.
const U=.00165, reel=groups.reel,support=groups.support;
// Positive X rotation carries the underfed cable upward at the lower tangent.
function underfeedAngle(r){return Math.atan2(p.guide_distance,p.guide_height-p.shaft)+Math.acos(Math.min(.999,r/Math.hypot(p.guide_height-p.shaft,p.guide_distance)));}
const rails=new T.Group(),guide=new T.Group(),dimensionLines=new T.Group();machine.add(rails,guide,dimensionLines);
const blue=material('blueGuide',0x087fc4,.2,.65,{clearcoat:.7});
const topCable=material('activeLayerCable',0xffa11a,.38,.035,{clearcoat:.3,clearcoatRoughness:.3,envMapIntensity:.45});
const underCable=cable,olderCable=cable;
let p=null, data=null, winding=null,tip=null,pathMesh=null,geometry=null,basePositions=null,baseNormals=null,allPoints=[],seedCount=0,lastModified=-1,layerRuns=[],coloredLayer=-1;
let current=null,dirty=true,showDimensions=true,dimTags=[],activeView='whole';
let yaw=.28,pitch=1.43,distance=17.8,target=V(0,2.8,.8),front=false;
const floorMat=new T.MeshStandardMaterial({color:0xe9edee,roughness:1,metalness:0,envMapIntensity:.08});
const floor=new T.Mesh(new T.PlaneGeometry(200,200),floorMat);floor.rotation.x=-Math.PI/2;floor.position.y=-.02;floor.receiveShadow=true;scene.add(floor);scene.background=new T.Color(0xedf0ec);scene.fog=new T.Fog(0xedf0ec,35,70);renderer.toneMappingExposure=1.05;
const dims=document.querySelector('#dimension-labels');
function clear(g){for(const child of [...g.children]){child.traverse(o=>o.geometry?.dispose());g.remove(child);}}
function removeDynamic(){for(const obj of [winding,tip,pathMesh])if(obj){obj.parent?.remove(obj);obj.geometry.dispose();}winding=tip=pathMesh=null;geometry=null;lastModified=-1;coloredLayer=-1;layerRuns=[];}
function dimension(a,b,label,offset=V(),moves=true){const mat=new T.LineBasicMaterial({color:0x6b947a,transparent:true,opacity:.75});const av=V(...a),bv=V(...b);const points=[av,bv];const line=new T.Line(new T.BufferGeometry().setFromPoints(points),mat);line.userData.moves=moves;dimensionLines.add(line);const axis=bv.clone().sub(av).normalize();let tick=V(0,0,1).cross(axis);if(tick.length()<.1)tick=V(1,0,0);tick.normalize().multiplyScalar(.08);for(const v of [av,bv]){const tickLine=new T.Line(new T.BufferGeometry().setFromPoints([v.clone().sub(tick),v.clone().add(tick)]),mat);tickLine.userData.moves=moves;dimensionLines.add(tickLine);}const e=document.createElement('span');e.className='dimension-tag';e.textContent=label;dims.appendChild(e);dimTags.push({e,moves,p:av.clone().add(bv).multiplyScalar(.5).add(offset)});}
function configure(params){
 p=params;data=null;current=null;removeDynamic();for(const g of [reel,support,rails,guide,dimensionLines])clear(g);dims.replaceChildren();dimTags=[];
 const R=p.flange*U/2,W=p.width*U,C=p.core*U/2,H=p.shaft*U,G=p.guide_height*U,Z=p.guide_distance*U,D=p.cable*U;
 reel.position.set(0,H,0);reel.rotation.set(0,0,0);support.position.x=0;
 cyl(C,W+.08,0,0,0,darkSteel,reel,'x',128);cyl(.18,W+1,0,0,0,steelDull,reel);
 for(const side of [-1,1]){
  const x=side*(W/2+.06);
  cyl(R*.972,.08,x,0,0,darkSteel,reel,'x',128);
  ring(R,R*.944,.17,x+side*.06,0,0,steelDull,reel);
  ring(R*.947,R*.895,.13,x+side*.085,0,0,green,reel);
  ring(R*.265,.19,.18,x+side*.105,0,0,green,reel);
  for(let i=0;i<24;i++){const a=i*TAU/24,o=box(.13,R*.69,.055,x+side*.105,R*.59*Math.cos(a),R*.59*Math.sin(a),green,reel);o.rotation.x=a;bolt(x+side*.2,R*.87*Math.cos(a),R*.87*Math.sin(a),.022,reel,'x');}
  cyl(.255,.31,x+side*.16,0,0,steelDull,reel);ring(.31,.25,.08,x+side*.25,0,0,steel,reel);boltsCircle(.4,8,[x+side*.22,0,0],'x',reel,.023);
  ring(C+.01,C-.03,.035,side*(W/2-.035),0,0,steelDull,reel);
  for(let i=0;i<8;i++){const a=i*TAU/8;rod([x-side*.043,C*.6*Math.cos(a),C*.6*Math.sin(a)],[x-side*.043,R*.94*Math.cos(a),R*.94*Math.sin(a)],.003,black,reel);}
 }
 const rollerR=R*.145,rollerZ=R*.64,rollerY=H-Math.sqrt((R+rollerR)**2-rollerZ**2),cabinetH=Math.max(.4,rollerY-.22);
 for(const side of [-1,1]){
  const x=side*(W/2+.15);
  rounded(.67,cabinetH,2*rollerZ+.65,x,cabinetH/2+.16,0,stand,support,.035);
  box(.7,.1,2*rollerZ+.7,x,.14,0,darkSteel,support);
  box(.69,.055,2*rollerZ+.67,x,cabinetH+.19,0,greenEdge,support);
  rounded(.021,cabinetH*.64,.68,x+side*.345,cabinetH*.5,0,greenEdge,support,.008);
  for(const z of [-rollerZ,rollerZ]){
   cyl(rollerR,.43,side*(W/2+.045),rollerY,z,steel,support);cyl(rollerR*1.01,.028,side*(W/2+.24),rollerY,z,chrome,support);
   cyl(.11,.88,x,rollerY,z,steelDull,support);rounded(.28,.42,.49,x+side*.25,rollerY,z,stand,support);
   rounded(.065,.55,.6,x+side*.13,rollerY,z,yellow,support);cyl(.16,.085,x+side*.185,rollerY,z,steelDull,support);
   for(const yy of [rollerY-.19,rollerY+.19])for(const zz of [-.2,.2])bolt(x+side*.22,yy,z+zz,.022,support,'x');
   cyl(.18,.18,x,.2,z,darkSteel,support,'z');cyl(.1,.195,x,.2,z,steelDull,support,'z');
  }
  for(const yy of [.4,cabinetH-.12])for(const zz of [-rollerZ*.8,rollerZ*.8])bolt(x+side*.36,yy,zz,.018,support,'x');
 }
 for(const z of [-rollerZ,rollerZ]){box(W+.4,.24,.25,0,.36,z,stand,support);cyl(.046,W+.5,0,cabinetH*.75,z,steelDull,support);box(W*3+.8,.085,.17,0,.015,z,steelDull,rails);box(W*3+.8,.024,.07,0,.067,z,chrome,rails);for(let x=-W*1.45;x<W*1.46;x+=.6){box(.17,.045,.32,x,-.02,z,darkSteel,rails);bolt(x,.03,z+.12,.022,rails);}}
 const sx=W/2+.68;for(const z of [-rollerZ-.1,0,rollerZ+.1])rod([sx,.16,z],[sx,cabinetH+.9,z],.032,yellow,support);
 for(const y of [cabinetH+.4,cabinetH+.9])rod([sx,y,-rollerZ-.1],[sx,y,rollerZ+.1],.032,yellow,support);
 rounded(.55,.5,.46,W/2+.55,rollerY,-rollerZ,stand,support);cyl(.21,.6,W/2+1.02,rollerY,-rollerZ,darkSteel,support);for(let i=0;i<10;i++)cyl(.23,.017,W/2+.78+i*.045,rollerY,-rollerZ,steelDull,support);
 const wr=Math.max(.14,D*2.2),wy=G-wr*.46,half=Math.max(.25,D*2.7);
 const gp=[[-half,wr],[-half*.8,wr],[-half*.45,wr*.63],[-D*.6,wr*.46],[D*.6,wr*.46],[half*.45,wr*.63],[half*.8,wr],[half,wr]].map(a=>new T.Vector2(a[1],a[0]));
 const wheel=mesh(new T.LatheGeometry(gp,96),blue,guide);wheel.rotation.z=Math.PI/2;wheel.position.set(0,wy,Z);
 cyl(.05,half*2+.27,0,wy,Z,chrome,guide);for(const x of [-half-.09,half+.09]){rounded(.09,Math.max(.16,wy-.05),.17,x,Math.max(.16,wy-.05)/2+.05,Z,steelDull,guide);bolt(x,wy-.07,Z+.1,.028,guide,'z');}box(half*2+.4,.09,.72,0,.05,Z,stand,guide);
 buildVideoRobot();for(const group of [reel,support,rails,guide])mergeStatic(group);
 dimension([-W/2,H+R+.35,0],[W/2,H+R+.35,0],`안쪽 폭 ${p.width.toLocaleString()} mm`);
 dimension([-W/2-.4,H-R,0],[-W/2-.4,H+R,0],`외경 ${p.flange.toLocaleString()} mm`,V(-.2,0,0));
 dimension([W/2+.95,0,0],[W/2+.95,H,0],`축 높이 ${p.shaft.toLocaleString()} mm`,V(.3,0,.1));
 dimension([0,.12,.2],[0,.12,Z],`가이드 거리 ${p.guide_distance.toLocaleString()} mm`,V(.4,0,0),false);
 dimensionLines.visible=showDimensions;viewPreset('whole');dirty=true;
 drawInitial();
}
function drawInitial(){if(!p)return;const limit=(p.width-p.cable)/2,c=p.start==='left'?-(limit-1.8*p.cable):p.start==='full'?-limit:limit-1.8*p.cable;const a=underfeedAngle((p.core+p.cable)/2);update({q:0,x:-c,c,r:(p.core+p.cable)/2,a,v:0,layer:0,direction:p.start==='left'?-1:1});}
const sides=10,ringSize=sides+1;
function fillRing(i,pt,positions,normals){const [q,c,r]=pt,a=underfeedAngle(r),theta=a-q*TAU,rad=p.cable*U/2;const center=V(c*U,r*U*Math.cos(theta),r*U*Math.sin(theta));let tangent;if(i>0&&i<allPoints.length-1){const before=allPoints[i-1],after=allPoints[i+1];tangent=V((after[1]-before[1])*U,(after[2]*Math.cos(underfeedAngle(after[2])-after[0]*TAU)-before[2]*Math.cos(underfeedAngle(before[2])-before[0]*TAU))*U,(after[2]*Math.sin(underfeedAngle(after[2])-after[0]*TAU)-before[2]*Math.sin(underfeedAngle(before[2])-before[0]*TAU))*U).normalize();}else tangent=V(0,Math.sin(theta),-Math.cos(theta));const n=V(0,Math.cos(theta),Math.sin(theta));n.addScaledVector(tangent,-n.dot(tangent)).normalize();const b=V().crossVectors(tangent,n).normalize();for(let j=0;j<ringSize;j++){const t=j/sides*TAU,nx=n.x*Math.cos(t)+b.x*Math.sin(t),ny=n.y*Math.cos(t)+b.y*Math.sin(t),nz=n.z*Math.cos(t)+b.z*Math.sin(t),k=(i*ringSize+j)*3;positions[k]=center.x+rad*nx;positions[k+1]=center.y+rad*ny;positions[k+2]=center.z+rad*nz;normals[k]=nx;normals[k+1]=ny;normals[k+2]=nz;}}
function cableLayer(radius){return Math.max(0,Math.floor((radius-p.core/2-p.cable/2)/p.cable+.5));}
function colorLayers(s){
 const active=Math.max(s.layer||0,cableLayer(s.r));
 if(geometry&&coloredLayer!==active){
  for(let i=0;i<layerRuns.length;i++)geometry.groups[i].materialIndex=layerRuns[i].layer>=active?0:layerRuns[i].layer===active-1?1:2;
  coloredLayer=active;
 }
 const top=document.querySelector('#top-layer-label'),under=document.querySelector('#under-layer-label');
 if(top)top.textContent=`감기는 층 · ${active+1}층`;
 if(under)under.textContent=active>0?`바로 아래 · ${active}층`:'바로 아래 · 아직 없음';
 root.dataset.cableColors=JSON.stringify({currentLayer:active+1,underLayer:active>0?active:null,current:'#ffa11a',under:'#0d121a',older:'#0d121a',scope:'entire active layer; existing black below'});
}
function setData(value){
 configure(value.parameters);data=value;
 const initial=value.initial,first=value.deposition[0];
 const from=-initial.direction*(p.width-p.cable)/2,turns=Math.max(0,initial.direction*(initial.c-from)/p.cable),n=Math.ceil(turns*72);
 allPoints=[];
 for(let i=0;i<n;i++){const f=i/Math.max(1,n);allPoints.push([-turns*(1-f),from+(initial.c-from)*f,first[2],first[3]]);}
 seedCount=n;allPoints.push(...value.deposition);if(allPoints.length<2)allPoints.push([...first]);
 const positions=new Float32Array(allPoints.length*ringSize*3),normals=new Float32Array(positions.length),indices=new Uint32Array((allPoints.length-1)*sides*6);
 for(let i=0;i<allPoints.length;i++)fillRing(i,allPoints[i],positions,normals);
 for(let i=0,k=0;i<allPoints.length-1;i++)for(let j=0;j<sides;j++){const a=i*ringSize+j,b=a+ringSize;indices.set([a,a+1,b,b,a+1,b+1],k);k+=6;}
 basePositions=positions.slice();baseNormals=normals.slice();geometry=new T.BufferGeometry();
 geometry.setAttribute('position',new T.BufferAttribute(positions,3).setUsage(T.DynamicDrawUsage));
 geometry.setAttribute('normal',new T.BufferAttribute(normals,3).setUsage(T.DynamicDrawUsage));geometry.setIndex(new T.BufferAttribute(indices,1));geometry.computeBoundingSphere();
 // Each segment belongs to the layer at its outer endpoint. Only the grouping
 // changes colors; recorded cable coordinates and the simulation remain intact.
 layerRuns=[];
 for(let i=0;i<allPoints.length-1;i++){
  const layer=cableLayer(allPoints[i+1][2]),last=layerRuns[layerRuns.length-1];
  if(last&&last.layer===layer)last.count+=sides*6;
  else layerRuns.push({layer,start:i*sides*6,count:sides*6});
 }
 for(const run of layerRuns)geometry.addGroup(run.start,run.count,2);
 winding=mesh(geometry,[topCable,underCable,olderCable],reel);
 tip=mesh(new T.SphereGeometry(p.cable*U*.53,20,12),topCable,reel);update(value.trace[0]);
}
function previewDimensions(params){
 // A fresh initial layout shows the new size. This is not a rescaled old
 // simulation result; motion is calculated again only by the Python worker.
 const saved={yaw,pitch,distance,target:target.clone(),center:reel.position.clone(),view:activeView};
 const limit=(params.width-params.cable)/2,direction=params.start==='left'?-1:1;
 const c=params.start==='full'?-limit:direction*(limit-1.8*params.cable);
 const r=(params.core+params.cable)/2,dy=params.guide_height-params.shaft;
 const a=underfeedAngle(r);
 const s={t:0,q:0,x:-c,c,r,a,v:0,layer:0,direction,relay:0,state:'dimension_preview'};
 setData({parameters:params,initial:{c,direction},deposition:[[0,c,r,a]],trace:[s]});
 activeView=saved.view;root.dataset.focus=activeView;document.querySelector('#view').value=activeView;
 if(['robot','tool'].includes(activeView))viewPreset(activeView);
 else{yaw=saved.yaw;pitch=saved.pitch;distance=saved.distance;target.copy(saved.target).add(reel.position.clone().sub(saved.center));}
 dimensionLines.visible=showDimensions&&!['robot','tool'].includes(activeView);dirty=true;
}
function update(s){if(!p||!s)return;s={...s,a:underfeedAngle(s.r)};current=s;colorLayers(s);reel.position.set(s.x*U,p.shaft*U,0);reel.rotation.x=s.q*TAU;support.position.x=s.x*U;const tipLocal=V(s.c*U,s.r*U*Math.cos(s.a-s.q*TAU),s.r*U*Math.sin(s.a-s.q*TAU));
 if(geometry){const pos=geometry.attributes.position,norm=geometry.attributes.normal;if(lastModified>=0){const k=lastModified*ringSize*3;pos.array.set(basePositions.subarray(k,k+ringSize*3),k);norm.array.set(baseNormals.subarray(k,k+ringSize*3),k);}let lo=seedCount,hi=allPoints.length-1;while(lo<hi){const mid=Math.ceil((lo+hi)/2);if(allPoints[mid][0]<=s.q)lo=mid;else hi=mid-1;}const tipIndex=Math.min(allPoints.length-1,lo+1);fillRing(tipIndex,[s.q,s.c,s.r,s.a],pos.array,norm.array);const start=Math.min(lastModified<0?tipIndex:lastModified,tipIndex)*ringSize*3,end=(Math.max(lastModified,tipIndex)+1)*ringSize*3;for(const a of [pos,norm]){a.updateRange.offset=start;a.updateRange.count=end-start;a.needsUpdate=true;}lastModified=tipIndex;geometry.setDrawRange(0,tipIndex*sides*6);tip.position.copy(tipLocal);}
 const contact=V((s.x+s.c)*U,(p.shaft+s.r*Math.cos(s.a))*U,s.r*Math.sin(s.a)*U),g=V(0,p.guide_height*U,p.guide_distance*U);const incoming=V(0,p.guide_height*U*.75,p.guide_distance*U+1.6);
 const hand=poseVideoRobot(contact,g,s),lc=handLoadCell(contact,g,hand.x);
 const sensorCable=V(hand.x+Math.max(-lc.gap,Math.min(lc.gap,lc.offset))*U,g.y+(contact.y-g.y)*lc.fraction,hand.z);
 updatePressureIndicator(lc,p.loadcell||'off',s.lc_drive||0);
 if(pathMesh){pathMesh.geometry.dispose();groups.path.remove(pathMesh);}pathMesh=tube([incoming,V(0,p.guide_height*U*.9,p.guide_distance*U+.5),g,g.clone().lerp(contact,.04),sensorCable,g.clone().lerp(contact,.85),contact],p.cable*U/2,topCable,groups.path,96,12);
 root.dataset.robot=JSON.stringify(robotDiagnostics);
 root.dataset.geometry=JSON.stringify({q:s.q,drumX:s.x,cableX:s.c,guideX:0,width:p.width,flange:p.flange,shaft:p.shaft,cable:p.cable,core:p.core,guideHeight:p.guide_height,guideDistance:p.guide_distance,feed:'bottom_to_top',contactHeight:contact.y/U,contactZ:contact.z/U,surfaceVelocityY:-Math.sin(s.a)*s.r*p.rpm*TAU/60,drawCount:geometry?.drawRange.count||0,attachmentGap:contact.distanceTo(tipLocal.clone().applyAxisAngle(V(1,0,0),reel.rotation.x).add(reel.position))});dirty=true;}
function setCamera(){const d=distance*Math.max(1,Math.min(1.5,1.0/camera.aspect));camera.position.set(target.x+d*Math.sin(pitch)*Math.sin(yaw),target.y+d*Math.cos(pitch),target.z+d*Math.sin(pitch)*Math.cos(yaw));camera.lookAt(target);camera.updateMatrixWorld();}
function viewPreset(name){
 if(!p)return;activeView=name;root.dataset.focus=name;dimensionLines.visible=showDimensions&&!['robot','tool'].includes(name);const H=p.shaft*U,R=p.flange*U/2,W=p.width*U,Z=p.guide_distance*U;
 if(name==='robot'){target.set(robotRig.baseX*.46,1.7*robotRig.scale,robotRig.baseZ);yaw=.08;pitch=1.38;distance=8.6*robotRig.scale;}
 else if(name==='tool'){const grip=robotDiagnostics?.grip||[0,H*.45,Z*.8];target.fromArray(grip);target.y+=.28*robotRig.scale;yaw=.55;pitch=1.4;distance=3.8*robotRig.scale;}
 else if(name==='drum'){target.set((current?.x||0)*U,H,0);yaw=.28;pitch=1.38;distance=Math.max(9,R*4.8,W*2.3);}
 else if(name==='video'){target.set(-.35,H*.64,.7);yaw=.12;pitch=1.62;distance=Math.max(15,(H+R)*2.75);}
 else{target.set(-.28,H*.56,.8);yaw=.28;pitch=1.43;distance=Math.max(15,Math.max(H+R,W*1.3,Z)*2.75);}
 const v=document.querySelector('#view');if(v)v.value=name;dirty=true;
}
function draw(){for(const o of dimensionLines.children)o.position.x=o.userData.moves?(current?.x||0)*U:0;setCamera();renderer.render(scene,camera);for(const tag of dimTags){const q=tag.p.clone().add(V(tag.moves?(current?.x||0)*U:0,0,0)).project(camera);tag.e.hidden=!showDimensions||['robot','tool'].includes(activeView)||q.z>1||Math.abs(q.x)>.95||Math.abs(q.y)>.95;tag.e.style.left=((q.x+1)*stage.clientWidth/2)+'px';tag.e.style.top=((1-q.y)*stage.clientHeight/2)+'px';}dirty=false;}
function resize(){if(!stage.clientWidth||!stage.clientHeight)return;renderer.setSize(stage.clientWidth,stage.clientHeight);camera.aspect=stage.clientWidth/stage.clientHeight;camera.updateProjectionMatrix();dirty=true;}new ResizeObserver(resize).observe(stage);
const pointer=new Map();let lastPinch=0;
stage.addEventListener('pointerdown',e=>{if(e.target.tagName!=='CANVAS')return;stage.setPointerCapture(e.pointerId);pointer.set(e.pointerId,{x:e.clientX,y:e.clientY});});
stage.addEventListener('pointermove',e=>{const prev=pointer.get(e.pointerId);if(!prev)return;pointer.set(e.pointerId,{x:e.clientX,y:e.clientY});if(pointer.size===2){const [a,b]=[...pointer.values()],d=Math.hypot(a.x-b.x,a.y-b.y);if(lastPinch)distance=Math.max(3,Math.min(50,distance*lastPinch/d));lastPinch=d;}else if(e.shiftKey||e.buttons===2){const right=V().setFromMatrixColumn(camera.matrix,0),up=V().setFromMatrixColumn(camera.matrix,1);target.addScaledVector(right,-(e.clientX-prev.x)*distance*.0015).addScaledVector(up,(e.clientY-prev.y)*distance*.0015);}else{yaw-=(e.clientX-prev.x)*.006;pitch=Math.max(.2,Math.min(1.7,pitch+(e.clientY-prev.y)*.005));}dirty=true;});
for(const name of ['pointerup','pointercancel'])stage.addEventListener(name,e=>{pointer.delete(e.pointerId);lastPinch=0;});stage.addEventListener('contextmenu',e=>e.preventDefault());stage.addEventListener('wheel',e=>{e.preventDefault();distance=Math.max(3,Math.min(50,distance*Math.exp(e.deltaY*.0008)));dirty=true;},{passive:false});
document.querySelector('#dimensions').onclick=e=>{showDimensions=!showDimensions;dimensionLines.visible=showDimensions;e.target.setAttribute('aria-pressed',String(showDimensions));dirty=true;};document.querySelector('#camera').onclick=e=>{front=!front;viewPreset('whole');yaw=front?0:.28;pitch=front?Math.PI/2:1.43;e.target.textContent=front?'입체':'정면';dirty=true;};
document.querySelector('#view').onchange=e=>{viewPreset(e.target.value);document.querySelector('.view-title').scrollIntoView({block:'start',behavior:'auto'});};
document.querySelector('#fence-toggle').onchange=e=>{groups.fence.visible=e.target.checked;dirty=true;};
function animate(){if(dirty)draw();requestAnimationFrame(animate);}resize();requestAnimationFrame(animate);
window.WindingView={configure,setData,update,previewDimensions};root.dataset.ready='true';
})();
