// Visible assemblies traced from 133232000.mp4. Link proportions are estimated.
// The hand load cells remain around the cable. Normal pose extends right.
// Only the explicitly selected alternative load-cell loop translates the hand.
// No factory robot program is
// available, so this inverse kinematics pose must not be used as a robot command.
let robotRig=null,robotDiagnostics=null;
function buildVideoRobot(){
 const robot=groups.robot,tool=groups.tool;robotRig?.indicatorTexture?.dispose();clear(robot);clear(tool);clear(groups.fence);
 const robotScale=Math.max(.65,Math.min(2.5,Math.max(p.shaft/2600,p.width/1900)));
 const toolCableDiameter=p.cable*U/robotScale;
 const clearance=750*U; // Stand-off is to the left, not in front of the wrist.
 const baseX=-1.45*robotScale-clearance;
 const r=(p.core+p.cable)*U/2,a=underfeedAngle((p.core+p.cable)/2);
 const sensorPoint=V(0,p.guide_height*U,p.guide_distance*U).lerp(V(0,p.shaft*U+r*Math.cos(a),r*Math.sin(a)),.22);
 const baseZ=sensorPoint.z;
 robot.position.set(baseX+1.65*robotScale,0,baseZ-2.35*robotScale);robot.scale.setScalar(robotScale);
 tool.scale.setScalar(robotScale);
 /* ROBOT_GEOMETRY */
 // The video has a near-horizontal forearm. Shorten the prior generic casting
 // to the observed upper-arm / forearm proportion, including its distal joint.
 forearm.scale.y=1.95/L2;
 robotRig={robot,tool,lower,forearm,wrist,dress,dressMeshes,shoulder,L1,L2:1.95,
           scale:robotScale,baseX,baseZ,clearance,sensorPoint,lastPose:null};
 // Paired side load cells in the hand, separate from the pressure display.
 const gap=p.cable*U*.16/robotScale,edge=toolCableDiameter/2+gap;
 const leftCell=material('handLoadCellLeft',0x4d9a87,.3,.45),rightCell=material('handLoadCellRight',0x4d9a87,.3,.45);
 for(const [side,mat] of [[-1,leftCell],[1,rightCell]]){
  rounded(.085,.21,.16,side*(edge+.07),-.48,.015,steelDull,tool,.012);
  cyl(.045,.19,side*(edge+.045),-.48,.015,mat,tool,'y',32);
  box(.065,.09,.025,side*(edge+.09),-.48,.11,mat,tool);
 }
 // User identified the device above the arm as the pressure indicator.
 const indicator=new T.Group();forearm.add(indicator);indicator.position.set(-.39,.75,.16);indicator.rotation.z=Math.PI/2;
 rounded(.75,.27,.16,0,0,0,black,indicator,.018);
 const indicatorTexture=canvasTexture(640,192,()=>{});
 const face=mesh(new T.PlaneGeometry(.67,.205),new T.MeshBasicMaterial({map:indicatorTexture}),indicator);face.position.z=.087;
 Object.assign(robotRig,{leftCell,rightCell,indicatorTexture,indicator,lastDisplay:''});
 // Standalone machine cabinet, screen, pushbuttons and emergency stop.
 const cabinet=new T.Group();robot.add(cabinet);
 const cx=-2.9,cz=3.0;
 rounded(1.02,1.85,.48,cx,.995,cz,steelDull,cabinet,.025);
 const panel=material('cabinetPanel',0xd9d8cd,.68,.12);
 rounded(.98,1.77,.035,cx,1.02,cz+.265,panel,cabinet,.012);
 box(.52,.42,.055,cx,1.3,cz+.3,black,cabinet);
 const screen=canvasTexture(512,320,(c,w,h)=>{c.fillStyle='#112524';c.fillRect(0,0,w,h);c.fillStyle='#8bc7aa';c.font='bold 24px Arial';c.fillText('WINDING CONTROL',24,42);for(let i=0;i<6;i++){c.fillStyle='#668e7a';c.fillRect(25,70+i*33,110,12);c.fillStyle=i===1?'#71d38f':'#2d6658';c.fillRect(165,67+i*33,290,20);}c.fillStyle='#daaf50';c.fillRect(30,278,420,12);});
 const screenPlane=mesh(new T.PlaneGeometry(.435,.302),new T.MeshBasicMaterial({map:screen}),cabinet);screenPlane.position.set(cx,1.3,cz+.332);
 for(const [i,m] of [green,green,black,steelDull].entries())cyl(.034,.025,cx-.3+i*.12,.97,cz+.31,m,cabinet,'z',24);
 cyl(.065,.028,cx+.31,.97,cz+.31,yellow,cabinet,'z');cyl(.046,.05,cx+.31,.97,cz+.34,red,cabinet,'z');
 box(.025,.2,.026,cx-.41,.78,cz+.3,darkSteel,cabinet);decal('권취 자동화 M/C',.66,.1,cabinet,cx,1.61,cz+.298,'#31453b',null,54);
 // Slim warning tower on the left machine stand.
 const tower=groups.fence;cyl(.022,.4,-p.width*U/2-.48,1.94,1.5,steelDull,tower,'y');
 for(const [i,m] of [green,yellow,red].entries())cyl(.07,.11,-p.width*U/2-.48,2.17+i*.115,1.5,m,tower,'y',32);
 // Optional wire-mesh safety panels around the front of the cell.
 const fence=groups.fence;const fz=baseZ+.82,fx1=baseX-1.05*robotScale,fx2=baseX+1.06*robotScale,fy=2.25*robotScale;
 for(const x of [fx1,fx2]){box(.07,fy,.07,x,fy/2,fz,black,fence);box(.11,fy*.91,.095,x+.065,fy*.53,fz,yellow,fence);box(.025,fy*.8,.02,x+.065,fy*.53,fz+.06,black,fence);box(.25,.05,.25,x,.05,fz,steelDull,fence);}
 for(const y of [.12,fy*.5,fy])rod([fx1,y,fz],[fx2,y,fz],.019,black,fence);
 for(let x=fx1+.05;x<fx2;x+=.075*robotScale)rod([x,.12,fz],[x,fy,fz],.0038,black,fence);
 for(let y=.16;y<fy;y+=.075*robotScale)rod([fx1,y,fz],[fx2,y,fz],.0038,black,fence);
 fence.visible=!!document.querySelector('#fence-toggle')?.checked;
 for(const g of [robot,lower,forearm,wrist,tool,cabinet,fence])mergeStatic(g);
}
function poseVideoRobot(contact,guidePoint,snapshot={}){
 if(!robotRig)return null;
 const rr=robotRig,s=rr.scale;
 // The shoulder and wrist share Z: the arm extends along +X (video right).
 // Tool rods remain vertical; they do not point along the incoming cable.
 const grip=rr.sensorPoint.clone().add(V((snapshot.robot_x||0)*U,0,0));
 const u=V(0,1,0),wristWorld=grip.clone().addScaledVector(u,.97*s);
 const local=wristWorld.clone().sub(rr.robot.position).divideScalar(s);
 const delta=local.clone().sub(rr.shoulder),h=Math.hypot(delta.x,delta.z);
 const plane=V(delta.x,0,delta.z).normalize();
 const requested=Math.hypot(h,delta.y),d=Math.min(rr.L1+rr.L2-.0001,Math.max(Math.abs(rr.L2-rr.L1)+.0001,requested));
 const base=Math.atan2(delta.y,h),angle=base+Math.acos(Math.max(-1,Math.min(1,(rr.L1**2+d*d-rr.L2**2)/(2*rr.L1*d))));
 const elbow=rr.shoulder.clone().addScaledVector(plane,rr.L1*Math.cos(angle)).add(V(0,rr.L1*Math.sin(angle),0));
 const reachableWrist=rr.shoulder.clone().addScaledVector(delta,requested>0?d/requested:0);
 rr.lower.position.copy(rr.shoulder);rr.lower.quaternion.setFromUnitVectors(V(0,1,0),elbow.clone().sub(rr.shoulder).normalize());
 rr.forearm.position.copy(elbow);rr.forearm.quaternion.setFromUnitVectors(V(0,1,0),reachableWrist.clone().sub(elbow).normalize());
 rr.wrist.position.copy(reachableWrist);rr.wrist.quaternion.setFromUnitVectors(V(0,1,0),u);
 const actualWrist=reachableWrist.clone().multiplyScalar(s).add(rr.robot.position);
 rr.tool.position.copy(actualWrist).addScaledVector(u,-.49*s);rr.tool.quaternion.setFromUnitVectors(V(0,1,0),u);
 const actualGrip=actualWrist.clone().addScaledVector(u,-.97*s);
 // Corrugated cable dressing follows the articulated arm, with hose clamps.
 if(!rr.hose){rr.hose=tube([[-1.83,.32,2.64],[-1.92,.85,2.63],[elbow.x-.3,elbow.y-.13,elbow.z+.31],[elbow.x-.13,elbow.y+.15,elbow.z+.35],[elbow.x+.4,elbow.y+.13,elbow.z+.24],[reachableWrist.x-.3,reachableWrist.y+.13,reachableWrist.z+.1]],.047,rubber,rr.dress,52,10);}
 if(!rr.lastPose||rr.lastPose.distanceToSquared(local)>.0003){
  if(rr.hose){rr.dress.remove(rr.hose);rr.hose.geometry.dispose();}
  const hosePoints=[[-1.83,.32,2.64],[-1.92,.85,2.63],[elbow.x-.3,elbow.y-.13,elbow.z+.31],[elbow.x-.13,elbow.y+.15,elbow.z+.35],[elbow.x+.4,elbow.y+.13,elbow.z+.24],[reachableWrist.x-.3,reachableWrist.y+.13,reachableWrist.z+.1]];
  rr.hose=tube(hosePoints,.047,rubber,rr.dress,52,10);rr.lastPose=local.clone();
 }
 robotDiagnostics={base:[rr.baseX,0,rr.baseZ],grip:actualGrip.toArray(),desiredGrip:grip.toArray(),
   shoulder:rr.shoulder.clone().multiplyScalar(s).add(rr.robot.position).toArray(),
   elbow:elbow.clone().multiplyScalar(s).add(rr.robot.position).toArray(),wrist:actualWrist.toArray(),
   link1:rr.L1*s,link2:rr.L2*s,reachError:actualGrip.distanceTo(grip),
   jointGap:Math.abs(elbow.distanceTo(reachableWrist)-rr.L2)*s,
   evidence:'133232000.mp4 + 085959000.mp4',mode:Math.abs(snapshot.robot_x||0)>.001?'loadcell_robot_correction':'fixed_sensor_pose',extension:'right (+X)',toolAxis:'vertical',extraStandOffMm:rr.clearance/U,motion:'fixed rightward sensing pose; optional assumed load-cell hand translation'};
 return actualGrip;
}
function updatePressureIndicator(lc,mode,drive){
 const rr=robotRig;if(!rr)return;
 const calibrated=mode!=='off',left=calibrated?Math.round(lc.left*100):0,right=calibrated?Math.round(lc.right*100):0;
 rr.leftCell.color.setHex(left?0xff6633:0x4d9a87);rr.rightCell.color.setHex(right?0xff6633:0x4d9a87);
 const stamp=`${mode}:${left}:${right}`;
 if(rr.lastDisplay!==stamp){
  const canvas=rr.indicatorTexture.image,c=canvas.getContext('2d');c.fillStyle='#111c23';c.fillRect(0,0,640,192);
  c.fillStyle='#c5d9df';c.font='22px Malgun Gothic, Arial';c.fillText('압력 표시기 / 가상 접촉 지수',22,30);
  c.font='bold 65px Consolas';c.fillStyle=left?'#ff9565':'#69dbc8';c.fillText('L '+(calibrated?left:'--'),24,111);
  c.fillStyle=right?'#ff9565':'#69dbc8';c.fillText('R '+(calibrated?right:'--'),336,111);
  c.font='20px Arial';c.fillStyle='#aab7c2';c.fillText('ASSUMED CONTACT / NOT N OR BAR',24,163);
  rr.indicatorTexture.needsUpdate=true;rr.lastDisplay=stamp;
 }
 const readout=document.querySelector('#loadcell-readout');
 const action=drive?(mode==='robot'?`로봇 ${drive>0?'→':'←'}`:`언더롤러 ${drive>0?'←':'→'}`):'보정 대기';
 if(readout)readout.textContent=calibrated?`로드셀 접촉 지수(가정) · 좌 ${left}% / 우 ${right}% · ${action}`:'손: 로드셀 · 팔 위: 압력 표시기 · 압력 보정 미포함 결과';
 root.dataset.loadcell=JSON.stringify({...lc,mode,drive,indexUnits:'dimensionless assumed contact, not calibrated pressure'});
}
function handLoadCell(contact,guidePoint,handX){
 const z=robotRig.sensorPoint.z,f=(z-guidePoint.z)/(contact.z-guidePoint.z);
 const cableX=guidePoint.x+(contact.x-guidePoint.x)*f,offset=(cableX-handX)/U,gap=p.cable*.16;
 const excess=Math.max(0,Math.abs(offset)-gap),signal=excess>0?(offset>0?1:-1):0,index=Math.min(1,excess/(p.cable*.5));
 return {offset,gap,signal,left:signal<0?index:0,right:signal>0?index:0,cableX,fraction:f};
}
function mergeStatic(group){
 const batches=new Map();for(const o of [...group.children]){
  if(!o.isMesh||!o.geometry.attributes.normal||!o.geometry.attributes.uv||Array.isArray(o.material)||o.material.map)continue;
  o.updateMatrix();const key=o.material.uuid;if(!batches.has(key))batches.set(key,{material:o.material,items:[]});batches.get(key).items.push(o);
 }
 for(const {material,items} of batches.values()){
  if(items.length<4)continue;let size=0;const list=items.map(o=>{const g=o.geometry.index?o.geometry.toNonIndexed():o.geometry.clone();g.applyMatrix4(o.matrix);size+=g.attributes.position.count;return g;});
  const pos=new Float32Array(size*3),norm=new Float32Array(size*3),uv=new Float32Array(size*2);let offset=0;
  for(const g of list){pos.set(g.attributes.position.array,offset*3);norm.set(g.attributes.normal.array,offset*3);uv.set(g.attributes.uv.array,offset*2);offset+=g.attributes.position.count;g.dispose();}
  const g=new T.BufferGeometry();g.setAttribute('position',new T.BufferAttribute(pos,3));g.setAttribute('normal',new T.BufferAttribute(norm,3));g.setAttribute('uv',new T.BufferAttribute(uv,2));mesh(g,material,group);for(const o of items)group.remove(o);
 }
}
