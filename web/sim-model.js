(()=>{
'use strict';
const root=document.getElementById('sim-root'), $=s=>root.querySelector(s), stage=$('#sim-stage'), T=THREE;
const V=(x=0,y=0,z=0)=>new T.Vector3(x,y,z), TAU=Math.PI*2;
let renderer;
try{renderer=new T.WebGLRenderer({antialias:true,alpha:true,powerPreference:'high-performance'});}catch(e){$('#wp-error').hidden=false;return;}
renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,2));renderer.outputEncoding=T.sRGBEncoding;
renderer.toneMapping=T.ACESFilmicToneMapping;renderer.toneMappingExposure=1.13;
renderer.shadowMap.enabled=true;renderer.shadowMap.type=T.PCFSoftShadowMap;
stage.prepend(renderer.domElement);
const scene=new T.Scene(),camera=new T.PerspectiveCamera(34,1,.05,100);
const machine=new T.Group();scene.add(machine);
const groups={};for(const key of ['reel','robot','tool','support','fence','path']){groups[key]=new T.Group();groups[key].userData.part=key;machine.add(groups[key]);}
const materials={};
function material(name,color,roughness=.45,metalness=.1,extra={}){const m=new T.MeshPhysicalMaterial({color,roughness,metalness,envMapIntensity:.65,...extra});m.color.convertSRGBToLinear();return materials[name]=m;}
// Object colors preserve the photographed paint and cable, independently of the surrounding UI theme.
const green=material('green',0x366959,.43,.45),stand=material('stand',0x5a6552,.56,.38),greenEdge=material('greenEdge',0x659083,.38,.6);
const orange=material('orange',0xdf4b0a,.32,.12,{clearcoat:.55,clearcoatRoughness:.23});
const orangeEdge=material('orangeEdge',0xb83b06,.4,.18),black=material('black',0x171c20,.46,.38),rubber=material('rubber',0x11171b,.75,.02);
const cable=material('cable',0x0d121a,.205,.08,{clearcoat:.9,clearcoatRoughness:.16});
const steel=material('steel',0xa7b0b6,.25,.91),steelDull=material('steelDull',0x6b777c,.43,.8),chrome=material('chrome',0xd0d9dc,.145,.98);
const darkSteel=material('darkSteel',0x343e42,.4,.75),yellow=material('yellow',0xd9a926,.45,.25),red=material('red',0xb5141c,.3,.2,{emissive:0x620209,emissiveIntensity:.35});
const flow=material('flow',0x47d8f0,.25,.1,{emissive:0x159ab9,emissiveIntensity:.5});
const geomCache=new Map();
function mesh(geometry,mat,parent=machine){const o=new T.Mesh(geometry,mat);o.castShadow=true;o.receiveShadow=true;parent.add(o);return o;}
function box(w,h,d,x,y,z,mat,parent=machine){const key=`b:${w},${h},${d}`;if(!geomCache.has(key))geomCache.set(key,new T.BoxGeometry(w,h,d));const o=mesh(geomCache.get(key),mat,parent);o.position.set(x,y,z);return o;}
function bevel(w,h,d,r=.035){const key=`r:${w},${h},${d},${r}`;if(geomCache.has(key))return geomCache.get(key);const x=-w/2,y=-h/2,s=new T.Shape();s.moveTo(x+r,y);s.lineTo(x+w-r,y);s.quadraticCurveTo(x+w,y,x+w,y+r);s.lineTo(x+w,y+h-r);s.quadraticCurveTo(x+w,y+h,x+w-r,y+h);s.lineTo(x+r,y+h);s.quadraticCurveTo(x,y+h,x,y+h-r);s.lineTo(x,y+r);s.quadraticCurveTo(x,y,x+r,y);const g=new T.ExtrudeGeometry(s,{depth:d-2*r,bevelEnabled:true,bevelSegments:3,steps:1,bevelSize:r*.45,bevelThickness:r,curveSegments:5});g.translate(0,0,-d/2+r);geomCache.set(key,g);return g;}
function rounded(w,h,d,x,y,z,mat,parent=machine,r=.025){const o=mesh(bevel(w,h,d,r),mat,parent);o.position.set(x,y,z);return o;}
function cylinder(rt,rb,len,x,y,z,mat,parent=machine,axis='x',segments=48){const key=`c:${rt},${rb},${len},${segments}`;if(!geomCache.has(key))geomCache.set(key,new T.CylinderGeometry(rt,rb,len,segments));const o=mesh(geomCache.get(key),mat,parent);o.position.set(x,y,z);if(axis==='x')o.rotation.z=-Math.PI/2;if(axis==='z')o.rotation.x=Math.PI/2;return o;}
const cyl=(r,l,x,y,z,m,p,a='x',n=48)=>cylinder(r,r,l,x,y,z,m,p,a,n);
function rod(a,b,r,m,parent=machine,r2=r){const aa=Array.isArray(a)?V(...a):a,bb=Array.isArray(b)?V(...b):b,o=cylinder(r2,r,aa.distanceTo(bb),0,0,0,m,parent,'y',16);o.position.copy(aa).add(bb).multiplyScalar(.5);o.quaternion.setFromUnitVectors(V(0,1,0),bb.clone().sub(aa).normalize());return o;}
function beam(a,b,w,d,m,parent=machine){const aa=V(...a),bb=V(...b),o=box(w,aa.distanceTo(bb),d,0,0,0,m,parent);o.position.copy(aa).add(bb).multiplyScalar(.5);o.quaternion.setFromUnitVectors(V(0,1,0),bb.sub(aa).normalize());return o;}
function ring(R,r,depth,x,y,z,m,p=machine,axis='x'){
 const b=Math.min(.012,depth*.2,(R-r)*.2),d=depth/2;
 const profile=[[r+b,-d],[R-b,-d],[R,-d+b],[R,d-b],[R-b,d],[r+b,d],[r,d-b],[r,-d+b],[r+b,-d]].map(q=>new T.Vector2(...q));
 const g=new T.LatheGeometry(profile,128);const o=mesh(g,m,p);o.position.set(x,y,z);if(axis==='x')o.rotation.z=-Math.PI/2;if(axis==='z')o.rotation.x=Math.PI/2;return o;
}
function tube(points,r,m,p=machine,segments=72,radial=10){const curve=new T.CatmullRomCurve3(points.map(q=>Array.isArray(q)?V(...q):q));const o=mesh(new T.TubeGeometry(curve,segments,r,radial,false),m,p);return o;}
function bolt(x,y,z,r=.027,p=machine,axis='y') {cyl(r*1.28,.012,x,y,z,steelDull,p,axis,16);const a=axis==='x'?[x+.017,y,z]:axis==='y'?[x,y+.017,z]:[x,y,z+.017];cyl(r,.025,...a,steel,p,axis,6);}
function boltsCircle(radius,count,c,axis,p,size=.023){for(let i=0;i<count;i++){const a=i*TAU/count;const q=axis==='x'?[c[0],c[1]+radius*Math.sin(a),c[2]+radius*Math.cos(a)]:axis==='z'?[c[0]+radius*Math.sin(a),c[1]+radius*Math.cos(a),c[2]]:[c[0]+radius*Math.sin(a),c[1],c[2]+radius*Math.cos(a)];bolt(...q,size,p,axis);}}
function canvasTexture(w,h,draw){const c=document.createElement('canvas');c.width=w;c.height=h;draw(c.getContext('2d'),w,h);const tex=new T.CanvasTexture(c);tex.encoding=T.sRGBEncoding;tex.anisotropy=Math.min(8,renderer.capabilities.getMaxAnisotropy());return tex;}
// Subtle paint grain and brushed-metal variation remain visible when zooming in.
let seed=18799;const random=()=>{seed=(seed*1664525+1013904223)>>>0;return seed/4294967296;};
const grain=canvasTexture(128,128,(c,w,h)=>{const img=c.createImageData(w,h);for(let i=0;i<img.data.length;i+=4){const v=116+Math.round(random()*24);img.data[i]=img.data[i+1]=img.data[i+2]=v;img.data[i+3]=255;}c.putImageData(img,0,0);});grain.encoding=T.LinearEncoding;grain.wrapS=grain.wrapT=T.RepeatWrapping;grain.repeat.set(4,4);
for(const m of [green,stand,orange,orangeEdge]){m.bumpMap=grain;m.bumpScale=.0007;}
const brushed=canvasTexture(128,128,(c,w,h)=>{c.fillStyle='#dadada';c.fillRect(0,0,w,h);for(let y=0;y<h;y++){const v=200+Math.round(random()*45);c.fillStyle=`rgb(${v},${v},${v})`;c.fillRect(0,y,w,1);}});brushed.encoding=T.LinearEncoding;brushed.wrapS=brushed.wrapT=T.RepeatWrapping;brushed.repeat.set(2,4);steel.roughnessMap=brushed;steelDull.roughnessMap=brushed;
function decal(text,width,height,parent,x,y,z,fg='#c4ccd0',bg=null,size=86){const tex=canvasTexture(1024,256,(c,w,h)=>{if(bg){c.fillStyle=bg;c.fillRect(0,0,w,h);}c.fillStyle=fg;c.font=`bold ${size}px Arial`;c.textAlign='center';c.textBaseline='middle';c.fillText(text,w/2,h/2);});const m=new T.MeshStandardMaterial({map:tex,transparent:!bg,roughness:.55,metalness:.15,depthWrite:false});const o=new T.Mesh(new T.PlaneGeometry(width,height),m);o.position.set(x,y,z);parent.add(o);return o;}
// Reflection studio: luminous panels create long highlights on every cable turn.
const envScene=new T.Scene(),room=new T.Mesh(new T.BoxGeometry(28,18,28),new T.MeshBasicMaterial({color:0x090c10,side:T.BackSide}));envScene.add(room);
function softbox(w,h,x,y,z,rx,ry,intensity){const o=new T.Mesh(new T.PlaneGeometry(w,h),new T.MeshBasicMaterial({color:new T.Color(intensity,intensity,intensity),side:T.DoubleSide}));o.position.set(x,y,z);o.rotation.set(rx,ry,0);envScene.add(o);}
softbox(3,14,-8,3,0,0,Math.PI/2,3);softbox(5,12,8,3,-1,0,Math.PI/2,2);softbox(12,2,0,7,4,Math.PI/2,0,4);softbox(1.1,9,-3,2,11,0,0,2);softbox(1.4,10,4,2,11,0,0,1.5);
const pmrem=new T.PMREMGenerator(renderer);const env=pmrem.fromScene(envScene,.035,.1,50);scene.environment=env.texture;pmrem.dispose();
scene.add(new T.HemisphereLight(0xe8f1ff,0x526057,.36));
const key=new T.DirectionalLight(0xfff6e5,1.65);key.position.set(-3,10,7);key.castShadow=true;key.shadow.mapSize.set(2048,2048);Object.assign(key.shadow.camera,{left:-6,right:6,top:8,bottom:-5,near:.5,far:25});key.shadow.bias=-.00015;key.shadow.normalBias=.025;key.shadow.radius=4;scene.add(key);
const fill=new T.DirectionalLight(0xc4dcff,.9);fill.position.set(7,5,-4);scene.add(fill);
const rimLight=new T.DirectionalLight(0xffffff,1.6);rimLight.position.set(-5,7,-6);scene.add(rimLight);
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
 

rounded(1.02,.12,.94,-1.65,.085,2.35,darkSteel,robot,.045);
for(const x of [-2.05,-1.25])for(const z of [2,2.7])bolt(x,.17,z,.046,robot);
cylinder(.32,.43,.47,-1.65,.38,2.35,orange,robot,'y');ring(.36,.28,.06,-1.65,.61,2.35,black,robot,'y');cyl(.32,.14,-1.65,.67,2.35,orange,robot,'y');
const upperBase=rounded(.67,.42,.57,-1.65,.8,2.35,orange,robot,.07);cyl(.24,.71,-1.65,.89,2.35,orange,robot,'z');cyl(.17,.045,-1.65,.89,2.74,orangeEdge,robot,'z');boltsCircle(.18,8,[-1.65,.89,2.768],'z',robot);
const lower=new T.Group(),forearm=new T.Group(),wrist=new T.Group();robot.add(lower,forearm,wrist);
const L1=1.64,L2=2.55,shoulder=V(-1.65,.92,2.35);
function castLink(length,w0,w1,depth,parent){
 const s=new T.Shape();s.moveTo(-w0/2,0);s.quadraticCurveTo(-w0/2-.025,-w0*.38,0,-w0*.4);s.quadraticCurveTo(w0/2+.025,-w0*.38,w0/2,0);s.lineTo(w1/2,length);s.quadraticCurveTo(w1/2,length+w1*.35,0,length+w1*.4);s.quadraticCurveTo(-w1/2,length+w1*.35,-w1/2,length);s.closePath();const g=new T.ExtrudeGeometry(s,{depth:depth-.08,bevelEnabled:true,bevelThickness:.04,bevelSize:.035,bevelSegments:4,curveSegments:10});g.translate(0,0,-depth/2+.04);return mesh(g,orange,parent);
}
castLink(L1,.48,.37,.4,lower);rounded(.21,1.12,.04,0,L1*.48,.225,orangeEdge,lower,.035);rounded(.14,1.02,.025,0,L1*.48,.25,orange,lower,.025);
for(const y of [.24,1.12])for(const x of [-.12,.12])bolt(x,y,.256,.019,lower,'z');
cyl(.28,.59,0,L1,0,orange,lower,'z');cyl(.22,.055,0,L1,.32,orangeEdge,lower,'z');ring(.243,.222,.035,0,L1,.355,orange,lower,'z');boltsCircle(.21,10,[0,L1,.38],'z',lower,.021);
// Tapered main forearm and telescoping-looking distal casting.
cylinder(.135,.25,1.65,0,.78,0,orange,forearm,'y');castLink(.74,.29,.21,.29,forearm).position.y=1.65;
cyl(.255,.08,0,.22,0,orangeEdge,forearm,'y');cyl(.251,.08,0,1.48,0,orangeEdge,forearm,'y');boltsCircle(.225,12,[0,1.53,0],'y',forearm,.017);
const brand=decal('KUKA',.92,.21,forearm,0,.83,.255,'#b8b7b0',null,174);brand.rotation.z=Math.PI/2;
rounded(.48,.64,.14,0,.25,-.3,black,forearm);for(const yy of [.08,.37]){rounded(.27,.17,.035,0,yy,-.392,darkSteel,forearm);}
// Paired rods and equipment box on top of the moving forearm.
for(const zz of [-.18,.18]){rod([-.27,.2,zz],[-.27,1.95,zz],.022,chrome,forearm);rounded(.065,.11,.065,-.27,.3,zz,steelDull,forearm);}


cyl(.19,.37,0,L2,0,orange,forearm,'z');cyl(.15,.035,0,L2,.2,orangeEdge,forearm,'z');boltsCircle(.135,6,[0,L2,.235],'z',forearm,.018);
rounded(.29,.37,.3,0,-.12,0,orange,wrist,.08);cyl(.16,.13,0,-.34,0,orange,wrist,'y');cyl(.164,.04,0,-.415,0,steelDull,wrist,'y');cyl(.125,.045,0,-.445,0,black,wrist,'y');
const dress=new T.Group();robot.add(dress);let dressMeshes=[];
// End effector: long twin linear cylinders, crossplates, rollers, clamps and sensor blocks.

rounded(.68,.09,.53,0,0,0,black,tool);for(const x of [-.28,.28])for(const z of [-.19,.19])bolt(x,.05,z,.021,tool);
for(const x of [-.25,.25]){
 rounded(.085,.76,.1,x,-.46,.19,steelDull,tool);cyl(.055,.63,x,-.49,-.12,steel,tool,'y');cyl(.025,.86,x,-.45,-.12,chrome,tool,'y');
 for(const y of [-.13,-.8]){rounded(.15,.095,.29,x,y,.02,darkSteel,tool,.013);bolt(x,y+.055,.025,.023,tool);}
 rounded(.12,.18,.13,x+(x>0?.09:-.09),-.37,.16,steelDull,tool,.012);cyl(.035,.035,x+(x>0?.17:-.17),-.36,.16,black,tool,'x');
}
for(const y of [-.16,-.77]){box(.51,.06,.18,0,y,.2,black,tool);for(const x of [-(toolCableDiameter/2+.085),toolCableDiameter/2+.085])cyl(.076,.27,x,y,.03,steel,tool,'z');}
box(.1,.5,.06,-.06,-.47,-.22,black,tool);tube([[-.32,-.35,.2],[-.41,-.27,.24],[-.39,-.04,.17],[-.2,.08,.1]],.017,rubber,tool,24,8);


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
