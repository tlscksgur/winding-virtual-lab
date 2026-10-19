"""Reuse the established local render helpers; geometry and motion are new."""
from pathlib import Path
import shutil
base=Path(__file__).resolve().parent
source=(base.parent/'artifacts/winding-3d/reference-model.js').read_text(encoding='utf-8')
prefix=source.split('// Reel with recessed face plates')[0]
prefix=prefix.replace("document.getElementById('winding-reference')","document.getElementById('sim-root')").replace("$('#wp-stage')","$('#sim-stage')")
old_robot=source.split('// Robot base, reduction housings, cast links, service covers and cable dressing.')[1].split('// Incoming cable is updated')[0]
old_robot=old_robot.replace('const robot=groups.robot;','').replace('const tool=groups.tool;','').replace('robot.visible=false;tool.visible=false;','')
old_robot=old_robot.replace('[-.115,.115]', '[-(toolCableDiameter/2+.085),toolCableDiameter/2+.085]')
# Replace the guessed equipment box with the user-identified pressure display.
old_robot=old_robot.replace('rounded(.34,.89,.2,-.38,.52,0,black,forearm,.025);','')
old_robot=old_robot.replace('for(const yy of [.3,.7]){rounded(.02,.24,.13,-.565,yy,.045,steelDull,forearm,.008);box(.015,.12,.078,-.58,yy,.045,red,forearm);}','')
body=(base/'web/sim-scene-body.js').read_text(encoding='utf-8')
robot=(base/'web/sim-robot-body.js').read_text(encoding='utf-8').replace('/* ROBOT_GEOMETRY */',old_robot)
# The scene invokes these hoisted functions only after this script initializes.
body=body.replace('function configure(params){',robot+'\nfunction configure(params){')
(base/'web/sim-model.js').write_text(prefix+body,encoding='utf-8')
shutil.copyfile(base.parent/'artifacts/winding-3d/three.min.js',base/'web/three.min.js')
shutil.copyfile(base.parent/'artifacts/winding-3d/reference-133232000/frame-0300.jpg',base/'web/reference-video.jpg')
