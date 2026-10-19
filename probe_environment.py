"""Check the provided runtime without touching the remote system or loading a model."""
import faulthandler,json,sys,traceback
from pathlib import Path
from safety import install_offline_guard
ROOT=Path(__file__).resolve().parent
report={'ai_ready':False,'python':sys.executable}
try:
    faulthandler.dump_traceback_later(30,repeat=False)
    install_offline_guard()
    sys.path.insert(0,str(ROOT/'working'/'SaigeWindingMonitoring'))
    print('Loading torch...',flush=True)
    import torch,cv2,numpy
    print('Loading Saige segmentation library...',flush=True)
    from importlib.metadata import version
    from winding.modules.segmentation import SegmentationEngine
    report.update(ai_ready=True,reason='원본 추론 라이브러리 import 확인',torch=torch.__version__,opencv=cv2.__version__,toolkit=version('SaigeToolkit'))
except Exception as e:
    report['reason']=type(e).__name__+': '+str(e);traceback.print_exc()
faulthandler.cancel_dump_traceback_later()
(ROOT/'environment-status.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
