"""Focused checks for source preservation, the offline boundary and real state-code replay."""
import json,shutil,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import lab

class LabTests(unittest.TestCase):
    def test_original_snapshot_matches_download(self):
        result=lab.verify_original();self.assertTrue(result['ok']);self.assertEqual(result['files'],len(lab.MANIFEST['files']))
    def test_network_input_is_rejected(self):
        for value in ['rtsp://camera/live','http://192.168.220.130/source','\\\\server\\share\\a.mp4']:
            with self.assertRaises(ValueError):lab.local_input(value)
    def test_worker_cannot_connect_or_write_original(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as name:
            code="from safety import install_offline_guard; import socket; install_offline_guard("+repr(name)+"); socket.socket().connect(('192.168.220.130',80))"
            result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertIn('connect blocked',result.stderr)
            code="from safety import install_offline_guard; install_offline_guard("+repr(name)+"); open("+repr(str(ROOT/'original'/'blocked-write.txt'))+",'w').write('x')"
            result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertIn('writes outside',result.stderr)
            self.assertFalse((ROOT/'original'/'blocked-write.txt').exists())
    def test_original_state_classes_and_changed_threshold_compare(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as name:
            temp=Path(name);outputs=[]
            for variant in ['baseline','modified']:
                folder=temp/variant;folder.mkdir();source=folder/'source'
                shutil.copytree(lab.WORKING/'winding',source/'winding',ignore=shutil.ignore_patterns('__pycache__','*.pyd'))
                if variant=='modified':
                    file=source/'winding'/'engine'/'config.py';text=file.read_text(encoding='utf-8');file.write_text(text.replace('"tightening_delay_angle": 60.0','"tightening_delay_angle": 160.0'),encoding='utf-8')
                config={'source':str(source),'output':str(folder),'variant':variant,'site':'taping4'}
                (folder/'config.json').write_text(json.dumps(config),encoding='utf-8')
                result=subprocess.run([sys.executable,str(ROOT/'scenario_worker.py'),str(folder/'config.json')],cwd=folder,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                outputs.append(json.loads((folder/'summary.json').read_text(encoding='utf-8')))
            self.assertIn('second_swing_end',outputs[0]['states']);self.assertFalse(outputs[0]['ai_inference'])
            self.assertNotEqual(outputs[0]['states'],outputs[1]['states'])
if __name__=='__main__':unittest.main(verbosity=2)
