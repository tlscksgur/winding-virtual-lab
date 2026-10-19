"""Causal regression checks for the assumed plant and the real copied controller."""
import json, math, shutil, subprocess, sys, tempfile, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from simulation_worker import parameters, Plant

class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(dir=ROOT/'outputs');cls.base=Path(cls.temp.name)
        cls.normal=cls.execute('normal',{})
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    @classmethod
    def execute(cls,name,plant,modify=False,site='taping4'):
        folder=cls.base/name;folder.mkdir();source=folder/'source'
        # Use a fixed controller fixture so user-edited working settings do not change test expectations.
        # Only this disposable snapshot is modified in the test.
        shutil.copytree(ROOT/'original/source/SaigeWindingMonitoring/winding',source/'winding',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        if modify:
            file=source/'winding/engine/config.py';s=file.read_text(encoding='utf-8')
            assert 'default_swing_shift_strength: int = 1100' in s
            file.write_text(s.replace('default_swing_shift_strength: int = 1100','default_swing_shift_strength: int = 450'),encoding='utf-8')
        cfg={'source':str(source),'output':str(folder),'variant':name,'plant':{'start':'right','seconds':180,**plant},'site':site}
        file=folder/'config.json';file.write_text(json.dumps(cfg),encoding='utf-8')
        result=subprocess.run([sys.executable,str(ROOT/'simulation_worker.py'),str(file)],cwd=folder,capture_output=True,text=True,encoding='utf-8')
        if result.returncode:raise AssertionError(result.stderr)
        return json.loads((folder/'simulation.json').read_text(encoding='utf-8'))
    def test_default_starts_empty_and_runs_complete_winding(self):
        p=parameters({});self.assertEqual(p['start'],'full');self.assertEqual(p['seconds'],2400)
        plant=Plant(p);self.assertEqual(plant.q,0);self.assertEqual(plant.c,-plant.limit)

    def test_real_controller_receives_feedback_and_completes_return(self):
        states=[e['state'] for e in self.normal['events']]
        self.assertIn('first_swing',states);self.assertIn('second_swing_end',states)
        self.assertEqual(states[-1],'winding')
        self.assertTrue(self.normal['summary']['actual_python_controller'])
        self.assertFalse(self.normal['summary']['ai_inference'])
    def test_same_inputs_are_deterministic(self):
        repeat=self.execute('repeat',{})
        self.assertEqual(self.normal['trace'],repeat['trace'])
        self.assertEqual(self.normal['commands'],repeat['commands'])
    def test_python_code_change_alters_actuation_and_plant(self):
        modified=self.execute('modified',{},True)
        self.assertNotEqual(self.normal['commands'],modified['commands'])
        self.assertNotEqual(self.normal['trace'][-1]['x'],modified['trace'][-1]['x'])
        self.assertIn(.45,[c['duration'] for c in modified['commands']])
    def test_rotation_and_dimensions_drive_calculated_motion(self):
        slower=self.execute('slow',{'rpm':3,'width':1600})
        self.assertAlmostEqual(slower['summary']['turns'],self.normal['summary']['turns']/2,places=2)
        self.assertEqual(slower['summary']['ideal_traverse_speed'],3)
        self.assertNotEqual(slower['events'],self.normal['events'])
        self.assertAlmostEqual(slower['summary']['turns_per_layer'],(1600-60)/60,places=2)
    def test_latency_is_applied_to_command_not_animation(self):
        lag=self.execute('lag',{'delay':3})
        command=lag['commands'][0];trace=lag['trace']
        before=[s for s in trace if command['t']<=s['t']<command['t']+3-.1]
        self.assertTrue(before);self.assertTrue(all(abs(s['v'])<1e-6 for s in before))
        after=[s for s in trace if command['t']+3+.3<s['t']<command['t']+4]
        self.assertTrue(any(abs(s['v'])>1 for s in after))
    def test_long_run_covers_three_complete_round_trips(self):
        full=self.execute('long',{'seconds':2400,'start':'full'})
        self.assertEqual(full['trace'][0]['q'],0)
        self.assertEqual(full['initial']['c'],-(1900-60)/2)
        self.assertEqual(full['initial']['direction'],1)
        self.assertGreaterEqual(sum(e['state']=='first_swing' for e in full['events']),2)
        self.assertGreaterEqual(full['summary']['full_traverses'],6)
        self.assertGreaterEqual(full['summary']['round_trips'],3)
        self.assertIsNone(full['summary']['fault'])
        # Continuous deposited path never jumps radially by more than a diameter.
        points=full['deposition']
        self.assertTrue(all(abs(b[2]-a[2])<61 for a,b in zip(points,points[1:])))
    def test_underfeed_is_lower_tangent_with_upward_surface_motion(self):
        for dims in ({},{'core':1500,'guide_height':1000,'guide_distance':2500}):
            plant=Plant(parameters(dims));p=plant.p;r=plant.radius;a=plant.contact_angle
            # Radius and incoming span are perpendicular at a true tangent.
            y,z=r*math.cos(a),r*math.sin(a)
            incoming=(y-(p['guide_height']-p['shaft']),z-p['guide_distance'])
            self.assertLess(y,0)
            self.assertAlmostEqual(y*incoming[0]+z*incoming[1],0,places=6)
            # Positive X rotation carries cable upward and into the same tangent.
            tangent=(-math.sin(a),math.cos(a))
            self.assertGreater(tangent[0],0)
            self.assertGreater(tangent[0]*incoming[0]+tangent[1]*incoming[1],0)
            self.assertAlmostEqual(tangent[0]*incoming[1]-tangent[1]*incoming[0],0,places=6)
            # A deposited point remains attached while the drum rotates.
            q=.37;local=a-q*2*math.pi;rotated=local+q*2*math.pi
            self.assertAlmostEqual(math.cos(rotated),math.cos(a),places=10)
            self.assertAlmostEqual(math.sin(rotated),math.sin(a),places=10)

    def test_loadcell_underroller_feedback_is_delayed_and_moves_only_drum(self):
        plant=Plant(parameters({'loadcell':'underroller'}));plant.x+=250;x0=plant.x
        self.assertEqual(plant.loadcell()['signal'],1)
        for _ in range(36):plant.step(1/60)
        self.assertEqual(plant.x,x0)
        for _ in range(84):plant.step(1/60)
        self.assertLess(plant.x,x0);self.assertEqual(plant.robot_x,0)
        self.assertGreater(plant.lc_hits,0)

    def test_loadcell_robot_alternative_moves_hand_without_pressure_drum_drive(self):
        plant=Plant(parameters({'loadcell':'robot'}));plant.x+=250;x0=plant.x
        for _ in range(120):plant.step(1/60)
        self.assertGreater(plant.robot_x,0);self.assertEqual(plant.x,x0)
        centered=Plant(parameters({'loadcell':'robot'}))
        centered.step(1/60);self.assertEqual(centered.robot_x,0)

    def test_pressure_priority_conflict_is_explicit_assumption(self):
        plant=Plant(parameters({'loadcell':'underroller'}));plant.x+=250
        plant.queue.append((0,1,10))
        for _ in range(120):plant.step(1/60)
        self.assertGreater(plant.lc_conflict_seconds,0)
        self.assertEqual(plant.relay,1)  # The original AI request remains visible.
        self.assertLess(plant.v,0)      # Separate assumed pressure loop overrides it.

    def test_loadcell_modes_execute_copied_controller_and_record_hand_motion(self):
        for mode in ('underroller','robot'):
            result=self.execute('loop_'+mode,{'loadcell':mode})
            self.assertTrue(result['summary']['actual_python_controller'])
            self.assertEqual(result['summary']['loadcell_mode'],mode)
            self.assertGreater(result['summary']['loadcell_contacts'],0)
            motion=max(abs(s['robot_x']) for s in result['trace'])
            if mode=='robot':self.assertGreater(motion,1)
            else:self.assertEqual(motion,0)

    def test_impossible_geometry_is_rejected(self):
        for values in [{'core':2500},{'shaft':1000},{'width':200,'cable':80},{'rpm':float('nan')}]:
            with self.assertRaises(ValueError):parameters(values)
    def test_left_endpoint_and_site7_execute_real_controller(self):
        left=self.execute('left',{'start':'left','seconds':240})
        site7=self.execute('site7',{},site='taping7')
        self.assertIn('tightening',[e['state'] for e in left['events']])
        self.assertEqual(site7['site'],'taping7')
        self.assertNotEqual(self.normal['commands'],site7['commands'])

if __name__=='__main__':unittest.main(verbosity=2)
