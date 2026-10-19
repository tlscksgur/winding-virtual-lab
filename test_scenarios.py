"""Scenario integration checks execute copied code in disposable snapshots."""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from motion_settings import patch, read
from simulation_worker import Plant, Controller, parameters, DEFAULTS
from simulation_scenarios import catalog
from simulation_diagnostics import Diagnostics

ROOT = Path(__file__).resolve().parent


class ScenarioTests(unittest.TestCase):
    def test_handover_operating_values_reach_monitor_before_construction(self):
        source = ROOT/'original/source/SaigeWindingMonitoring'
        for site, strength, interval in [('taping4', 3000, 5), ('taping7', 4000, 6)]:
            control = Controller(source, site, Plant(parameters({'operating': 'handover'})))
            self.assertEqual(control.m.shift_strength, strength)
            self.assertEqual(control.m.shift_config.shift_alarm_interval, interval)
            self.assertEqual(control.m.shift_strength_second, strength//1000)

    def test_on_delay_short_pulse_has_no_motion_but_transport_delay_does(self):
        result = dict(require_drum_shift=True, drum_moving_direction='left_to_right',
                      shift_strength=1100, process_state='first_swing')
        for mode in ['queued', 'on_delay']:
            plant = Plant(parameters(dict(delay=1.5, relay_mode=mode)))
            initial = plant.x
            plant.command(result)
            for _ in range(240): plant.step(1/60)
            if mode == 'on_delay':
                self.assertEqual(plant.x, initial)
                self.assertEqual(plant.missed_pulses, 1)
                self.assertEqual(plant.delivered_pulses, 0)
            else:
                self.assertGreater(plant.x, initial+1)
                self.assertEqual(plant.missed_pulses, 0)
                self.assertEqual(plant.delivered_pulses, 1)

    def test_gap_measurement_uses_one_turn_and_excludes_layer_transition(self):
        plant = Plant(parameters({}))
        diagnostics = Diagnostics(60)
        for n in range(151):
            plant.q=n/100;plant.c=80*plant.q;plant.t=n/10
            diagnostics.deposition(plant)
        self.assertAlmostEqual(diagnostics.gap, 20, places=6)
        diagnostics = Diagnostics(60)
        for n in range(151):
            plant.q=n/100;plant.c=80*plant.q;plant.layer=int(plant.q)
            diagnostics.deposition(plant)
        self.assertEqual(diagnostics.gap, 0)

    def test_scenarios_execute_and_have_measured_consequences(self):
        reports = {}
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as temp:
            base = Path(temp)
            definition = catalog(DEFAULTS)
            for item in definition['scenarios']:
                folder=base/item['id'];folder.mkdir();source=folder/'source'
                shutil.copytree(ROOT/'original/source/SaigeWindingMonitoring/winding', source/'winding',
                                ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
                config_path=source/'winding/engine/config.py'
                record=read(config_path,'taping4')
                if item['motion']:
                    patch(config_path,'taping4',{**record['values'],**item['motion']},record['hash'],folder/'backups')
                plant={**definition['common'],**item['plant'],'scenario':item['id']}
                cfg=dict(source=str(source),output=str(folder),variant='verification',plant=plant,site='taping4')
                file=folder/'config.json';file.write_text(json.dumps(cfg),encoding='utf-8')
                proc=subprocess.run([sys.executable,str(ROOT/'simulation_worker.py'),str(file)],
                                    cwd=folder,capture_output=True,text=True,encoding='utf-8')
                self.assertEqual(proc.returncode,0,proc.stderr)
                data=json.loads((folder/'simulation.json').read_text(encoding='utf-8'))
                self.assertTrue(data['summary']['actual_python_controller'])
                self.assertFalse(data['summary']['ai_inference'])
                reports[item['id']]={k:data[k] for k in
                    ['parameters','summary','observation_points','operating_settings','control_settings']}
        ref=reports['reference']['summary']
        self.assertGreater(reports['coast']['summary']['max_coast_mm'],ref['max_coast_mm'])
        self.assertGreater(reports['short_pulse']['summary']['missed_pulses'],0)
        self.assertGreater(reports['loadcell']['summary']['loadcell_conflict_seconds'],0)
        self.assertEqual(reports['robot']['summary']['loadcell_conflict_seconds'],0)
        self.assertNotEqual(reports['delay']['summary'],ref)
        self.assertNotEqual(reports['slow']['summary']['extra_edge_turns'],ref['extra_edge_turns'])
        self.assertGreater(reports['fast']['summary']['turns'],ref['turns'])
        self.assertEqual(reports['late_return']['control_settings']['hold_right'],180)


if __name__=='__main__': unittest.main(verbosity=2)
