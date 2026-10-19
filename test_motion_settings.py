"""Controls must patch real copied configuration without altering the original."""
import tempfile
import unittest
from pathlib import Path
import motion_settings as motion

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'original/source/SaigeWindingMonitoring/winding/engine/config.py'


class MotionTests(unittest.TestCase):
    def test_reads_real_site_defaults(self):
        self.assertEqual(motion.read(SOURCE, 'taping4')['values']['pulse_ms'], 1100)
        self.assertEqual(motion.read(SOURCE, 'taping4')['values']['hold_right'], 50)
        self.assertEqual(motion.read(SOURCE, 'taping7')['values']['hold_right'], 30)

    def test_patch_preserves_unrelated_bytes_and_other_site(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as name:
            path = Path(name)/'config.py';before = SOURCE.read_bytes();path.write_bytes(before)
            read = motion.read(path, 'taping4');values = {**read['values'], 'pulse_ms': 450}
            saved = motion.patch(path, 'taping4', values, read['hash'], Path(name)/'backups')
            self.assertTrue(saved['changed'])
            self.assertEqual(path.read_bytes(), before.replace(b'default_swing_shift_strength: int = 1100', b'default_swing_shift_strength: int = 450'))
            values.update(range_left=1.4, hold_right=70, reset_left=720, start_left=45)
            motion.patch(path, 'taping4', values, saved['hash'], Path(name)/'backups')
            self.assertEqual(motion.read(path, 'taping4')['values'], values)
            self.assertEqual(motion.read(path, 'taping7')['values']['hold_right'], 30)
            self.assertEqual(len(list((Path(name)/'backups').iterdir())), 2)
            restored = motion.patch(path, 'taping4', read['values'], motion.read(path, 'taping4')['hash'], Path(name)/'backups')
            self.assertEqual(restored['values'], read['values'])

    def test_stale_hash_and_invalid_values_do_not_write(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as name:
            path = Path(name)/'config.py';path.write_bytes(SOURCE.read_bytes())
            read = motion.read(path, 'taping4');before = path.read_bytes()
            for values, sha in [(read['values'], 'stale'), ({**read['values'], 'pulse_ms': 1.5}, read['hash']),
                                ({**read['values'], 'range_right': float('nan')}, read['hash'])]:
                with self.assertRaises(ValueError):
                    motion.patch(path, 'taping4', values, sha, Path(name)/'backups')
                self.assertEqual(path.read_bytes(), before)

    def test_no_change_has_no_backup_or_write(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as name:
            path = Path(name)/'config.py';path.write_bytes(SOURCE.read_bytes())
            read = motion.read(path, 'taping4')
            self.assertFalse(motion.patch(path, 'taping4', read['values'], read['hash'], Path(name)/'backups')['changed'])
            self.assertFalse((Path(name)/'backups').exists())


if __name__ == '__main__':
    unittest.main()
