import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "differentiate-seismic-solver" / "scripts"
sys.path.insert(0, str(SCRIPTS))
from model_recovery_report import validate, volume_payload


class ModelRecoveryScaleTest(unittest.TestCase):
    def test_full_volume_sets_the_shared_scale_before_slicing(self):
        axis = np.arange(3.0)
        model = np.full((3, 3, 3), 3000.0)
        data = dict(x=axis, y=axis, depth=axis, true=model.copy(),
                    initial=model.copy(), inverted=model.copy())
        data["true"][0, 0, 0] = 1400.0
        data["inverted"][2, 2, 2] = 4900.0
        data["initial"][0, 2, 0] = 9000.0
        metadata = dict(coordinate_unit="m", value_unit="m/s", accepted_updates=1,
                        model_is_last_accepted=True, status="running", limits=[2500, 3500])

        for masked, expected_max in [(False, 9000.0), (True, 4900.0)]:
            with self.subTest(masked=masked):
                if masked:
                    data["display_mask"] = np.ones(model.shape, dtype=bool)
                    data["display_mask"][0, 2, 0] = False
                arrays, meta, sections = validate(data, metadata)
                self.assertEqual(meta["limits"], [1400.0, expected_max])
                self.assertTrue(all(section["index"] == 1 for section in sections.values()))
                payload = volume_payload(arrays, meta)
                self.assertEqual(payload["limits"], [1400.0, expected_max])
                self.assertEqual(set(payload["fields"]), {"true", "initial", "inverted"})
                masked_cell = payload["fields"]["initial"][6]
                if masked:
                    self.assertIsNone(masked_cell)
                else:
                    self.assertEqual(masked_cell, 9000.0)
