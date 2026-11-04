"""CPU-only checks of the user's strict LA GPU cutoff (including DST)."""
import datetime as dt
import unittest
import os
from unittest.mock import patch
from launch_smoke import bounded_deadline


class Limits(unittest.TestCase):
    def test_absolute_permission_never_rolls_forward(self):
        with patch.dict(os.environ,{'RDD_GPU_CUTOFF_UTC':'2026-10-06T15:00:00+00:00'}):
            for stamp in ['2026-10-06T15:00:00+00:00','2026-10-07T07:00:00+00:00']:
                with self.assertRaises(RuntimeError):
                    bounded_deadline(dt.datetime.fromisoformat(stamp),7200)

    def test_absolute_long_run_is_capped(self):
        with patch.dict(os.environ,{'RDD_GPU_CUTOFF_UTC':'2026-10-06T15:00:00+00:00'}):
            now=dt.datetime.fromisoformat('2026-10-06T14:00:00+00:00')
            self.assertEqual(bounded_deadline(now,7200).isoformat(),'2026-10-06T15:00:00+00:00')

    def test_absolute_naive_time_rejected(self):
        with patch.dict(os.environ,{'RDD_GPU_CUTOFF_UTC':'2026-10-06T15:00:00'}):
            with self.assertRaises(ValueError):
                bounded_deadline(dt.datetime.now(dt.timezone.utc),100)

    def test_normal_budget(self):
        now = dt.datetime.fromisoformat("2026-10-05T07:00:00+00:00")
        self.assertEqual((bounded_deadline(now, 290) - now).total_seconds(), 290)

    def test_summer_cutoff_caps(self):
        now = dt.datetime.fromisoformat("2026-10-05T14:59:00+00:00")
        self.assertEqual(bounded_deadline(now, 290).isoformat(), "2026-10-05T15:00:00+00:00")

    def test_winter_cutoff_caps(self):
        now = dt.datetime.fromisoformat("2026-12-05T15:59:00+00:00")
        self.assertEqual(bounded_deadline(now, 290).isoformat(), "2026-12-05T16:00:00+00:00")

    def test_refuses_cutoff_after_and_too_close(self):
        for stamp in ["2026-10-05T14:59:50+00:00", "2026-10-05T15:00:00+00:00",
                      "2026-10-05T18:00:00+00:00"]:
            with self.assertRaises(RuntimeError):
                bounded_deadline(dt.datetime.fromisoformat(stamp), 240)


if __name__ == "__main__":
    unittest.main()
