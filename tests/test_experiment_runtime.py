import tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from traser.io import ROOT
from traser.experiments.runtime import experiment_lock


class ExperimentRuntime(unittest.TestCase):
    def setUp(self):
        (ROOT / "outputs").mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=ROOT / "outputs")
        self.root = Path(self.directory.name)
        self.scope = patch("traser.experiments.runtime.ROOT", self.root)
        self.scope.start()

    def tearDown(self):
        self.scope.stop()
        self.directory.cleanup()

    def test_second_complete_job_is_rejected(self):
        with experiment_lock():
            with self.assertRaisesRegex(RuntimeError, "Another complete experiment"):
                with experiment_lock():
                    pass

    def test_exception_releases_owner_without_stale_pid_recovery(self):
        with self.assertRaises(ValueError):
            with experiment_lock():
                raise ValueError("interrupted unit")
        with experiment_lock():
            pass
