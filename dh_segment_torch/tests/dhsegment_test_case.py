import os
import pathlib
import shutil
import tempfile
from unittest import TestCase

TEMPORARY_DIR = tempfile.mkdtemp(prefix="dhsegment_tests")


class DhSegmentTestCase(TestCase):
    PROJECT_ROOT = (pathlib.Path(__file__).parent / ".." / "..").resolve()
    MODULE_ROOT = PROJECT_ROOT / "dh_segment_torch"
    TESTS_ROOT = MODULE_ROOT / "tests"
    FIXTURES_ROOT = TESTS_ROOT / "fixtures"

    def setUp(self):
        self.TEMPORARY_DIR = pathlib.Path(TEMPORARY_DIR)

        os.makedirs(self.TEMPORARY_DIR, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.TEMPORARY_DIR)
