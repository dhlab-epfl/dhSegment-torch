import pytest

from dh_segment_torch.config import RegistrableError, Registrable
from dh_segment_torch.tests.dhsegment_test_case import DhSegmentTestCase


class Dataset(Registrable):
    default_implementation = "default"

    def __init__(self, x: int = 5):
        self.x = x


Dataset.register("default")(Dataset)


class TestRegistrable(DhSegmentTestCase):
    def test_registrable(self):
        base_class = Dataset

        assert "dummy" not in base_class.get_available()

        @base_class.register("dummy")
        class Dummy(base_class):
            pass

        assert base_class.get("dummy")[0] == Dummy

        assert "dummy" in base_class.get_available()
        default = base_class.default_implementation
        if default is not None:
            assert base_class.get_available()[0] == default
            base_class.default_implementation = "dummy"
            assert base_class.get_available()[0] == "dummy"
            with pytest.raises(RegistrableError):
                base_class.default_implementation = "not here"
                base_class.get_available()
        base_class.default_implementation = default

        with pytest.raises(RegistrableError):

            @base_class.register("dummy")
            class Dummy2(base_class):
                pass

        @base_class.register("dummy", exist_ok=True)
        class Dummy2(base_class):
            pass

        assert base_class.get("dummy")[0] != Dummy
        assert base_class.get("dummy")[0] == Dummy2

        assert base_class.get_type(Dummy2) == "dummy"

        del Registrable._register[base_class]["dummy"]
