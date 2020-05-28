import pytest

from dh_segment_torch.config import ConfigurationError
from dh_segment_torch.config.params import Params
from dh_segment_torch.tests.dhsegment_test_case import DhSegmentTestCase


class TestParams(DhSegmentTestCase):
    def test_write_read_from_file(self):
        config_dict = {"test": "hello"}
        params = Params(config_dict)
        assert params.as_dict() == config_dict

        write_path = self.TEST_DIR / "dummy_config.json"
        params.to_file(str(write_path))

        params2 = Params.from_file(str(write_path))

        assert params.as_dict() == params2.as_dict()

        assert params.pop("test") == "hello"
        assert params.pop("test2", "none") == "none"
        with pytest.raises(ConfigurationError):
            params.pop("test")

    def test_pop_nested_param(self):
        config_dict = {"model": {
            "type": "test",
            "other_param": 1
        }}

        params = Params(config_dict)

        assert isinstance(params.pop("model"), Params)

    def test_assert_empty(self):
        config_dict = {"test": "hello"}
        params = Params(config_dict)
        with pytest.raises(ConfigurationError):
            params.assert_empty("dummy")
        assert params.pop("test") == "hello"
        params.assert_empty("dummy")

    def test_read_jsonnet(self):
        with pytest.raises(RuntimeError):
            Params.from_file(self.FIXTURES_ROOT / "configs" / "resnet50_unet_bad.jsonnet")
        params = Params.from_file(self.FIXTURES_ROOT / "configs" / "resnet50_unet.jsonnet")
        assert len(params) == 3
        assert params['encoder']['type'] == 'resnet50'
        assert params['decoder']['type'] == 'unet'
        assert params['decoder']['decoder_channels'] == [512, 256, 128, 64, 32]

