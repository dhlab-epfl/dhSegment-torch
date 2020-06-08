import numpy as np
import pytest

from dh_segment_torch.config.params import Params
from dh_segment_torch.data.transform import Transform
from dh_segment_torch.tests.dhsegment_test_case import DhSegmentTestCase


class SegmentationModelTest(DhSegmentTestCase):
    def test_blur(self):
        blur = Transform.from_params(Params({"type": "blur", "p": 1}))
        img = np.random.random((128, 128, 3))
        res = blur(image=img)
        assert img.mean() != res["image"].mean()
        blur = Transform.from_params(Params({"type": "blur", "p": 0}))
        img = np.random.random((128, 128, 3))
        res = blur(image=img)
        assert img.mean() == res["image"].mean()

    def test_compose(self):
        param = {
            "type": "compose",
            "transforms": [
                "blur",
                {"type": "horizontal_flip", "always_apply": True},
                {"type": "random_crop", "height": 32, "width": 32},
            ],
            "additional_targets": {"image2": "image", "mask1": "mask", "label": "mask"},
        }

        compose = Transform.from_params(Params(param))
        img = np.random.random((128, 128, 3))
        mask = np.random.randint(0, 20, (128, 128))
        res = compose(image2=img, image=img, mask1=mask, label=mask)

        assert np.allclose(res["image"], res["image2"])
        assert np.allclose(res["mask1"], res["label"])
        assert res["image"].shape[:2] == (32, 32)
        assert res["mask1"].shape[:2] == (32, 32)

    def test_patches(self):
        img = np.random.random((64, 64, 3))
        mask = np.random.randint(0, 20, (63, 63))

        patch_shape = 32
        param = {"type": "sample_to_patches", "patch_shape": patch_shape}
        sample_to_patches = Transform.from_params(Params(param))
        res = sample_to_patches(image=img, mask=mask)
        assert all([p.shape[:2] == (patch_shape, patch_shape) for p in res["image"]])
        assert all([p.shape[:2] == (patch_shape, patch_shape) for p in res["mask"]])

        patch_shape = (32, 2)
        param = {"type": "sample_to_patches", "patch_shape": patch_shape}
        sample_to_patches = Transform.from_params(Params(param))
        res = sample_to_patches(image=img, mask=mask)
        assert all([p.shape[:2] == patch_shape for p in res["image"]])
        assert all([p.shape[:2] == patch_shape for p in res["mask"]])

        with pytest.raises(ValueError):
            sample_to_patches(image=np.ones((2, 32, 32, 3)))

        with pytest.raises(IndexError):
            sample_to_patches(image=np.ones((2,)))

        with pytest.raises(ValueError):
            patch_shape = (1, 1)
            param = {"type": "sample_to_patches", "patch_shape": patch_shape}
            sample_to_patches = Transform.from_params(Params(param))
            res = sample_to_patches(image=img, mask=mask)
