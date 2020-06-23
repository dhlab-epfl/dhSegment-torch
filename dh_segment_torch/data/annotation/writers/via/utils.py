from typing import Callable, Any, Union

import torch
from torch.utils.data import Dataset

from dh_segment_torch.data.annotation.annotation import Annotation
from dh_segment_torch.data.annotation.annotation_iterator import AnnotationIterator


class AnnotationProcessorDataset(Dataset):
    def __init__(
        self,
        annotation_iterator: AnnotationIterator,
        process_annotation: Callable[[Annotation], Any],
    ):
        self.annotation_iterator = annotation_iterator
        self.process_annotation = process_annotation

    def __getitem__(self, idx: Union[int, torch.Tensor]):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        annotation = self.annotation_iterator[idx]
        return self.process_annotation(annotation)

    def __len__(self):
        return len(self.annotation_iterator)


DEFAULT_VIA2_DICT = {
    "_via_settings": {
        "ui": {
            "annotation_editor_height": 25,
            "annotation_editor_fontsize": 0.8,
            "leftsidebar_width": 18,
            "image_grid": {
                "img_height": 80,
                "rshape_fill": "none",
                "rshape_fill_opacity": 0.3,
                "rshape_stroke": "yellow",
                "rshape_stroke_width": 2,
                "show_region_shape": True,
                "show_image_policy": "all",
            },
            "image": {
                "region_label": "__via_region_id__",
                "region_color": "__via_default_region_color__",
                "region_label_font": "10px Sans",
                "on_image_annotation_editor_placement": "NEAR_REGION",
            },
        },
        "core": {"buffer_size": 18, "filepath": {}, "default_filepath": ""},
        "project": {"name": "via_project_23Jun2020_16h58m"},
    },
    "_via_data_format_version": "2.0.10",
}

DEFAULT_VIA3_DICT = {
    "project": {
        "pid": "__VIA_PROJECT_ID__",
        "rev": "__VIA_PROJECT_REV_ID__",
        "rev_timestamp": "__VIA_PROJECT_REV_TIMESTAMP__",
        "pname": "Unnamed VIA Project",
        "creator": "VGG Image Annotator (http://www.robots.ox.ac.uk/~vgg/software/via)",
        "created": 1592928407652,
        "vid_list": [],
    },
    "config": {
        "file": {"loc_prefix": {"1": "", "2": "", "3": "", "4": ""}},
        "ui": {
            "file_content_align": "center",
            "file_metadata_editor_visible": True,
            "spatial_metadata_editor_visible": True,
            "spatial_region_label_attribute_id": "",
            "gtimeline_container_height": "45",
        },
    },
}
