import logging
import os
from typing import Union, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dh_segment_torch.config.lazy import Lazy
from dh_segment_torch.data.annotation.annotation_iterator import AnnotationIterator
from dh_segment_torch.data.annotation.annotation_painter import AnnotationPainter
from dh_segment_torch.data.annotation.annotation_reader import AnnotationReader
from dh_segment_torch.data.annotation.annotation_writer import AnnotationWriter
from dh_segment_torch.data.annotation.image_size import ImageSize
from dh_segment_torch.data.annotation.utils import (
    write_image,
    extract_image_ext,
    iiif_url_to_resized,
)
from dh_segment_torch.data.color_labels import ColorLabels
from dh_segment_torch.data.transforms.albumentation import Compose
from dh_segment_torch.data.transforms.fixed_resize import FixedResize

logger = logging.getLogger(__name__)


class ImageWriterDataset(Dataset):
    def __init__(
        self,
        annotation_iterator: AnnotationIterator,
        annotation_painter: AnnotationPainter,
        labels_dir: str,
        images_dir: Optional[str] = None,
        copy_images: bool = False,
        overwrite: bool = True,
        resizer: Optional[FixedResize] = None,
        compose: Optional[Compose] = None,
    ):
        self.labels_dir = labels_dir
        if images_dir:
            self.images_dir = images_dir
        else:
            self.images_dir = labels_dir

        if os.path.exists(self.labels_dir) and not os.path.isdir(self.labels_dir):
            raise ValueError("The path to the label directory is not a directory")
        else:
            os.makedirs(self.labels_dir, exist_ok=True)
        if os.path.exists(self.images_dir) and not os.path.isdir(self.images_dir):
            raise ValueError("The path to the label directory is not a directory")
        else:
            os.makedirs(self.images_dir, exist_ok=True)

        self.copy_images = copy_images
        self.overwrite = overwrite

        self.annotation_iterator = annotation_iterator
        self.annotation_painter = annotation_painter
        self.resizer = resizer
        self.compose = compose

        if (self.resizer or self.compose) and not self.copy_images:
            logger.warning(
                "Transformations are used, however copy_images was set to false. "
                f"Since transformations will modify the image, we copy it to {self.images_dir}."
            )
            self.copy_images = True

    def __getitem__(self, idx: Union[int, torch.Tensor]):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        annotation = self.annotation_iterator[idx]

        if self.resizer:
            if annotation.is_iiif:
                annotation.uri = iiif_url_to_resized(
                    annotation.uri, self.resizer.height, self.resizer.width
                )
            image = annotation.image
            image = self.resizer(image=image)["image"]
            image_size = ImageSize.from_image_array(image)
        else:
            image = annotation.image
            image_size = annotation.image_size
        label_image = self.annotation_painter.paint(
            image_size, annotation.labels_annotations, track_colors=True
        )

        if self.compose:
            result = self.compose(image=image, mask=label_image)
            image = result["image"]
            label_image = result["mask"]

        basename = annotation.image_id
        image_extension = extract_image_ext(annotation.uri)

        image_path = os.path.join(self.images_dir, basename + image_extension)
        if not self.copy_images:
            if annotation.uri.startswith("https"):
                logger.warning(
                    "copy_images is set to false, however some images are URLs, check if that is wanted."
                )
            image_path = annotation.uri

        if self.labels_dir == self.images_dir and self.copy_images:
            suffix = "_label"
        else:
            suffix = ""

        label_path = os.path.join(self.labels_dir, basename + suffix + ".png")
        # TODO try to move actual writing outside
        if self.copy_images:
            write_image(image_path, image, self.overwrite)

        write_image(label_path, label_image, self.overwrite)

        return image_path, label_path, self.annotation_painter.used_colors

    def __len__(self):
        return len(self.annotation_iterator)


@AnnotationWriter.register("image", "from_reader")
class ImageWriter(AnnotationWriter):
    def __init__(
        self,
        annotation_iterator: AnnotationIterator,
        annotation_painter: AnnotationPainter,
        color_labels: ColorLabels,
        labels_dir: str,
        color_labels_file_path: str,
        csv_path: str,
        images_dir: Optional[str] = None,
        only_basenames: bool = False,
        copy_images: bool = False,
        overwrite: bool = True,
        resizer: Optional[FixedResize] = None,
        compose: Optional[Compose] = None,
        filter_color_labels: bool = True,
        progress: bool = False,
    ):
        super().__init__(annotation_iterator, overwrite, progress)
        self.annotation_dataset_writer = ImageWriterDataset(
            self.annotation_iterator,
            annotation_painter,
            labels_dir,
            images_dir,
            copy_images,
            self.overwrite,
            resizer,
            compose,
        )
        self.color_labels = color_labels
        self.color_labels_file_path = color_labels_file_path
        self.csv_path = csv_path
        self.filter_color_labels = filter_color_labels
        self.only_basename = only_basenames
        self.progress = progress

    def write(self, num_workers: int = 0):
        annotation_writer_dataloder = DataLoader(
            dataset=self.annotation_dataset_writer,
            batch_size=1,
            num_workers=min(num_workers, len(self.annotation_dataset_writer)),
            collate_fn=_collate_fn,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
        )
        all_images_paths = []
        all_labels_paths = []
        all_used_colors = set()

        for images_paths, labels_path, used_colors in tqdm(
            annotation_writer_dataloder,
            total=len(self.annotation_dataset_writer),
            disable=not self.progress,
        ):
            all_images_paths.extend(images_paths)
            all_labels_paths.extend(labels_path)
            all_used_colors.update(used_colors)

        if self.filter_color_labels:
            self.color_labels = ColorLabels.from_filter_by_colors(
                self.color_labels, all_used_colors
            )

        self.color_labels.to_json(self.color_labels_file_path)

        data = self.paths_to_data(all_images_paths, all_labels_paths)

        data.to_csv(self.csv_path, header=False, index=False)

        return data

    def write_csv(self, images_paths, labels_paths):
        csv_str = "\n".join(
            f"{image_path},{label_path}"
            for image_path, label_path in zip(images_paths, labels_paths)
        )
        with open(self.csv_path, "w", encoding="utf-8") as outfile:
            outfile.write(csv_str)

    def paths_to_data(self, images_paths, labels_paths) -> pd.DataFrame:
        data = pd.DataFrame([images_paths, labels_paths]).T
        data.columns = ["image", "label"]
        if self.only_basename:
            data = data.applymap(lambda path: os.path.basename(path))
        return data

    @classmethod
    def from_reader(
        cls,
        annotation_reader: AnnotationReader,
        color_labels: ColorLabels,
        labels_dir: str,
        images_dir: Optional[str] = None,
        color_labels_file_path: Optional[str] = None,
        csv_path: Optional[str] = None,
        annotation_painter: Optional[Lazy[AnnotationPainter]] = None,
        only_basename: bool = False,
        copy_images: bool = False,
        overwrite: bool = True,
        resizer: Optional[FixedResize] = None,
        compose: Optional[Compose] = None,
        filter_color_labels: bool = True,
        progress: bool = False,
    ):
        if annotation_painter:
            annotation_painter = annotation_painter.construct(color_labels=color_labels)
        else:
            annotation_painter = AnnotationPainter(color_labels)

        if color_labels_file_path is None:
            color_labels_file_path = os.path.join(labels_dir, "color_labels.json")
        if csv_path is None:
            csv_path = os.path.join(labels_dir, "data.csv")

        return cls(
            annotation_reader.annotation_iterator,
            annotation_painter,
            color_labels,
            labels_dir,
            color_labels_file_path,
            csv_path,
            images_dir,
            only_basename,
            copy_images,
            overwrite,
            resizer,
            compose,
            filter_color_labels,
            progress,
        )


def _collate_fn(examples):
    if not isinstance(examples, list):
        examples = [examples]
    all_images_paths = []
    all_labels_path = []
    all_used_colors = set()
    for image_path, label_path, used_colors in examples:
        all_images_paths.append(image_path)
        all_labels_path.append(label_path)
        all_used_colors.update(used_colors)
    return all_images_paths, all_labels_path, all_used_colors
