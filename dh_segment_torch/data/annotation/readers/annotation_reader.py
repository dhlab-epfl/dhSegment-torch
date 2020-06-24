import logging
from typing import List, Union, Optional, Tuple

import pandas as pd

from dh_segment_torch.config.registrable import Registrable
from dh_segment_torch.data.annotation.annotation import Annotation
from dh_segment_torch.data.annotation.annotation_iterator import AnnotationIterator

logger = logging.getLogger(__name__)


class AnnotationReader(Registrable):
    def __init__(
        self,
        file_path: Union[str, List[str]],
        images_dir: Optional[str] = None,
        image_auth: Optional[Tuple[str, str]] = None,
    ):
        self.image_auth = image_auth
        self.file_path = file_path
        self.images_dir = images_dir
        self._annotation_iterator = None

    @property
    def annotation_iterator(self) -> AnnotationIterator:
        if self._annotation_iterator is None:
            if isinstance(self.file_path, str):
                self._annotation_iterator = self._read(self.file_path, self.images_dir)
            else:
                self._annotation_iterator = self._read_paths(
                    self.file_path, self.images_dir
                )
        return self._annotation_iterator

    def _read(self, path: str, image_dir: Optional[str] = None) -> AnnotationIterator:
        data = self._read_data(path, image_dir)
        return AnnotationIterator(data, self._transform_data_row_to_annot)

    def _read_paths(
        self, paths: List[str], image_dirs: Optional[Union[List[str], str]] = None
    ) -> AnnotationIterator:
        if image_dirs is not None:
            if isinstance(image_dirs, str):
                image_dirs = [image_dirs] * len(paths)
            elif len(image_dirs) != len(paths):
                raise ValueError(
                    "Got a list of image dirs, but it did not match the list of paths."
                )
            paths = [
                {"path": path, "image_dir": image_dir}
                for path, image_dir in zip(paths, image_dirs)
            ]
        else:
            paths = [{"path": path} for path in paths]
        datas = pd.concat([self._read_data(**path) for path in paths]).reset_index()
        return AnnotationIterator(datas, self._transform_data_row_to_annot)

    def _read_data(self, path: str, image_dir: Optional[str] = None) -> pd.DataFrame:
        raise NotImplementedError

    def _transform_data_row_to_annot(self, row: pd.Series) -> Annotation:
        raise NotImplementedError
