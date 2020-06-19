import logging
from typing import List, Union, Optional, Tuple, Callable

import pandas as pd

from dh_segment_torch.config.registrable import Registrable
from dh_segment_torch.data.annotation.annotation import Annotation

logger = logging.getLogger(__name__)


class AnnotationIterator:
    def __init__(
        self, data: pd.DataFrame, row_transform: Callable[[pd.Series], Annotation]
    ):
        self.data = data
        self.row_transform = row_transform

    def __getitem__(self, index) -> Union[Annotation, List[Annotation]]:
        if isinstance(index, int):
            return self.row_transform(self.data.iloc[index])
        else:
            return (
                self.data.iloc[index].apply(self.row_transform, axis=1).values.tolist()
            )

    def __iter__(self) -> Annotation:
        for _, row in self.data.iterrows():
            yield self.row_transform(row)

    def __len__(self):
        return len(self.data)


class AnnotationReader(Registrable):
    def __init__(
        self,
        file_path: Union[str, List[str]],
        images_dir: Optional[str] = None,
        image_auth: Optional[Tuple[str, str]] = None,
    ):
        self.image_auth = image_auth
        if isinstance(file_path, str):
            self.annotation_iterator = self._read(
                file_path, images_dir
            )
        else:
            self.annotation_iterator = self._read_paths(
                file_path, images_dir
            )

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
