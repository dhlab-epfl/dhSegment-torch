from pathlib import Path
from typing import List, Union, Optional

from dh_segment_torch.config.registrable import Registrable
from dh_segment_torch.data.annotation.annotation import Annotation


class AnnotationReader(Registrable):
    def read(self, path: str, image_dir: Optional[str] = None) -> List[Annotation]:
        raise NotImplementedError

    def read_paths(self, paths: List[str], image_dirs: Optional[Union[List[str], str]] = None) -> List[Annotation]:
        if image_dirs is not None:
            if isinstance(image_dirs, str):
                image_dirs = [image_dirs] * len(paths)
            elif len(image_dirs) != len(paths):
                raise ValueError("Got a list of image dirs, but it did not match the list of paths.")
            paths = [{'path': path, 'image_dir': image_dir} for path, image_dir in zip(paths, image_dirs)]
        else:
            paths = [{'path': path} for path in paths]
        return [annot for path in paths for annot in self.read(**path)]
