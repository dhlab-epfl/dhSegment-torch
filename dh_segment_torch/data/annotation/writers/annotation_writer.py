from dh_segment_torch.config.registrable import Registrable
from dh_segment_torch.data.annotation.annotation_iterator import AnnotationIterator


class AnnotationWriter(Registrable):
    def __init__(
        self,
        annotation_iterator: AnnotationIterator,
        overwrite: bool = True,
        progress: bool = False,
    ):
        self.annotation_iterator = annotation_iterator
        self.overwrite = overwrite
        self.progress = progress

    def write(self, num_workers: int = 0):
        raise NotImplementedError
