from typing import Callable, Any, Union

import torch
from torch.utils.data import Dataset

from dh_segment_torch.data.annotation import AnnotationIterator, Annotation


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


def _collate_fn(examples):
    res = {}
    for example in examples:
        res.update(example)
    return res