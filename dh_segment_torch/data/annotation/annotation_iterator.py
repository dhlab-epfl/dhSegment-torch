from typing import Callable, Union, List

import pandas as pd

from dh_segment_torch.data.annotation.annotation import Annotation


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
