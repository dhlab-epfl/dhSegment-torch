from typing import List, Tuple

from dh_segment_torch.data.annotation.shape import Shape
from dh_segment_torch.post_processing.operation import Operation


@Operation.register("assign_label")
class AssignLabel(Operation):
    def __init__(self, label: str):
        self.label = label

    def apply(self, input: List[Shape], *args, **kwargs) -> List[Tuple[str, Shape]]:
        return list(zip([self.label] * len(input), input))
