from typing import List, Any

import numpy as np

from dh_segment_torch.config.registrable import Registrable
from dh_segment_torch.post_processing.operation import Operation


class PostProcessingPipeline(Registrable):
    default_implementation = "default"

    # TODO intermediary result store to be used later in the pipeline ?

    def __init__(self, operations: List[Operation]):
        self.operations = operations

    def apply(self, probabilities: np.array) -> Any:
        result = probabilities
        for operation in self.operations:
            result = operation(result)
        return result


PostProcessingPipeline.register("default")(PostProcessingPipeline)
