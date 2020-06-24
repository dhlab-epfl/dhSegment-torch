from typing import Any, Union, List, Optional, TypeVar

import numpy as np
from shapely import geometry

from dh_segment_torch.config.registrable import Registrable
from dh_segment_torch.post_processing.utils import recursively_merge

T = TypeVar("T")
U = TypeVar("U")


class Operation(Registrable):
    def __call__(self, input: T) -> U:
        return self.apply(input)

    def apply(self, input: T) -> U:
        raise NotImplementedError


@Operation.register('noop')
class NoOperation(Operation):
    def __init__(self):
        pass
    def apply(self, input: T) -> U:
        return input


@Operation.register("split")
class SplitOperation(Operation):
    def __init__(self, operations_splits: List[List[Operation]]):
        self.operations_splits = operations_splits

    def apply(self, input: T) -> U:
        results = []
        for operations_split in self.operations_splits:
            result = input
            for operation in operations_split:
                result = operation(result)
            results.append(result)
        return results


@Operation.register("merge_lists")
class MergeListsOperation(Operation):
    def __init__(self):
        pass
    def apply(self, lists: List[List[T]]) -> List[T]:
        return recursively_merge(lists)


@Operation.register("intermediary_output")
class IntermediaryOutput(SplitOperation):
    def __init__(self, operations: List[Operation]):
        super().__init__([[NoOperation()], operations])


class ClasswiseOperation(Operation):
    def __init__(self, classes_sel: Optional[Union[int, List[int]]] = None):
        if isinstance(classes_sel, int):
            classes_sel = [classes_sel]
        elif isinstance(classes_sel, list):
            if len(classes_sel) == 0:
                raise ValueError("Classes selection cannot be none.")
        self.classes_sel = classes_sel

    def apply_by_sel(self, input: T) -> List[U]:
        classes_selection = self.classes_sel if self.classes_sel else range(len(input))
        result = []
        for class_ in classes_selection:
            result.append(self._apply_wrapper(input[class_]))
        return result

    def _apply_wrapper(self, input: T) -> U:
        return self.apply(input)


class ProbasOperation(ClasswiseOperation):
    def __call__(self, probas: np.array) -> np.array:
        return np.stack(self.apply_by_sel(probas))

    def apply(self, input: np.array) -> np.array:
        raise NotImplementedError


class ProbasIntOperation(ProbasOperation):
    def _apply_wrapper(self, probas: np.array) -> np.array:
        assert probas.ndim == 2
        probas_int = np.uint8(probas * 255)
        probas_transformed = self.apply(probas_int)
        return (probas_transformed.astype(np.float64) / 255.0).astype(probas.dtype)


class BinaryToGeometriesOperation(ClasswiseOperation):
    def __call__(self, binary: np.array) -> List[List[geometry.base.BaseGeometry]]:
        binary = binary.astype(np.uint8)
        if len(set(np.unique(binary).tolist()).difference({0, 1})) != 0:
            raise ValueError("Input should be binary, got more than 0 and 1 values.")
        return self.apply_by_sel(binary)

    def apply(self, binary: np.array) -> List[geometry.base.BaseGeometry]:
        raise NotImplementedError


class GeometriesToGeometriesOperation(ClasswiseOperation):
    def __call__(
        self, geometries: List[List[geometry.base.BaseGeometry]]
    ) -> List[List[geometry.base.BaseGeometry]]:
        return self.apply_by_sel(geometries)

    def apply(
        self, input: List[geometry.base.BaseGeometry]
    ) -> List[geometry.base.BaseGeometry]:
        raise NotImplementedError
