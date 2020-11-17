from typing import Union, List, Optional, TypeVar

import numpy as np
from dh_segment_torch.data.annotation.image_size import ImageSize
from shapely import geometry

from dh_segment_torch.config.registrable import Registrable
from dh_segment_torch.data.annotation import Shape
from dh_segment_torch.post_processing.utils import merge_lists

T = TypeVar("T")
U = TypeVar("U")


class Operation(Registrable):
    def __call__(self, *args, **kwargs) -> U:
        return self.apply(*args, **kwargs)

    def apply(self, *args, **kwargs) -> U:
        raise NotImplementedError


# @Operation.register("concat_ops")
# class ConcatOps(Operation):
#     def __init__(self, ops: List[Operation]):
#         self.ops = ops
#
#     def apply(self, input: T, *args, **kwargs) -> U:
#         result = input
#         for op in self.ops:
#             result = op.apply(input, *args, **kwargs)
#         return result


@Operation.register("noop")
class NoOperation(Operation):
    def __init__(self):
        pass

    def apply(self, *args, **kwargs) -> U:
        return input


@Operation.register("extract_index")
class ExtractIndexOpration(Operation):
    def __init__(self, index: int):
        self.index = index

    def apply(self, input: List[T], *args, **kwargs) -> T:
        return input[self.index]


@Operation.register("split")
class ConcatLists(Operation):
    def __init__(self, operations_splits: List[List[Operation]]):
        self.operations_splits = operations_splits

    def apply(self, input: T, *args, **kwargs) -> U:
        results = []
        for operations_split in self.operations_splits:
            result = input
            for operation in operations_split:
                result = operation(result)
            results.append(result)
        return results


@Operation.register("merge_lists")
class MergeLists(Operation):
    def __init__(self, recursive: bool = False):
        self.recursive = recursive

    def apply(self, lists: List[List[T]], *args, **kwargs) -> List[T]:
        return merge_lists(lists, self.recursive)


@Operation.register("concat_lists")
class ConcatLists(Operation):
    def __init__(self):
        pass

    def apply(self, list_: List[T], *lists, **kwargs) -> List[T]:
        result = list(list_)
        for list_ in lists:
            result += list_
        return result

@Operation.register("probas_to_image_size")
class ProbasToImageSize(Operation):
    def __init__(self):
        pass

    def __call__(self, probas: np.array, *args, **kwargs) -> ImageSize:
        return self.apply(probas, *args, **kwargs)

    def apply(self, probas: np.array, *args, **kwargs) -> ImageSize:
        return ImageSize(*probas.shape[-2:])


class ClasswiseOperation(Operation):
    def __init__(self, classes_sel: Optional[Union[int, List[int]]] = None):
        if isinstance(classes_sel, int):
            classes_sel = [classes_sel]
        elif isinstance(classes_sel, list):
            if len(classes_sel) == 0:
                raise ValueError("Classes selection cannot be none.")
        self.classes_sel = classes_sel

    def apply(self, *args, **kwargs) -> U:
        raise NotImplementedError

    def apply_by_sel(self, input: T, *args, **kwargs) -> List[U]:
        classes_selection = self.classes_sel if self.classes_sel else range(len(input))
        result = []
        for class_ in classes_selection:
            result.append(self._apply_wrapper(input[class_]))
        return result

    def _apply_wrapper(self, input: T, *args, **kwargs) -> U:
        return self.apply(input, *args, **kwargs)


@Operation.register("classwise_noop")
class ClasswiseNoOperation(ClasswiseOperation):
    def __call__(self, input: T, *args, **kwargs) -> U:
        return self.apply_by_sel(input, *args, **kwargs)

    def apply(self, input: T, *args, **kwargs) -> U:
        return input


class BinaryToGeometriesOperation(Operation):
    def __call__(
        self, binary: np.array, *args, **kwargs
    ) -> List[geometry.base.BaseGeometry]:
        binary = binary.astype(np.uint8)
        if len(set(np.unique(binary).tolist()).difference({0, 1})) != 0:
            raise ValueError("Input should be binary, got more than 0 and 1 values.")
        return self.apply(binary, *args, **kwargs)

    def apply(
        self, binary: np.array, *args, **kwargs
    ) -> List[geometry.base.BaseGeometry]:
        raise NotImplementedError


# class BinaryToGeometriesOperation(ClasswiseOperation):
#     def __call__(self, binary: np.array, *args, **kwargs) -> List[List[geometry.base.BaseGeometry]]:
#         binary = binary.astype(np.uint8)
#         if len(set(np.unique(binary).tolist()).difference({0, 1})) != 0:
#             raise ValueError("Input should be binary, got more than 0 and 1 values.")
#         return self.apply_by_sel(binary, *args, **kwargs)
#
#     def apply(self, binary: np.array, *args, **kwargs) -> List[geometry.base.BaseGeometry]:
#         raise NotImplementedError


class GeometriesToGeometriesOperation(ClasswiseOperation):
    def __call__(
        self, geometries: List[List[geometry.base.BaseGeometry]], *args, **kwargs
    ) -> List[List[geometry.base.BaseGeometry]]:
        return self.apply_by_sel(geometries, *args, **kwargs)

    def apply(
        self, input: List[geometry.base.BaseGeometry], *args, **kwargs
    ) -> List[geometry.base.BaseGeometry]:
        raise NotImplementedError


class GeometriesToShapesOperation(ClasswiseOperation):
    def __init__(self):
        super().__init__()
        pass

    def apply(
        self, input: List[geometry.base.BaseGeometry], *args, **kwargs
    ) -> List[Shape]:
        return [self.apply_to_geom(geom, *args, **kwargs) for geom in input]

    def apply_to_geom(
        self, input: geometry.base.BaseGeometry, *args, **kwargs
    ) -> Shape:
        raise NotImplementedError
