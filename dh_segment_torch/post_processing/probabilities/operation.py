import numpy as np

from dh_segment_torch.post_processing.operation import Operation


class ProbasOperation(Operation):
    def __call__(self, probas: np.array, *args, **kwargs) -> np.array:
        return self._apply_wrapper(probas, *args, **kwargs)

    def apply(self, input: np.array, *args, **kwargs) -> np.array:
        raise NotImplementedError

    def _apply_wrapper(self, input: np.array, *args, **kwargs) -> np.array:
        return self.apply(input, *args, **kwargs)


class ProbasIntOperation(ProbasOperation):
    def apply(self, input: np.array, *args, **kwargs) -> np.array:
        raise NotImplementedError

    def _apply_wrapper(self, probas: np.array, *args, **kwargs) -> np.array:
        assert probas.ndim == 2
        probas_int = np.uint8(probas * 255)
        probas_transformed = self.apply(probas_int, *args, **kwargs)
        return (probas_transformed.astype(np.float64) / 255.0).astype(probas.dtype)
