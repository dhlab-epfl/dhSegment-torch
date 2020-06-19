from dataclasses import dataclass
from typing import Tuple
import numpy as np

from dh_segment_torch.config.registrable import Registrable


@dataclass
class ImageSize(Registrable):
    height: int
    width: int

    @classmethod
    def from_image_array(cls, image: np.array):
        return cls(height=image.shape[0], width=image.shape[1])

    def __add(self, other) -> Tuple[int, int]:
        if isinstance(other, ImageSize):
            return self.height + other.height, self.width + other.width
        elif isinstance(other, int):
            return self.height + other, self.width + other
        elif isinstance(other, float):
            return self.height + int(round(other)), self.width + int(round(other))
        elif isinstance(other, tuple) and list(map(type, other)) == [int, int]:
            return self.height + other[0], self.width + other[1]
        else:
            raise TypeError("Not supported for image size addition.")

    def __add__(self, other):
        new_height, new_width = self.__add(other)
        return ImageSize(height=new_height, width=new_width)

    def __iadd__(self, other):
        new_height, new_width = self.__add(other)
        self.height = new_height
        self.width = new_width
        return self

    def __sub(self, other) -> Tuple[int, int]:
        if isinstance(other, ImageSize):
            return self.height - other.height, self.width - other.width
        elif isinstance(other, int):
            return self.height - other, self.width - other
        elif isinstance(other, float):
            return self.height - int(round(other)), self.width - int(round(other))
        elif isinstance(other, tuple) and list(map(type, other)) == [int, int]:
            return self.height - other[0], self.width - other[1]
        else:
            raise TypeError("Not supported for image size substraction.")

    def __sub__(self, other):
        new_height, new_width = self.__add(other)
        return ImageSize(height=new_height, width=new_width)

    def __isub__(self, other):
        new_height, new_width = self.__add(other)
        self.height = new_height
        self.width = new_width
        return self

    def __mul(self, other) -> Tuple[int, int]:
        if isinstance(other, ImageSize):
            return self.height * other.height, self.width * other.width
        elif isinstance(other, int):
            return self.height * other, self.width * other
        elif isinstance(other, float):
            return self.height * int(round(other)), self.width * int(round(other))
        elif isinstance(other, tuple) and list(map(type, other)) == [int, int]:
            return self.height * other[0], self.width * other[1]
        else:
            raise TypeError("Not supported for image size multiplication.")

    def __mul__(self, other):
        new_height, new_width = self.__mul(other)
        return ImageSize(height=new_height, width=new_width)

    def __imul__(self, other):
        new_height, new_width = self.__mul(other)
        self.height = new_height
        self.width = new_width
        return self

    def __truediv(self, other) -> Tuple[int, int]:
        if isinstance(other, ImageSize):
            return int(round(self.height / other.height)), int(round(self.width / other.width))
        elif isinstance(other, int) or isinstance(other, float):
            return int(round(self.height / other)), int(round(self.width / other))
        elif isinstance(other, tuple) and list(map(type, other)) == [int, int]:
            return int(round(self.height / other[0])), int(round(self.width / other[1]))
        else:
            raise TypeError("Not supported for image size true division.")

    def __truediv__(self, other):
        new_height, new_width = self.__truediv(other)
        return ImageSize(height=new_height, width=new_width)

    def __itruediv__(self, other):
        new_height, new_width = self.__truediv(other)
        self.height = new_height
        self.width = new_width
        return self

    def __floordiv(self, other) -> Tuple[int, int]:
        if isinstance(other, ImageSize):
            return self.height // other.height, self.width // other.width
        elif isinstance(other, int):
            return self.height // other, self.width // other
        elif isinstance(other, float):
            return self.height // int(round(other)), self.width // int(round(other))
        elif isinstance(other, tuple) and list(map(type, other)) == [int, int]:
            return self.height // other[0], self.width // other[1]
        else:
            raise TypeError("Not supported for image size floor division.")

    def __floordiv__(self, other):
        new_height, new_width = self.__floordiv(other)
        return ImageSize(height=new_height, width=new_width)

    def __ifloordiv__(self, other):
        new_height, new_width = self.__floordiv(other)
        self.height = new_height
        self.width = new_width
        return self


ImageSize.register("image_size")(ImageSize)
