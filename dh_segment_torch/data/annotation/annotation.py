from typing import Optional, Tuple

from dh_segment_torch.config.registrable import Registrable
from dh_segment_torch.data.annotation.image_size import ImageSize
from dh_segment_torch.data.annotation.labels_annotations import LabelsAnnotations
from dh_segment_torch.data.annotation.utils import (
    load_image,
    extract_image_basename,
    is_iiif_url,
    iiif_url_to_image_size,
)


class Annotation(Registrable):
    default_implementation = "default"

    def __init__(
        self,
        uri: str,
        image_id: Optional[str] = None,
        image_size: Optional[ImageSize] = None,
        labels_annotations: LabelsAnnotations = None,
        normalize_shapes: bool = True,
        auth: Optional[Tuple[str, str]] = None,
        cache_image: bool = True,
    ):
        """Holds an annotation for a file.

        It consists in an image with its path/url and a dict of labels annotations

        :param uri: The path or url to the image
        :param image_id: The optional image id, will be inferred from uri if not given
        :param image_size: The image size, will be inferred if not given
        :param labels_annotations: The label annotations dictionary
        :param normalize_shapes: whether to normalize the shapes of the annotations
        :param auth: an optional http auth tuple for fetching images
        :param cache_image: Whether to cache the image once read once
        """
        self.uri = uri
        self.auth = auth
        self.cache_image = cache_image
        self._image = None

        if image_id:
            self.image_id = image_id
        else:
            self.image_id = self.get_image_id()

        if image_size:
            self.image_size = image_size
        else:
            self.image_size = self.get_image_size()

        if labels_annotations:
            self.labels_annotations = labels_annotations
        else:
            self.labels_annotations = LabelsAnnotations()

        if normalize_shapes:
            self.labels_annotations.normalize_shapes(self.image_size)

    def get_image_id(self) -> str:
        return extract_image_basename(self.uri)

    def get_image_size(self) -> ImageSize:
        if self.is_iiif:
            return iiif_url_to_image_size(self.uri, self.auth)
        else:
            return ImageSize.from_image_array(self.image)

    @property
    def image(self):
        if self.cache_image:
            if self._image is None:
                self._image = load_image(self.uri, self.auth)
            return self._image
        else:
            return load_image(self.uri, self.auth)

    @property
    def is_iiif(self):
        return is_iiif_url(self.uri)


Annotation.register("default")(Annotation)
