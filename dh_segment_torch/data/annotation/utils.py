import os
import re
import time
from typing import Tuple, Union, Dict, TypeVar, Optional, List

import cv2
import numpy as np
import requests

from dh_segment_torch.data.annotation.image_size import ImageSize

Coordinates = Tuple[Union[int, float], Union[int, float]]


def convert_coord_to_image(
    coord: Coordinates, height: int, width: int
) -> Tuple[int, int]:
    x, y = coord
    return int(round(x * width)), int(round(y * height))


def convert_coord_to_normalized(
    coord: Coordinates, height: int, width: int
) -> Tuple[float, float]:
    x, y = coord
    return x / float(width), y / float(height)


def int_coords(
    coords: List[Tuple[Union[float, int], Union[float, int]]]
) -> List[Tuple[int, int]]:
    return [(int(round(x)), int(round(y))) for x, y in coords]


T = TypeVar("T")
U = TypeVar("U")


def append_image_dir(uri: str, image_dir: str = None):
    if image_dir is None or uri.startswith("http"):
        return uri
    elif uri.startswith("file://"):
        return uri.replace("file://", "file://" + image_dir)
    else:
        return os.path.join(image_dir, uri)


def reverse_dict(dico: Dict[T, U]) -> Dict[U, T]:
    return {v: k for k, v in dico.items()}


def write_image(path: str, image: np.array, overwrite: bool = True):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if overwrite or not os.path.exists(path):
        cv2.imwrite(path, image)
    return path


def load_image(uri: str, auth: Optional[Tuple[str, str]] = None) -> np.array:
    if is_url(uri):
        image = load_image_from_url(uri, auth)
    else:
        uri = uri.replace("file://", "")
        image = load_image_from_fs(uri)
    return image


def load_image_from_fs(file_path: str) -> np.ndarray:
    """
    Create an image array from a path or bytes
    :param img_data: Path to the image or bytes
    :return: The image in BGR format
    """
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Unable to load image from path: {file_path}.")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_image_from_url(
    url: str, auth: Optional[str] = None, retry: int = 5
) -> np.ndarray:
    """
    Download an image from its url

    :param url: Url of the image\
    :param auth: Optional authentication under the form of a tuple (user,password)
    :return: A numpy array with the image data
    """
    # Try at least once
    if retry <= 0:
        retry = 1

    trials_left = retry

    while trials_left > 0:
        img_request = requests.get(url, auth=auth, stream=True)
        try:
            img_request.raise_for_status()
            break
        except requests.HTTPError:
            time.sleep(2 ** (retry - trials_left))  # Exponential backoff
            trials_left -= 1

    if not img_request.ok:
        raise ValueError(f"Could not fetch {url}")
    img_bytes = np.asarray(bytearray(img_request.content), dtype="uint8")
    image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to load image from url: {url}.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def is_url(uri: str) -> bool:
    if uri.startswith("http://") or uri.startswith("https://"):
        return True
    return False


iiif_regex = re.compile(
    r"(https?://"  # scheme
    r"(?:(.*?)/){2,})"  # server + (prefix) + identifier match is always on identifier
    r"(?:info.json|"  # either image information or image request
    r"("  # region
    r"full|"
    r"square|"
    r"(?:\d+,){3}\d+|"  # x,y,w,h
    r"pct:"  # pct:x,y,w,h but can be floats
    r"(?:[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?,){3}[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
    r"/("  # size
    r"full|"
    r"max|"
    r"\d+,|"
    r",\d+|"
    r"!?\d+,\d+"
    r"|pct:\d+)"
    r"/(!?[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"  # rotation
    r"/((?:color|gray|bitonal|default|native)"  # quality
    r".(?:jpg|tif|png|gif|jp2|pdf|webp)))"  # format
)


def is_iiif_url(url):
    return bool(iiif_regex.match(url))


def extract_image_filename(img_path: str) -> str:
    if is_iiif_url(img_path):
        return extract_iiif_filename(img_path)
    elif img_path.startswith("http"):
        return img_path.split("/")[-1]
    else:
        return os.path.basename(img_path)


def make_safe(string: str):
    keepcharacters = (" ", ".", "_")
    return "".join(c for c in string if c.isalnum() or c in keepcharacters).rstrip()


def extract_iiif_filename(iiif_url: str):
    first_part = iiif_regex.match(iiif_url)[1]
    without_address = first_part.split(".")[-1]
    return "_".join([make_safe(s) for s in without_address.split("/")[1:]]).strip("_")


def extract_image_ext(img_path: str) -> str:
    return "." + img_path.split(".")[-1]


def extract_image_basename(img_path: str) -> str:
    return extract_image_filename(img_path).split(".")[0]


def extract_image_name_with_ext(img_path: str) -> str:
    return extract_image_basename(img_path) + extract_image_ext(img_path)


def iiif_url_to_image_size(iiif_url, auth=None, retry: int = 5) -> ImageSize:
    if not iiif_url.endswith(".json"):
        iiif_url = iiif_regex.search(iiif_url).group(1) + "info.json"
        # Try at least once
    if retry <= 0:
        retry = 1

    trials_left = retry

    while trials_left > 0:
        try:
            r = requests.get(iiif_url, auth=auth)
            r.raise_for_status()
            return ImageSize(r.json()["height"], r.json()["width"])
        except requests.HTTPError:
            time.sleep(2 ** (retry - trials_left))  # Exponential backoff
            trials_left -= 1
        except ValueError:
            break
    raise ValueError(f"Unable to get image size from url: {iiif_url}.")


def iiif_url_to_resized(
    iiif_url: str,
    height: Optional[Union[int, float]] = None,
    width: Optional[Union[int, float]] = None,
) -> str:
    if height is None and width is None:
        raise ValueError("To resize, either height or width or both should be defined.")
    query_str = ""
    if height:
        query_str += str(int(height))
    query_str += ","
    if width:
        query_str += str(int(width))

    match = iiif_regex.match(iiif_url)

    return f"{match.group(1)}{match.group(3)}/{query_str}/{match.group(5)}/{match.group(6)}"


def iiif_url_to_manifest(iiif_url: str,) -> str:
    match = iiif_regex.match(iiif_url)
    return f"{match.group(1)}info.json"
