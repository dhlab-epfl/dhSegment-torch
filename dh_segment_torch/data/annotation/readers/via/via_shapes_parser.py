from typing import List, Dict, Union, Optional, Any

import numpy as np

from dh_segment_torch.data.annotation.shape import (
    Shape,
    Circle,
    Rectangle,
    Ellipse,
    LineString,
    Polygon,
)


def parse_via2_shape(
    shape_info: Dict[str, Any], point_radius: int = 5, line_thickness: int = 2,
) -> Optional[Shape]:
    if len(shape_info) <= 1:
        return None
    shape = shape_info["name"]

    if shape == "point":
        coordinates = [shape_info["cx"], shape_info["cy"]]
        return parse_point(coordinates, point_radius)
    elif shape == "rect":
        coordinates = [
            shape_info["x"],
            shape_info["y"],
            shape_info["width"],
            shape_info["height"],
        ]
        return parse_rectangle(coordinates)
    elif shape == "circle":
        coordinates = [shape_info["cx"], shape_info["cy"], shape_info["r"]]
        return parse_circle(coordinates)
    elif shape == "ellipse":
        coordinates = [
            shape_info["cx"],
            shape_info["cy"],
            shape_info["rx"],
            shape_info["ry"],
            shape_info["theta"],
        ]
        return parse_ellipse(coordinates)
    elif shape == "polyline":
        coordinates = [
            y
            for x in zip(shape_info["all_points_x"], shape_info["all_points_y"])
            for y in x
        ]
        return parse_linestring(coordinates, line_thickness)
    elif shape == "polygon":
        coordinates = [
            y
            for x in zip(shape_info["all_points_x"], shape_info["all_points_y"])
            for y in x
        ]
        return parse_polygon(coordinates)
    else:
        raise ValueError(f"The shape {shape} is not supported")


def parse_via3_shape(
    shape_info: List[Union[int, float]],
    id_to_shape: Dict[Union[float, str, int], str],
    point_radius: int = 5,
    line_thickness: int = 2,
) -> Optional[Shape]:
    if len(shape_info) <= 1:
        return None
    shape = id_to_shape[shape_info[0]]
    coordinates = shape_info[1:]

    if shape == "POINT":
        return parse_point(coordinates, point_radius)
    elif shape == "RECTANGLE":
        return parse_rectangle(coordinates)
    elif shape == "CIRCLE":
        return parse_circle(coordinates)
    elif shape == "ELLIPSE":
        return parse_ellipse(coordinates)
    elif shape == "LINE" or shape == "POLYLINE":
        return parse_linestring(coordinates, line_thickness)
    elif shape == "POLYGON":
        return parse_polygon(coordinates)
    elif shape == "EXTREME_RECTANGLE":
        return parse_extreme_rectangle(coordinates)
    elif shape == "EXTREME_CIRCLE":
        return parse_extreme_circle(coordinates)
    else:
        raise ValueError(f"The shape {shape} is not supported")


def parse_point(coordinates: List[float], radius: int = 5) -> Shape:
    return Circle((coordinates[0], coordinates[1]), radius, normalized_coords=False)


def parse_rectangle(coordinates: List[float]) -> Shape:
    x, y, w, h = coordinates
    return Rectangle(((x, y), (x + w, y + h)), normalized_coords=False)


def parse_extreme_rectangle(coordinates: List[float]) -> Shape:
    coordinates = np.array(coordinates).reshape(4, 2)
    tl = coordinates.min(axis=0).tolist()
    br = coordinates.max(axis=0).tolist()
    return parse_rectangle(tl + br)


def parse_circle(coordinates: List[float]) -> Shape:
    center = (coordinates[0], coordinates[1])
    radiuses = (coordinates[2], coordinates[2])
    return Ellipse(center, radiuses, angle=0, normalized_coords=False)


def parse_extreme_circle(coordinates: List[float]) -> Shape:
    x1, y1, x2, y2, x3, y3 = coordinates
    # Find the circle defined by three points: http://www.ambrsoft.com/trigocalc/circle3d.htm
    xy1_2 = x1 ** 2 + y1 ** 2
    xy2_2 = x2 ** 2 + y2 ** 2
    xy3_2 = x3 ** 2 + y3 ** 2

    A = x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2
    B = xy1_2 * (y3 - y2) + xy2_2 * (y1 - y3) + xy3_2 * (y2 - y1)
    C = xy1_2 * (x2 - x3) + xy2_2 * (x3 - x1) + xy3_2 * (x1 - x2)
    D = (
        xy1_2 * (x3 * y2 - x2 * y3)
        + xy2_2 * (x1 * y3 - x3 * y1)
        + xy3_2 * (x2 * y1 - x1 * y2)
    )

    x = -B / 2 * A
    y = -C / 2 * A
    r = np.sqrt((B ** 2 + C ** 2 - 4 * A * D) / (4 * A ** 2))

    return parse_circle([x, y, r])


def parse_ellipse(coordinates: List[float]) -> Shape:
    center = (coordinates[0], coordinates[1])
    radiuses = (coordinates[2], coordinates[3])
    if len(coordinates) == 5:
        angle = coordinates[4]
    else:
        angle = 0
    return Ellipse(center, radiuses, angle=angle, normalized_coords=False)


def parse_linestring(coordinates: List[float], line_thickness: int = 2) -> Shape:
    return LineString(
        np.array(coordinates).reshape(-1, 2).tolist(),
        line_thickness,
        normalized_coords=False,
    )


def parse_polygon(coordinates: List[float]) -> Shape:
    return Polygon(
        np.array(coordinates).reshape(-1, 2).tolist(), normalized_coords=False
    )
