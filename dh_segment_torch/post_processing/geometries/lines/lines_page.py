from typing import List

import cv2
import numpy as np
from shapely import geometry

from dh_segment_torch.post_processing.operation import (
    BinaryToGeometriesOperation,
    Operation,
)


class LinesPage(BinaryToGeometriesOperation):
    def __init__(
        self,
        center_angle: int,
        angle_variance: float = 2,
        vote_threshold: int = 100,
        rho_res: float = 1.0,
        theta_res: float = 0.25,
    ):
        """
        Runs a hough transform on a binary image
        Make arguments human understandable compared to opencv
        All angles are degrees

        :param center_angle: angle on which we want to center, 180 = vertical, 90 = horizontal
        :param angle_variance: how much variance around center we allow (center_angle +- angle_variance)
        :param vote_threshold: number of votes to be considered valid (to be tweaked depends on input image)
        :param rho_res: resolution for the rho argument
        :param theta_res: degree step to consider for angle
        """
        self.center_angle = center_angle
        self.angle_variance = angle_variance
        self.vote_threshold = vote_threshold
        self.rho_res = rho_res
        self.theta_res = theta_res

    def apply(
        self, binary: np.array, *args, **kwargs
    ) -> List[geometry.base.BaseGeometry]:
        theta_res = np.deg2rad(self.theta_res)
        center_angle = np.deg2rad(self.center_angle)
        angle_variance = np.deg2rad(self.angle_variance)

        lines = cv2.HoughLines(
            binary,
            self.rho_res,
            theta_res,
            self.vote_threshold,
            min_theta=center_angle - angle_variance,
            max_theta=center_angle + angle_variance,
        )
        if lines is None or len(lines) <= 0:
            return []
        lines = lines.reshape(-1, 2)
        lines = [line2coords(line, binary.shape[:2]) for line in lines]
        return list(geometry.MultiLineString(lines))

    @classmethod
    def vertical_lines(
        cls,
        angle_variance: float = 2,
        vote_threshold: int = 100,
        rho_res: float = 1.0,
        theta_res: float = 0.25,
    ):
        return cls(0, angle_variance, vote_threshold, rho_res, theta_res)

    @classmethod
    def horizontal_lines(
        cls,
        angle_variance: float = 2,
        vote_threshold: int = 100,
        rho_res: float = 1.0,
        theta_res: float = 0.25,
    ):
        return cls(90, angle_variance, vote_threshold, rho_res, theta_res)


Operation.register("lines_page")(LinesPage)
Operation.register("vertical_lines_page", "vertical_lines")(LinesPage)
Operation.register("horizontal_lines_page", "horizontal_lines")(LinesPage)


def line2coords(line, shape):
    xmax, ymax = shape
    rho, theta = line
    a = np.cos(theta)
    b = np.sin(theta)

    ysmall, xsmall = rho / a, rho / b
    ylarge, xlarge = (rho - xmax * b) / a, (rho - ymax * a) / b

    coords = []

    if 0 <= xsmall <= xmax:
        coords.append((0, int(round(xsmall))))
    if 0 <= ysmall <= ymax:
        coords.append((int(round(ysmall)), 0))
    if 0 <= xlarge <= xmax:
        coords.append((ymax, int(round(xlarge))))
    if 0 <= ylarge <= ymax:
        coords.append((int(round(ylarge)), xmax))
    if len(coords) < 2:
        print(coords, line)
        print((ysmall, xsmall), (ylarge, xlarge))
        raise ValueError("Line did not fit in shape.")
    return coords[:2]
