
from __future__ import annotations

import enum
import math
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal, Optional, Tuple, Union

import numpy as np
import scipy.optimize
from numpy import cos, sin

from transformo._typing import CoordinateMatrix, Matrix, Vector
from transformo.core import Operator
from transformo.datatypes import Parameter

def _pairwise_euclidean_distances(*args: np.ndarray) -> np.ndarray:
    """Compute distances between all points"""
    coords = np.vstack([*args]).T
    diffs = coords[:, None, :] - coords[None, :, :]
    d = np.sqrt((diffs**2).sum(-1))
    return d


def _covariance_function(s: np.ndarray, c0: float, alpha: float) -> np.ndarray:
    """
    Compute covariance between points, given their pairwise-distances `s`

    c(s) = c0 * (1 + s/alpha) * exp(-s/alpha)
    """
    # ensure alpha > 0
    alpha = float(max(alpha, 1e-12))
    return c0 * (1.0 + s / alpha) * np.exp(-s / alpha)


class KrigingOperator(Operator):
    """
    Estimate kriging weights
    """

    type: Literal["kriging"] = "kriging"

    # Parameters
    kriging_weights: Optional[list[float]] = []

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # determine if the operation can estimate parameters or not
        # self._has_transformation_parameters_been_given()

        # self._sanitize_parameters()
        # ... from now on we can rely on parameters being useful


    def forward(self, coordinates: CoordinateMatrix) -> CoordinateMatrix:
        """
        Forward method of the 3 parameter Helmert.
        """
        coords = coordinates.copy()
        coords[:, 2] = coords[:, 2].dot(np.array(self.kriging_weights))
        return coords

        # preds = trend_pred + preds_resid # STKRE: læg ikke trend på igen
        # preds2 = retrend(px, py, preds_resid, coeff, method="bias")

        # Kriging-varians (proces-varians): c0 - c^T w + mu
        # cTw = np.einsum("im,im->m", cov_dp, weights)
        # sigma2 = c0 - cTw + mu
        # sigma2 = np.maximum(sigma2, 0.0)


    # def inverse(self, coordinates: CoordinateMatrix) -> CoordinateMatrix:
    #   no inverse method, since this is not a Coordinate operation

    def estimate(
        self,
        source_coordinates: CoordinateMatrix,
        target_coordinates: CoordinateMatrix,
        source_weights: CoordinateMatrix,
        target_weights: CoordinateMatrix,
    ) -> None:
        """
        Estimate kriging weights.

        Parameter `kriging_weights` of this operator *will* be overwritten once
        this method is called.

        Source coordinates antages på formen x,y,z hvor x,y = lat,lon og z="residuals", dvs
        (Nobs-Ngrav)-estimated_trend.

        Target coordinates er (lat,lon,Nobs).

        Men denne operator har ikke kendskab til Ngrav eller estimated_bias". Så vi finder bare
        den overflade som kan reduceres residuals til 0.

        coordinate_matrix = [[x1, y1, z1, t1], [x2, y2, z2, t2], ...]
        """

        avg_source = np.average(
            source_coordinates[:, 0:3], axis=0, weights=source_weights
        )
        avg_target = np.average(
            target_coordinates[:, 0:3], axis=0, weights=target_weights
        )

        mean_translation = avg_target - avg_source

        self.x = mean_translation[0]
        self.y = mean_translation[1]
        self.z = mean_translation[2]

def ordinary_kriging(

    # x_list: list,
    # y_list: list,
    # z_list: list = None, # kun relevant hvis geocentriske
    source_coordinates: CoordinateMatrix,
    target_coordinates: CoordinateMatrix,
    source_weights: CoordinateMatrix,
    target_weights: CoordinateMatrix,
    predict_x: Optional[list] = None,
    predict_y: Optional[list] = None,

    # kriging parameters:
    obs_var: Optional[Union[float, list, np.ndarray]] = None,
    alpha: float = 1.0,
    nugget: float = 0.0,
    c0_min: float = 0.005**2,  # min std afvigelse på 5 mm

) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ordinary kriging med eksplicit kovariansfunktion.

    obs_var er a priori variansen på den observerede geoide Nobs

    Parametre (nye relevante):
    - transform_coords: hvis True transformeres x_list/y_list og predict_x/predict_y
                        fra input_crs til target_crs vha. pyproj. Default True.
    - input_crs, target_crs: CRS strings, fx "EPSG:4326" -> "EPSG:25832".

    Øvrige parametre som tidligere (z_list, alpha, nugget, obs_var, ...).
    """
    n = source_coordinates.shape[0]
    x = source_coordinates[:, 0]
    y = source_coordinates[:, 1]
    residuals = source_coordinates[:, 2]

    # Behandl obs_var input
    if obs_var is None:
        obs_var_arr = np.zeros(n, dtype=float)
    else:
        if np.isscalar(obs_var):
            obs_var_arr = np.full(n, float(obs_var), dtype=float)
        else:
            obs_var_arr = np.asarray(obs_var, dtype=float)
            if obs_var_arr.shape[0] != n:
                raise ValueError("obs_var skal være length n (eller en skalar).")

    # hvis observationernes varians er lavere end minimumsværdien sættes de til denne
    obs_var_arr = np.maximum(obs_var_arr, c0_min)

    # Estimer proces-varians c0 (sill) fra residualerne
    c0 = float(np.var(residuals, ddof=1))
    c0 = max(c0, c0_min)

    # Parvise afstande (i målte enheder efter transformation)
    # For nu, vil enheden for være i (lat, lon) hvilket er meget forkert!!!
    # afstandende bør omregnes til kartesiske afstande på ellipsoiden
    # Dvs tag koordinaternes som lat,lon,0 og omregn til x,y,z og derefter regn afstand.
    D = _pairwise_euclidean_distances(x, y)

    # Kovariansfunktion
    cov_process_dd = _covariance_function(D, c0, alpha)

    # Læg observationers varians (obs_var) og nugget på diagonalen
    diag_add = obs_var_arr + float(nugget + 1e-12)
    cov_dd = cov_process_dd + np.eye(n) * diag_add

    # Build OK-system
    K = np.zeros((n + 1, n + 1))
    K[:n, :n] = cov_dd
    K[:n, n] = 1.0
    K[n, :n] = 1.0
    K[n, n] = 0.0

    # Forudsigelsespunkter: brug input coords hvis predict ikke angivet
    if predict_x is None or predict_y is None:
        px = x
        py = y
    else:
        px = np.asarray(predict_x, dtype=float)
        py = np.asarray(predict_y, dtype=float)
        # # Transformér predictionskoordinater hvis ønsket
        # px, py = _transform_coords_if_needed(
        #     px, py, transform_coords, input_crs, target_crs
        # )

    # Data-pred afstande og kovarians (brug proces-kovarians)
    dx = x[:, None] - px[None, :]
    dy = y[:, None] - py[None, :]
    D_dp = np.sqrt(dx * dx + dy * dy)

    cov_dp = _covariance_function(D_dp, c0, alpha)

    m = len(px)
    RHS = np.vstack([cov_dp, np.ones((1, m))])

    try:
        sol = np.linalg.solve(K, RHS)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(K, RHS, rcond=None)[0]

    weights = sol[:n, :]
    mu = sol[n, :]

    preds_resid = residuals.dot(weights)
    # preds = trend_pred + preds_resid # STKRE: læg ikke trend på igen
    # preds2 = retrend(px, py, preds_resid, coeff, method="bias")
    preds = preds_resid

    # Kriging-varians (proces-varians): c0 - c^T w + mu
    cTw = np.einsum("im,im->m", cov_dp, weights)
    sigma2 = c0 - cTw + mu
    sigma2 = np.maximum(sigma2, 0.0)

    return np.asarray(preds, dtype=float), np.asarray(sigma2, dtype=float)


if __name__ == "__main__":
    xx = np.array([1,2,3,4,5,6])
    yy = np.array([1,2,3,4,5,6])
    zz = np.array([1,2,3,4,5,6])

    dd = _pairwise_euclidean_distances(xx)
    print(dd)

    dd = _pairwise_euclidean_distances(xx, yy)
    print(dd)

    dd = _pairwise_euclidean_distances(xx, yy, zz)
    print(dd)

    dd = _pairwise_euclidean_distances(xx, xx,)
    print(dd)