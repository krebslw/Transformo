
from __future__ import annotations

from typing import Literal, Optional, Union

import numpy as np
from pyproj import Transformer
import rasterio
from rasterio import Affine

from transformo._typing import CoordinateMatrix, Matrix, Vector
from transformo.core import Operator
from transformo.datatypes import Parameter


def _read_geotiff_metadata(geotiff: str):
    """read metadata from geotiff"""
    with rasterio.open(geotiff, "r") as src:
        metadata = dict(
            meta = src.meta.copy(),
            width = src.width,
            height = src.height,
            transform = src.transform,
            crs = src.crs,
            nodata = src.nodata,
        )
    return metadata

def _write_geotiff(geotiff_out: str, data_flat: np.ndarray, metadata: dict):
    """
    write a geotiff
    """
    meta, crs, transform, height, width = (metadata[p] for p in ["meta", "crs", "transform", "height", "width"])
    out_meta = meta.copy()
    out_meta.update(
        {
            "count": 1,
            "dtype": "float32",
            "crs": crs,
            "transform": transform,
            "nodata": None,
        }
    )

    data = data_flat.reshape((height, width))
    with rasterio.open(geotiff_out, "w", **out_meta) as dst:
        dst.write(data.astype("float32"), 1)


def _get_prediction_points(geotiff: str):
    """
    Use extent and stepsize from a geotiff to build lists of prediction points
    """
    metadata = _read_geotiff_metadata(geotiff)

    height, width, transform = (metadata[p] for p in ["height", "width", "transform"])

    # Build flattened lists of pixel centres
    rows = np.repeat(np.arange(height), width)
    cols = np.tile(np.arange(width), height)
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    return xs, ys


def _pairwise_euclidean_distances(
        geogr1: tuple[np.ndarray],
        geogr2: tuple[np.ndarray] = None,
    ) -> np.ndarray:
    """
    Compute euclidean distances between points given in geographic coordinates

    `geogr` are assumed to be tuples containing two or three equal length 1D-arrays, each
    representing longitude, latitude and possibly ellipsoidal height coordinate
    components. If height is not given, 0 is assumed.

    `geogr1` and `geogr2` can have different number of points and if `geogr2` is not
    given, the distances between all points in `geogr1` are then computed.
    """
    if geogr2 is None:
        geogr2 = geogr1

    xyz1 = _transform_geogr_2_geocentr(*geogr1)
    xyz2 = _transform_geogr_2_geocentr(*geogr2)

    xyz1 = np.vstack([*xyz1]).T
    xyz2 = np.vstack([*xyz2]).T

    coordinate_diffs = xyz1[:, None, :] - xyz2[None, :, :]
    distances = np.sqrt((coordinate_diffs**2).sum(-1))
    return distances

def _transform_geogr_2_geocentr(lon: np.ndarray, lat: np.ndarray, ellpsh: np.ndarray=None):
    """
    Transform lists of coordinates from geographic to geocentric

    Ellipsoidal height of 0 is assumed if not given.
    """
    if ellpsh is None:
        ellpsh = np.zeros(lon.shape)

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:4936", always_xy=True)
    x, y, z = transformer.transform(lon, lat, ellpsh)

    return np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(z, dtype=float)


def _covariance_function(distances: np.ndarray, c0: float, alpha: float) -> np.ndarray:
    """
    Compute covariance between points, given their pairwise-distances

    A function of the form:

        c(s) = c0 * (1 + s/alpha) * exp(-s/alpha)

        1/2 * c(0) = c(x)
        => 1 == 2 * (1 + x/alpha) * exp(-x/alpha)
        => 1 == 2 * exp(-x/alpha) + 2 * x/alpha * exp(-x/alpha)
    is used, where `c0` is the assumed prior variance (i.e. the sill estimated from variogram analysis),
    and alpha is the range-parameter, i.e. the characteristic spatial scale of features. In case
    """

    # ensure alpha > 0
    alpha = float(max(alpha, 1e-12))
    return c0 * (1.0 + distances / alpha) * np.exp(-distances / alpha)


class KrigingOperator(Operator):
    """
    Estimate kriging weights
    """

    type: Literal["kriging"] = "kriging"

    # Parameters
    kriging_weights: Optional[list[float]] = []
    residual_surface_at_gnsslev: Optional[list] = None
    alpha: Optional[float] = None
    c0_min: Optional[float] = None
    c0: Optional[float] = None
    obs_var: Optional[float] = None
    nugget: Optional[float] = None
    inv_cov_xx: Optional[list] = None
    xmin: Optional[float] = None
    xmax: Optional[float] = None
    ymin: Optional[float] = None
    ymax: Optional[float] = None
    nx: Optional[float] = None
    yn: Optional[float] = None
    geotiffin: Optional[str] = None
    geotiffout: Optional[str] = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _parameter_list(self) -> list[Parameter]:
        params: list[Parameter] = []

        params.append(Parameter("alpha", self.alpha))
        params.append(Parameter("c0_min", self.c0_min))
        params.append(Parameter("c0", self.c0))
        params.append(Parameter("nugget", self.nugget))


        return params

    def _proj_name(self) -> str:
        return "gugugaga this attribute is required"

    def forward(self, coordinates: CoordinateMatrix) -> CoordinateMatrix:
        """
        Forward method of the operator.

        Takes as input the source coordinates, applies kriging weights
        and
        """
        coords = coordinates.copy()

        # Compute as Z^T * W, where both Z and W are column vectors. (in case
        # of multiple prediction points, W has multiple columns)
        modelled_residual_surface = (
            self.residual_surface_at_gnsslev[None, :] @ self.kriging_weights
        )

        # Add modelled residual surface
        coords[:, 2] = coords[:, 2] + modelled_residual_surface
        return coords

    # def inverse(self, coordinates: CoordinateMatrix) -> CoordinateMatrix:
    #   no inverse method, since this is not a Coordinate operation

    def invert_covariance_matrix(
        self,
        x: np.ndarray,
        y: np.ndarray,
        residuals: np.ndarray,
        obs_var: Optional[Union[float, list, np.ndarray]] = None,
    ):
        n = len(x)

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
        obs_var_arr = np.maximum(obs_var_arr, self.c0_min)

        # Estimer proces-varians c0 (sill) fra residualerne
        c0 = float(np.var(residuals, ddof=1))
        c0 = max(c0, self.c0_min)
        self.c0 = c0
        print(c0)

        distances = _pairwise_euclidean_distances(x=(x, y))

        # Kovariansfunktion
        cov_process_dd = _covariance_function(distances, c0, self.alpha)

        # Læg observationers varians (obs_var) og nugget på diagonalen
        diag_add = obs_var_arr + float(self.nugget + 1e-12)
        cov_dd = cov_process_dd + np.eye(n) * diag_add

        # Build OK-system
        K = np.zeros((n + 1, n + 1))
        K[:n, :n] = cov_dd
        K[:n, n] = 1.0
        K[n, :n] = 1.0
        K[n, n] = 0.0

        inv_cov_xx = np.linalg.inv(K)

        return inv_cov_xx


    def _compute_weights(self, x, y, px, py, inv_cov_xx: np.ndarray):
        """
        Compute kriging weights

        Uses the already inverted covariance matrix of observation
        """
        n = len(x)
        m = len(px)

        distances = _pairwise_euclidean_distances(x=(x, y), y=(px, py))
        cov_xy = _covariance_function(distances, self.c0, self.alpha)

        RHS = np.vstack([cov_xy, np.ones((1, m))])

        sol = inv_cov_xx@RHS
        weights = sol[:n, :]

        return weights

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

        Ordinary kriging med eksplicit kovariansfunktion.

        obs_var er a priori variansen på den observerede geoide Nobs

        Parametre (nye relevante):
        - transform_coords: hvis True transformeres x_list/y_list og predict_x/predict_y
                            fra input_crs til target_crs vha. pyproj. Default True.
        - input_crs, target_crs: CRS strings, fx "EPSG:4326" -> "EPSG:25832".

        Øvrige parametre som tidligere (z_list, alpha, nugget, obs_var, ...).
        """
        x = source_coordinates[:, 0]
        y = source_coordinates[:, 1]

        sz = source_coordinates[:, 2]
        tz = target_coordinates[:, 2]

        residuals = tz-sz
        self.residual_surface_at_gnsslev = residuals

        # Compute and invert covariance matrix (Cxx), and set self.invCxx
        obs_var = self.obs_var
        inv_cov_xx = self.invert_covariance_matrix(
            x,
            y,
            residuals,
            obs_var,
        )
        self.inv_cov_xx = inv_cov_xx

        # If geotiffin is given, we compute the residual surface at the
        # geotiff grid points and save the resulting geotiff
        if self.geotiffin:

            # read gravgeoid geotiff to get prediction points
            px, py =_get_prediction_points(self.geotiffin)

            weights = self._compute_weights(x, y, px, py, inv_cov_xx)

            Zhat_flat = residuals[None, :] @ weights

            # Write output raster, using metadata from input raster
            metadata = _read_geotiff_metadata(self.geotiffin)
            _write_geotiff(self.geotiffout, data_flat=Zhat_flat, metadata=metadata)

        # Now we compute the fitted geoid in the same points as source
        # i.e. px,py = x,y
        self.kriging_weights = self._compute_weights(x, y, x, y, inv_cov_xx)

        # TODO: her kunne vi også beregne kriging variansen således
        # Kriging-varians (proces-varians): c0 - c^T w + mu
        # cTw = np.einsum("im,im->m", cov_xy, weights)
        # sigma2 = c0 - cTw + mu
        # sigma2 = np.maximum(sigma2, 0.0)

        return


if __name__ == "__main__":
    xx = np.array([1,2,3,4,5,6])
    yy = np.array([1,2,3,4,5,6])
    zz = np.array([1,2,3,4,5,6])

    dd = _pairwise_euclidean_distances((xx,))
    print(dd)

    dd = _pairwise_euclidean_distances((xx, yy))
    print(dd)

    dd = _pairwise_euclidean_distances((xx, yy, zz))
    print(dd)

    dd = _pairwise_euclidean_distances((xx, xx,))
    print(dd)