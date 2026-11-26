
from __future__ import annotations

import csv
import enum
from typing import Literal, Optional, Union
from pathlib import Path

import numpy as np
from pyproj import Transformer
import rasterio
from rasterio import Affine

from transformo._typing import CoordinateMatrix, Matrix, Vector
from transformo.core import Operator,CoordinateOverrides
from transformo.datatypes import Parameter
from transformo.datasources import CsvDataSource, DataSource


import pyproj

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
        X_geogr: tuple[np.ndarray],
        Y_geogr: tuple[np.ndarray] = None,
    ) -> np.ndarray:
    """
    Compute euclidean distances between points given in geographic coordinates

    `geogr` are assumed to be tuples containing two or three equal length 1D-arrays, each
    representing longitude, latitude and possibly ellipsoidal height coordinate
    components. If height is not given, 0 is assumed.

    `geogr1` and `geogr2` can have different number of points and if `geogr2` is not
    given, the distances between all points in `geogr1` are then computed.
    """
    if Y_geogr is None:
        Y_geogr = X_geogr

    X_geocentr = _transform_geogr_2_geocentr(*X_geogr)
    Y_geocentr = _transform_geogr_2_geocentr(*Y_geogr)

    X_geocentr = np.vstack([*X_geocentr]).T
    Y_geocentr = np.vstack([*Y_geocentr]).T

    coordinate_diffs = X_geocentr[:, None, :] - Y_geocentr[None, :, :]
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


def _covariance_function(distances: np.ndarray, c0: float, alpha: float = None) -> np.ndarray:
    """
    Compute covariance between points, given their pairwise-distances

    A "2nd order Markov" function of the form:

        c(s) = c0 * (1 + s/alpha) * exp(-s/alpha)

    is used, where `c0` is the assumed prior variance (i.e. the sill estimated from variogram analysis),
    and alpha is the range-parameter, i.e. the characteristic spatial scale of features.

    We have approximately that alpha = 0.595*xhalf, where xhalf is the "halving distance" of the covariance
    function.
    """

    # ensure alpha > 0
    alpha = float(max(alpha, 1e-12))
    return c0 * (1.0 + distances / alpha) * np.exp(-distances / alpha)


class GeoidfitOperator(Operator):
    """
    A geoid fit operator that does everything

    Same as the gravsoft operator, this does not use the source and target coordinates of transformo
    In fact, theres no reason this operator should be in transformo and not be stand alone!
    """

    type: Literal["geoidfit"] = "geoidfit"

    # Parameters
    alpha: Optional[float] = None
    x_half: Optional[float] = None
    stddev_minimum: Optional[float] = None
    c0: Optional[float] = None

    # obs_var=observation variance.
    # # sættes manuelt for nu. Måske kan man gøre så den tages fra
    # datasourcens "sz" værdi, og bruge overrides=0.005**2 hvis data ikke har usikkerhed?
    # obs_var: Optional[float] = None

    nugget: Optional[float] = None
    # grid_extent: str = None # grid_extent is inferred from the geotiffin (gravgeoid.tif)

    gravsoft_data: str = None
    gnsslev_data: Optional[str] = None
    geotiffin: Optional[str] = None
    geotiffout: Optional[str] = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _parameter_list(self) -> list[Parameter]:
        params: list[Parameter] = []

        return params

    def _proj_name(self) -> str:
        return "noop"


    def forward(self, coordinates: CoordinateMatrix) -> CoordinateMatrix:
        """
        Forward method of the Gravsoft wrapper.

        The current implementation does not use the source- nor target-coordinates for anything!
        """
        return coordinates


    def invert_covariance_matrix(
        self,
        distances: np.ndarray,
        residuals: np.ndarray,
        obs_var: Optional[Union[float, list, np.ndarray]] = None,
    ):
        n = len(distances)
        c0_min = self.stddev_minimum**2

        if obs_var is None:
            obs_var = np.zeros(n, dtype=float)
        elif np.isscalar(obs_var):
            obs_var = np.full(n, obs_var, dtype=float)
        else:
            # ellers antages obs_var at være array med længde n
            pass

        # hvis observationernes varians er lavere end minimumsværdien sættes de til denne
        obs_var = np.maximum(obs_var, c0_min)

        # Estimer proces-varians c0 (sill) fra residualerne
        c0 = float(np.var(residuals, ddof=1))
        c0 = max(c0, c0_min)
        self.c0 = c0

        # Kovariansfunktion
        cov_process_dd = _covariance_function(distances, c0, self.alpha)

        # Læg observationers varians (obs_var) og nugget på diagonalen
        nugget = 0
        diag_add = obs_var + float(nugget + 1e-12)
        cov_dd = cov_process_dd + np.eye(n) * diag_add

        # Build OK-system
        K = np.zeros((n + 1, n + 1))
        K[:n, :n] = cov_dd
        K[:n, n] = 1.0
        K[n, :n] = 1.0
        K[n, n] = 0.0

        inv_cov_xx = np.linalg.inv(K)

        return inv_cov_xx

    def read_the_data(self) -> tuple[DataSource, DataSource]:
        koter = CsvDataSource(
            self.gnsslev_data,
            columns= [
                "station",
                "z",
                "skip",
                "skip",
                "y",
                "x",
                "skip",
                "skip",
                "t"
            ],
            # overrides=CoordinateOverrides(sx=0, sy=0, sz=0.005) # virker ikke
        )
        ellpsh = CsvDataSource(
            self.gnsslev_data,
            columns= [
                "station",
                "skip",
                "skip",
                "skip",
                "y",
                "x",
                "z",
                "skip",
                "t"
            ],
        )

        # construct new datasource by subtracting the z values of previous two
        geoid_observed_coords = koter.coordinate_matrix
        geoid_observed_coords[:,2] = ellpsh.coordinate_matrix[:,2] - koter.coordinate_matrix[:,2]

        geoid_obs = koter.update_coordinates(geoid_observed_coords)

        # gravimetric geoid is sampled at same locations
        geoid_grav_coords = self.sample_tif(koter.coordinate_matrix)

        geoid_grav = koter.update_coordinates(geoid_grav_coords)

        return geoid_obs, geoid_grav

    def read_gravsoft_data(self) -> DataSource:
        gs = CsvDataSource(
            self.gravsoft_data,
            columns= [
                "station",
                "y",
                "x",
                "t", # this is not t! but we dont need it but Transformo requires it...
                "z",
            ],
        )
        return gs

    def sample_tif(self, coordinates: CoordinateMatrix) -> CoordinateMatrix:
        """Sample tif-file using proj's vgridshift operator"""
        sampler = pyproj.Transformer.from_pipeline(f"+proj=vgridshift +grids={self.geotiffin} +multiplier=1")

        # set z value to zero so we just get the value of the tiff
        coordinates[:,2] = 0
        return np.asarray(sampler.transform(*coordinates.T)).T

    def estimate(
        self,
        source_coordinates: CoordinateMatrix,
        target_coordinates: CoordinateMatrix,
        source_weights: CoordinateMatrix,
        target_weights: CoordinateMatrix,
    ) -> None:
        """
        Estimate the fitted geoid to gnss-levelling data
        """
        # geoid_obs, geoid_grav = self.read_the_data()

        # use original data from DVR90(2023)
        geoid_obs = self.read_gravsoft_data()
        # gravimetric geoid is sampled at same locations
        geoid_grav_coords = self.sample_tif(geoid_obs.coordinate_matrix)
        geoid_grav = geoid_obs.update_coordinates(geoid_grav_coords)

        lon = geoid_obs.coordinate_matrix[:, 0]
        lat = geoid_obs.coordinate_matrix[:, 1]

        source_z = geoid_grav.coordinate_matrix[:,2]
        target_z = geoid_obs.coordinate_matrix[:, 2]

        # it is this residual surface between the observed geoid and the gravimetric geoid
        # that we are trying to model!
        residuals = target_z-source_z

        # Compute and invert covariance matrix (Cxx), to get inv(Cxx)
        self_distances = _pairwise_euclidean_distances(X_geogr=(lon, lat))
        inv_cov_xx = self.invert_covariance_matrix(
            self_distances,
            residuals,
            obs_var=0.005**2,
        )

        # read gravgeoid geotiff to get a list of all the grid points
        # these are used as prediction points for the kriging
        p_lon, p_lat =_get_prediction_points(self.geotiffin)

        # compute covariance matrix between all input points X (gnsslev) and the gridded
        # prediction points Y
        distances = _pairwise_euclidean_distances(X_geogr=(lon, lat), Y_geogr=(p_lon, p_lat))
        cov_xy = _covariance_function(distances, self.c0, self.alpha)

        # Construct the "right hand side" of the equation -- a matrix corresponding
        # to Cov_xy and 1's in the last row
        # [C(1,1), C(1,2), ..., C(1,M)]
        # [C(2,1), C(2,2), ..., C(2,M)]
        # [   ...,    ..., ...,    ...]
        # [C(N,1), C(N,2), ..., C(N,M)]
        # [   1  ,    1  , ...,   1   ]
        n = len(lon)
        m = len(p_lon)

        RHS = np.vstack([cov_xy, np.ones((1, m))])

        sol = inv_cov_xx@RHS

        weights = sol[:n, :]
        mu = sol[-1, :]

        # Compute predictions and their estimated variances
        Zhat_flat = residuals[None, :] @ weights

        # Kriging-varians (proces-varians): c0 - c^T w + mu
        cTw = np.einsum("im,im->m", cov_xy, weights)
        sigma2 = self.c0 - cTw + mu
        sigma2 = np.maximum(sigma2, 0.0)

        # Write output raster, using metadata from input raster
        metadata = _read_geotiff_metadata(self.geotiffin)
        self.geotiffout = Path(self.geotiffout)
        error_grid_filename = self.geotiffout.with_stem(self.geotiffout.stem+"_errors")

        _write_geotiff(self.geotiffout, data_flat=Zhat_flat, metadata=metadata)
        _write_geotiff(error_grid_filename, data_flat=sigma2, metadata=metadata)
        breakpoint()
        # Now we do all the same as above! But do it for the same points as the input points
        # i.e. p_lon, p_lat = lon, lat
        # mu, weights = self._compute_weights(lon, lat, lon, lat, inv_cov_xx)

        cov_xy = _covariance_function(self_distances, self.c0, self.alpha)

        n = len(lon)
        m = len(lon)

        RHS = np.vstack([cov_xy, np.ones((1, m))])

        sol = inv_cov_xx@RHS

        weights = sol[:n, :]
        mu = sol[-1, :]

        # Compute predictions and their estimated variances
        deltaN_flat = residuals[None, :] @ weights

        # Kriging-varians (proces-varians): c0 - c^T w + mu
        cTw = np.einsum("im,im->m", cov_xy, weights)
        sigma2 = self.c0 - cTw + mu
        sigma2 = np.maximum(sigma2, 0.0)


        # TODO: Save deltaN_flat and sigma2! these are the predictions of the geoid in the
        # fitting points. Subtract deltaN_flat from observed geoid in source points "geoid_obs"
        # get the "residuals" i.e. the main way we quantify the fitting!

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