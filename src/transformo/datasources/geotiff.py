import os
from pathlib import Path
from typing import Literal

import pydantic
import pyproj

import numpy as np

from transformo import logger
from transformo.core import DataSource
from transformo.datatypes import Coordinate
from transformo._typing import CoordinateMatrix

import rasterio
from rasterio import Affine


# Tiff file utilities
def read_geotiff_metadata(geotiff: Path):
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

def write_geotiff(geotiff: Path, data_flat: np.ndarray, metadata: dict):
    """
    Write a geotiff from a flattened array

    The data is reshaped using the metadata height and width.

    Additional metadata, such as meta, crs, transfrom is copied into the new
    geotiff.
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
    with rasterio.open(geotiff, "w", **out_meta) as dst:
        dst.write(data.astype("float32"), 1)

def sample_geotiff(geotiff: Path, coordinates: CoordinateMatrix) -> CoordinateMatrix:
    """Sample a geotiff at the given coordinates using proj's vgridshift operator"""

    sampler = pyproj.Transformer.from_pipeline(f"+proj=vgridshift +grids={geotiff} +multiplier=1")

    # set z value to zero so we just get the value of the tiff
    coordinates[:,2] = 0
    return np.asarray(sampler.transform(*coordinates.T)).T

def get_geotiff_points(geotiff: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Get flattened lists of grid point coordinates from geotiff
    """
    metadata = read_geotiff_metadata(geotiff)

    height, width, transform = (metadata[p] for p in ["height", "width", "transform"])

    # Build flattened lists of pixel centres
    rows = np.repeat(np.arange(height), width)
    cols = np.tile(np.arange(width), height)

    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    return xs, ys

# class GeoTiffDataSource(DataSource):
#     """
#     Datasource for geotiffs
#     """
#     type: Literal["geotiff"] = "geotiff"
#     filename: os.PathLike | str
#     image = rasterio.io.DatasetReader

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         image = rasterio.open(self.filename)

#         # with rasterio.open(self.filename) as src:
#         #     meta = src.meta.copy()
#         #     width = src.width
#         #     height = src.height
#         #     transform: Affine = src.transform
#         #     crs = src.crs
#         #     nodata = src.nodata
#         #     # læs hele båndet (kan være stort)
#         #     tif_array = src.read(1).astype(float)
#         #     breakpoint()