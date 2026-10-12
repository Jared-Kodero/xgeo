"""xgeo: geospatial analysis, plotting, and storage for climate data.

The top level exposes the geospatial and plotting API:

- ``plot``      Cartopy map plotting. Entry point :func:`xgeo.plot.geoplot`.
- ``regrid``, ``mask``, ``fillgaps``, ``sel_transect``, ``to_lon180``,
  ``add_local_solar_time``  Geospatial operations on xarray objects.
- ``to_xnpy`` and the ``open_xnpy_*`` readers  Memory-mapped array storage.
- ``stats``     Trends, correlations, and difference-of-means testing.
- ``preprocess``  Dataset-specific preprocessing (ERA5, IMERG, CMORPH, GPCP).
- ``cmaps``     Colormap registry spanning local IPCC tables, matplotlib,
  and cmocean.

and the shared utilities: ``LockedLogger``, ``LockFile``, ``RedirectStreams``,
``locked_print``, ``exclude_key``, ``nproc``, ``SerialProgressBar``,
``DaskProgressBar`` and ``operator``. The CDO wrapper is the
module :mod:`xgeo.cdo.pycdo`.

Serial NetCDF output uses xarray's native ``to_netcdf`` method. Distributed
arrays, MPI contexts, and parallel NetCDF output belong to the independent
``xrmpi`` package.

Two access patterns are supported and are equivalent::

    import xgeo as xg
    xg.plot.geoplot(da, method="contourf")

    import xgeo
    da.xgeo.plot.geoplot(method="contourf")

Importing the package registers the ``.xgeo`` accessor on
``xarray.DataArray`` and ``xarray.Dataset``, replaces the dask progress bar
with the styled one from :mod:`xgeo.core.progress`, and, inside a Jupyter
kernel, applies the widget CSS fix and switches inline figures to retina
resolution.

Public names are resolved on first attribute access through ``__getattr__``,
so each implementation module loads when first used. Regridding requires
``xesmf``, which is imported on first use. The rest of the package works
without it.
"""

from __future__ import annotations

import os

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import warnings
from importlib import import_module
from typing import TYPE_CHECKING, Any

import dask.diagnostics
import xarray  # noqa: F401  must load before core.climtools (circular import)

from .core.accessors import fix_xarray
from .core.climtools import apply_widget_css
from .core.progress import DaskProgressBar
from .core.xnpy import to_xnpy

if TYPE_CHECKING:
    from .core import operator, preprocess, stats
    from .core.climtools import (
        LockedLogger,
        LockFile,
        RedirectStreams,
        SharedMemoryObject,
        exclude_key,
        locked_print,
        nproc,
    )
    from .core.grib_io import get_grib_codes, open_grib, save_grib
    from .core.progress import SerialProgressBar
    from .core.xnpy import (
        open_xnpy_dataframe,
        open_xnpy_dataset,
        open_xnpy_ndarray,
        to_xnpy,
    )
    from .core.xr_utils import (
        SetupDask,
        add_local_solar_time,
        fillgaps,
        mask,
        regrid,
        sel_transect,
        wrap_lon,
    )
    from .viz import cmaps, plot

warnings.filterwarnings("ignore")
warnings.filterwarnings("always", module=r"xgeo\..*")

#: Public geospatial, plotting, preprocessing, and storage names.
__all__ = [
    "DaskProgressBar",
    "LockFile",
    "LockedLogger",
    "RedirectStreams",
    "SerialProgressBar",
    "SetupDask",
    "SharedMemoryObject",
    "add_local_solar_time",
    "cmaps",
    "exclude_key",
    "fillgaps",
    "get_grib_codes",
    "locked_print",
    "mask",
    "nproc",
    "open_grib",
    "open_xnpy_dataframe",
    "open_xnpy_dataset",
    "open_xnpy_ndarray",
    "operator",
    "plot",
    "preprocess",
    "regrid",
    "save_grib",
    "sel_transect",
    "stats",
    "to_xnpy",
    "wrap_lon",
]


_LAZY_IMPORTS: dict[str, tuple[str, str | None]] = {
    "DaskProgressBar": (".core.progress", "DaskProgressBar"),
    "LockFile": (".core.climtools", "LockFile"),
    "LockedLogger": (".core.climtools", "LockedLogger"),
    "RedirectStreams": (".core.climtools", "RedirectStreams"),
    "SerialProgressBar": (".core.progress", "SerialProgressBar"),
    "SetupDask": (".core.xr_utils", "SetupDask"),
    "SharedMemoryObject": (".core.climtools", "SharedMemoryObject"),
    "add_local_solar_time": (".core.xr_utils", "add_local_solar_time"),
    "cmaps": (".viz.cmaps", None),
    "exclude_key": (".core.climtools", "exclude_key"),
    "fillgaps": (".core.xr_utils", "fillgaps"),
    "get_grib_codes": (".core.grib_io", "get_grib_codes"),
    "locked_print": (".core.climtools", "locked_print"),
    "mask": (".core.xr_utils", "mask"),
    "nproc": (".core.climtools", "nproc"),
    "open_grib": (".core.grib_io", "open_grib"),
    "save_grib": (".core.grib_io", "save_grib"),
    "open_xnpy_dataframe": (".core.xnpy", "open_xnpy_dataframe"),
    "open_xnpy_dataset": (".core.xnpy", "open_xnpy_dataset"),
    "open_xnpy_ndarray": (".core.xnpy", "open_xnpy_ndarray"),
    "operator": (".core.operator", None),
    "plot": (".viz.plot", None),
    "preprocess": (".core.preprocess", None),
    "regrid": (".core.xr_utils", "regrid"),
    "sel_transect": (".core.xr_utils", "sel_transect"),
    "stats": (".core.stats", None),
    "to_lon180": (".core.xr_utils", "to_lon180"),
    "to_xnpy": (".core.xnpy", "to_xnpy"),
}


def __getattr__(name: str) -> Any:
    """Import a public object when it is first requested."""
    try:
        module_name, attribute = _LAZY_IMPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    module = import_module(module_name, __name__)
    value = module if attribute is None else getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazily exported objects in interactive discovery."""
    return sorted(set(globals()) | set(_LAZY_IMPORTS))


try:
    fix_xarray()
except Exception:
    ...

apply_widget_css()
dask.diagnostics.ProgressBar = DaskProgressBar

# from .update import _self_update  # noqa: E402

# _self_update()
