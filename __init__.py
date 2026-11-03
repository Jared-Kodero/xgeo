"""xgeo: geospatial analysis, plotting, and storage for climate data.

The top level holds only the light shared utilities: ``LockedLogger``,
``LockFile``, ``RedirectStreams``, ``locked_print``, ``exclude_key`` and
``nproc``. Everything else is reached through its module:

- ``plot``      Cartopy map plotting. Entry point :func:`xgeo.plot.geoplot`.
- ``plot_options``  Option groups for ``geoplot`` and ``animate``:
  ``Hillshade``, ``MapFeatures``, ``MapTicks``, ``ColorbarStyle``,
  ``ContourLabels``, ``Significance`` and ``VectorField``, used as
  ``xgeo.plot_options.Hillshade(...)``.
- ``cmaps``     Colormap registry spanning local IPCC tables, matplotlib,
  and cmocean.
- ``xr_utils``  Geospatial operations on xarray objects: ``regrid``, ``mask``,
  ``fillgaps``, ``sel_transect``, ``wrap_lon``, ``add_cyclic_point``,
  ``add_local_solar_time`` and ``SetupDask``.
- ``xnpy``      Memory-mapped array storage: ``to_xnpy``, ``open_dataset``,
  ``open_dataframe`` and ``load_array``.
- ``grib_io``   GRIB input and output: ``open_grib``, ``save_grib`` and
  ``get_grib_codes``.
- ``stats``     Trends, correlations, and difference-of-means testing.
- ``preprocess``  Dataset-specific preprocessing (ERA5, IMERG, CMORPH, GPCP).
- ``operator``  The standard operators plus common transformations.
- ``climtools`` Process utilities, including ``SharedMemoryObject``.

Serial NetCDF output uses xarray's native ``to_netcdf`` method. Distributed
arrays, MPI contexts, and parallel NetCDF output belong to the independent
``xrmpi`` package.

Two access patterns are supported and are equivalent::

    import xgeo as xg
    xg.plot.geoplot(da, method="contourf")

    import xgeo
    da.xgeo.plot.geoplot(method="contourf")

Importing the package registers the ``.xgeo`` accessor on
``xarray.DataArray`` and ``xarray.Dataset`` and, inside a Jupyter
kernel, applies the widget CSS fix and switches inline figures to retina
resolution.

Modules are imported on first attribute access through ``__getattr__``, so
each implementation module loads when first used. Regridding requires
``xesmf``, which is imported on first use. The rest of the package works
without it.
"""

from __future__ import annotations

import os
import warnings
from importlib import import_module
from typing import TYPE_CHECKING, Any

import xarray  # noqa: F401  must load before core.climtools (circular import)

from .core.accessors import fix_xarray
from .core.climtools import (
    LockedLogger,
    LockFile,
    RedirectStreams,
    apply_widget_css,
    exclude_key,
    locked_print,
    nproc,
)

if TYPE_CHECKING:
    from .core import (
        climtools,
        grib_io,
        operator,
        preprocess,
        stats,
        xnpy,
        xr_utils,
    )
    from .viz import cmaps, plot
    from .viz import options as plot_options

warnings.filterwarnings("ignore")
warnings.filterwarnings("always", module=r"xgeo\..*")

#: Light shared utilities and the public submodules.
__all__ = [
    "LockFile",
    "LockedLogger",
    "RedirectStreams",
    "climtools",
    "cmaps",
    "exclude_key",
    "grib_io",
    "locked_print",
    "nproc",
    "operator",
    "plot",
    "plot_options",
    "preprocess",
    "stats",
    "xnpy",
    "xr_utils",
]


_LAZY_MODULES: dict[str, str] = {
    "climtools": ".core.climtools",
    "cmaps": ".viz.cmaps",
    "grib_io": ".core.grib_io",
    "operator": ".core.operator",
    "plot": ".viz.plot",
    "plot_options": ".viz.options",
    "preprocess": ".core.preprocess",
    "stats": ".core.stats",
    "xnpy": ".core.xnpy",
    "xr_utils": ".core.xr_utils",
}


def __getattr__(name: str) -> Any:
    """Import a public submodule when it is first requested."""
    try:
        module_name = _LAZY_MODULES[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    module = import_module(module_name, __name__)
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    """Include lazily exported modules in interactive discovery."""
    return sorted(set(globals()) | set(_LAZY_MODULES))


try:
    fix_xarray()
except Exception:
    ...

apply_widget_css()
