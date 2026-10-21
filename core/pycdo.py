"""Minimal Python wrapper for the Climate Data Operators (CDO).

Functions
---------
remap      Remap a NetCDF file or xarray object onto a regular lon-lat grid.
remapbil, remapbic, remapnn, remapdis, remapcon, remaplaf
           ``remap`` with the method fixed.
mergetime  Concatenate NetCDF files along time.
run        Run any CDO command.

Examples
--------
>>> from xgeo import cdo
>>> cdo.remapcon("in.nc", "out.nc", resolution=0.5)
>>> da_025 = cdo.remapbil(da, resolution=0.25)  # xarray in, xarray out
>>> cdo.mergetime(["a.nc", "b.nc"], "merged.nc")
>>> cdo.run(["-timmean", "in.nc", "out.nc"])

For in-memory xarray data prefer ``obj.xgeo.regrid`` (xESMF); use this module
for file-to-file work on data that does not fit in memory.

CDO reference: https://code.mpimet.mpg.de/projects/cdo/embedded/cdo.pdf
"""

from __future__ import annotations

import math
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

import xarray as xr

Method = Literal["remapbil", "remapbic", "remapnn", "remapdis", "remapcon", "remaplaf"]

_METHODS = ("remapbil", "remapbic", "remapnn", "remapdis", "remapcon", "remaplaf")
_NO_EXTRAPOLATE = ("remapdis", "remapcon")
_LAT_NAMES = ("lat", "latitude")
_LON_NAMES = ("lon", "longitude")

__all__ = [
    "mergetime",
    "remap",
    "remapbic",
    "remapbil",
    "remapcon",
    "remapdis",
    "remaplaf",
    "remapnn",
    "run",
]


def run(
    cmd: list[str | os.PathLike[str]] | str,
    n_threads: int | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a CDO command, e.g. ``run(["-timmean", "in.nc", "out.nc"])``.

    ``cmd`` is a list or a string (split with :func:`shlex.split`); items may be
    paths. Options ``-s -O -P <n>`` are added. A leading ``"cdo"`` in ``cmd`` is
    ignored. ``env`` is applied to the subprocess only. Raises RuntimeError with
    the captured stderr if CDO is missing or fails.
    """
    if not shutil.which("cdo"):
        raise RuntimeError("CDO executable not found in PATH.")
    cmd = shlex.split(cmd) if isinstance(cmd, str) else [os.fspath(c) for c in cmd]
    if cmd and cmd[0] == "cdo":
        cmd = cmd[1:]
    threads = n_threads or min(8, os.cpu_count() or 1)
    full_env = {
        "CDO_VERSION_INFO": "false",
        "CDO_HISTORY_INFO": "false",
        **os.environ,
        **(env or {}),
    }
    try:
        return subprocess.run(
            ["cdo", "-s", "-O", "-P", str(threads), *cmd],
            check=True,
            text=True,
            capture_output=True,
            env=full_env,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"CDO failed: {(exc.stderr or '').strip()}") from exc


def _tag_lonlat(ds: xr.Dataset, lat: str, lon: str) -> xr.Dataset:
    """Return a copy whose lat/lon coordinates carry the CF attributes CDO needs.

    Without ``units`` and ``standard_name`` CDO aborts with "Unsupported
    generic coordinates", which is what plain xarray objects trigger.
    """
    for name, std, units, axis in (
        (lat, "latitude", "degrees_north", "Y"),
        (lon, "longitude", "degrees_east", "X"),
    ):
        if name not in ds.coords:
            raise ValueError(f"Input must have a {name!r} coordinate.")
        if ds[name].ndim != 1:
            raise ValueError(f"{name!r} must be one-dimensional for CDO remapping.")
    ds = ds.copy()
    for name, std, units, axis in (
        (lat, "latitude", "degrees_north", "Y"),
        (lon, "longitude", "degrees_east", "X"),
    ):
        ds[name].attrs.update(standard_name=std, units=units, axis=axis)
    return ds


def remap(
    obj: Path | str | xr.DataArray | xr.Dataset,
    outfile: Path | str | None = None,
    *,
    method: Method = "remapbil",
    resolution: float = 0.25,
    bbox: tuple[float, float, float, float] = (0, -90, 360, 90),
    extrapolate: bool = False,
    lat: str = "lat",
    lon: str = "lon",
    compression: str = "zip",
    datatype: str = "F32",
    n_threads: int | None = None,
) -> Path | xr.DataArray | xr.Dataset:
    """Remap onto a regular lon-lat grid.

    Parameters
    ----------
    obj : path, DataArray or Dataset
    outfile : output path. If given, the path is returned. If omitted, the
        result is returned as an xarray object (DataArray if ``obj`` was one).
    method : one of remapbil, remapbic, remapnn, remapdis, remapcon, remaplaf.
    resolution : target spacing in degrees; bbox spans must be multiples of it.
    bbox : (lon_min, lat_min, lon_max, lat_max) in degrees. A 360 degree
        longitude span omits the duplicate end meridian.
    extrapolate : set REMAP_EXTRAPOLATE=on (not valid for remapdis, remapcon).
    lat, lon : coordinate names of an xarray input. ``latitude``/``longitude``
        are used when the default names are absent. Ignored for file input.
    compression : CDO ``-z`` value.
    datatype : CDO ``-b`` value.
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {list(_METHODS)}, got {method!r}.")
    if extrapolate and method in _NO_EXTRAPOLATE:
        raise ValueError(f"extrapolate is not supported for {method!r}.")
    lon0, lat0, lon1, lat1 = bbox
    if lon1 <= lon0 or lat1 <= lat0 or resolution <= 0:
        raise ValueError(f"Invalid bbox {bbox} or resolution {resolution}.")

    def n_steps(span: float) -> int:
        n = round(span / resolution)
        if not math.isclose(n * resolution, span, abs_tol=1e-9):
            raise ValueError(
                f"Span {span} is not a multiple of resolution {resolution}."
            )
        return n

    nx = n_steps(lon1 - lon0) + (
        0 if math.isclose(lon1 - lon0, 360.0, abs_tol=1e-9) else 1
    )
    ny = n_steps(lat1 - lat0) + 1
    grid = (
        f"gridtype = lonlat\nxsize = {nx}\nysize = {ny}\n"
        f"xfirst = {lon0}\nxinc = {resolution}\nyfirst = {lat0}\nyinc = {resolution}\n"
    )

    with tempfile.TemporaryDirectory(prefix="cdo_py_") as tmp:
        tmp = Path(tmp)
        grid_file = tmp / "grid.txt"
        grid_file.write_text(grid)

        da_name = None
        source_name = getattr(obj, "name", None)
        if isinstance(obj, (xr.DataArray, xr.Dataset)):
            infile = tmp / "in.nc"
            if isinstance(obj, xr.DataArray):
                da_name = obj.name if obj.name is not None else "var"
                obj = obj.to_dataset(name=da_name)
            names = set(obj.coords)
            lat = next((n for n in (lat, *_LAT_NAMES) if n in names), lat)
            lon = next((n for n in (lon, *_LON_NAMES) if n in names), lon)
            _tag_lonlat(obj, lat, lon).to_netcdf(infile)
        else:
            infile = Path(obj)
            if not infile.exists():
                raise FileNotFoundError(f"Input file does not exist: {infile}")

        out = Path(outfile) if outfile is not None else tmp / "out.nc"
        out.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                "-b",
                datatype,
                "-z",
                compression,
                f"{method},{grid_file}",
                str(infile),
                str(out),
            ],
            n_threads=n_threads,
            env={"REMAP_EXTRAPOLATE": "on"} if extrapolate else None,
        )
        if outfile is not None:
            return out
        with xr.open_dataset(out) as ds:
            ds = ds.load()  # read into memory before the temp directory is removed
        if lat != "lat" or lon != "lon":
            ds = ds.rename({k: v for k, v in (("lat", lat), ("lon", lon)) if k in ds})
        if da_name is None:
            return ds
        da = ds[da_name]
        return da.rename(None) if source_name is None else da


def mergetime(
    infiles: list[Path | str],
    outfile: Path | str,
    *,
    compression: str = "zip_4",
    datatype: str = "F32",
    n_threads: int | None = None,
) -> Path:
    """Concatenate NetCDF files along time (CDO orders records by time)."""
    paths = sorted(Path(f).resolve() for f in infiles)
    if not paths:
        raise ValueError("No input files provided.")
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Input file does not exist: {missing[0]}")
    out = Path(outfile).resolve()
    if out in paths:
        raise ValueError(f"outfile must not be one of the inputs: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    run(
        ["-b", datatype, "-z", compression, "-mergetime", *map(str, paths), str(out)],
        n_threads=n_threads,
    )
    return out


def _fixed(method: Method):
    def wrapper(obj, outfile=None, **kwargs):
        return remap(obj, outfile, method=method, **kwargs)

    wrapper.__name__ = method
    wrapper.__qualname__ = method
    wrapper.__doc__ = f"``remap`` with ``method={method!r}``. See :func:`remap`."
    return wrapper


remapbil = _fixed("remapbil")
remapbic = _fixed("remapbic")
remapnn = _fixed("remapnn")
remapdis = _fixed("remapdis")
remapcon = _fixed("remapcon")
remaplaf = _fixed("remaplaf")
