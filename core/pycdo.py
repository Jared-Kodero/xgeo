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

import xarray as xr

METHODS = ("remapbil", "remapbic", "remapnn", "remapdis", "remapcon", "remaplaf")


__all__ = ["mergetime", "remap", "run"]


def run(
    cmd: list[str | os.PathLike[str]] | str,
    n_threads: int | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a CDO command, e.g. ``run(["-timmean", "in.nc", "out.nc"])``.

    ``cmd`` is a list (items may be paths) or a string split with
    :func:`shlex.split`. The options ``-s -O -P <n>`` are prepended and a
    leading ``"cdo"`` in ``cmd`` is ignored. ``env`` applies to the subprocess
    only. Raises RuntimeError with the captured stderr if CDO is missing or
    fails.
    """
    if not shutil.which("cdo"):
        raise RuntimeError("CDO executable not found in PATH.")
    args = shlex.split(cmd) if isinstance(cmd, str) else [os.fspath(c) for c in cmd]
    if args and args[0] == "cdo":
        args = args[1:]
    threads = n_threads or min(8, os.cpu_count() or 1)
    full_env = {
        "CDO_VERSION_INFO": "false",
        "CDO_HISTORY_INFO": "false",
        **os.environ,
        **(env or {}),
    }
    try:
        return subprocess.run(
            ["cdo", "-s", "-O", "-P", str(threads), *args],
            check=True,
            text=True,
            capture_output=True,
            env=full_env,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"CDO failed: {(exc.stderr or '').strip()}") from exc


def remap(
    obj: Path | str | xr.DataArray | xr.Dataset,
    outfile: Path | str | None = None,
    *,
    method: str = "remapbil",
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
        result is returned as an xarray object (a DataArray if ``obj`` was one).
    method : one of remapbil, remapbic, remapnn, remapdis, remapcon, remaplaf.
    resolution : target spacing in degrees; bbox spans must be multiples of it.
    bbox : (lon_min, lat_min, lon_max, lat_max) in degrees. A 360 degree
        longitude span omits the duplicate end meridian.
    extrapolate : set REMAP_EXTRAPOLATE=on (invalid for remapdis and remapcon).
    lat, lon : coordinate names of an xarray input. ``latitude`` and
        ``longitude`` are used when the defaults are absent. Ignored for file
        input.
    compression : CDO ``-z`` value.
    datatype : CDO ``-b`` value.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {list(METHODS)}, got {method!r}.")
    if extrapolate and method in ("remapdis", "remapcon"):
        raise ValueError(f"extrapolate is not supported for {method!r}.")

    lon0, lat0, lon1, lat1 = bbox
    if lon1 <= lon0 or lat1 <= lat0 or resolution <= 0:
        raise ValueError(f"Invalid bbox {bbox} or resolution {resolution}.")

    # Number of grid points per axis; bbox spans must be whole multiples of resolution.
    counts = []
    for span in (lon1 - lon0, lat1 - lat0):
        n = round(span / resolution)
        if not math.isclose(n * resolution, span, abs_tol=1e-9):
            raise ValueError(
                f"Span {span} is not a multiple of resolution {resolution}."
            )
        counts.append(n)
    is_global = math.isclose(lon1 - lon0, 360.0, abs_tol=1e-9)
    nx = counts[0] + (0 if is_global else 1)
    ny = counts[1] + 1
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
                da_name = "var" if obj.name is None else obj.name
                obj = obj.to_dataset(name=da_name)
            lat = next((n for n in (lat, "lat", "latitude") if n in obj.coords), lat)
            lon = next((n for n in (lon, "lon", "longitude") if n in obj.coords), lon)

            # CDO aborts on coordinates without CF units and standard_name.
            ds = obj.copy()
            for name, std, units, axis in (
                (lat, "latitude", "degrees_north", "Y"),
                (lon, "longitude", "degrees_east", "X"),
            ):
                if name not in ds.coords:
                    raise ValueError(f"Input must have a {name!r} coordinate.")
                if ds[name].ndim != 1:
                    raise ValueError(
                        f"{name!r} must be one-dimensional for CDO remapping."
                    )
                ds[name].attrs.update(standard_name=std, units=units, axis=axis)
            ds.to_netcdf(infile)
        else:
            infile = Path(obj)
            if not infile.exists():
                raise FileNotFoundError(f"Input file does not exist: {infile}")

        out = Path(outfile) if outfile is not None else tmp / "out.nc"
        out.parent.mkdir(parents=True, exist_ok=True)
        run(
            ["-b", datatype, "-z", compression, f"{method},{grid_file}", infile, out],
            n_threads=n_threads,
            env={"REMAP_EXTRAPOLATE": "on"} if extrapolate else None,
        )
        if outfile is not None:
            return out

        with xr.open_dataset(out) as result:
            result = (
                result.load()
            )  # read into memory before the temp directory is removed

    # CDO writes "lat" and "lon"; restore the caller's coordinate names.
    result = result.rename(
        {k: v for k, v in (("lat", lat), ("lon", lon)) if k in result and k != v}
    )
    if da_name is None:
        return result
    da = result[da_name]
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
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Input file does not exist: {p}")
    out = Path(outfile).resolve()
    if out in paths:
        raise ValueError(f"outfile must not be one of the inputs: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    run(
        ["-b", datatype, "-z", compression, "-mergetime", *paths, out],
        n_threads=n_threads,
    )
    return out


def remapbil(
    obj: Path | str | xr.DataArray | xr.Dataset,
    outfile: Path | str | None = None,
    *,
    resolution: float = 0.25,
    bbox: tuple[float, float, float, float] = (0, -90, 360, 90),
    extrapolate: bool = False,
    lat: str = "lat",
    lon: str = "lon",
    compression: str = "zip",
    datatype: str = "F32",
    n_threads: int | None = None,
) -> Path | xr.DataArray | xr.Dataset:
    """Bilinear interpolation. See :func:`remap`."""
    return remap(
        obj,
        outfile,
        method="remapbil",
        resolution=resolution,
        bbox=bbox,
        extrapolate=extrapolate,
        lat=lat,
        lon=lon,
        compression=compression,
        datatype=datatype,
        n_threads=n_threads,
    )


def remapbic(
    obj: Path | str | xr.DataArray | xr.Dataset,
    outfile: Path | str | None = None,
    *,
    resolution: float = 0.25,
    bbox: tuple[float, float, float, float] = (0, -90, 360, 90),
    extrapolate: bool = False,
    lat: str = "lat",
    lon: str = "lon",
    compression: str = "zip",
    datatype: str = "F32",
    n_threads: int | None = None,
) -> Path | xr.DataArray | xr.Dataset:
    """Bicubic interpolation. See :func:`remap`."""
    return remap(
        obj,
        outfile,
        method="remapbic",
        resolution=resolution,
        bbox=bbox,
        extrapolate=extrapolate,
        lat=lat,
        lon=lon,
        compression=compression,
        datatype=datatype,
        n_threads=n_threads,
    )


def remapnn(
    obj: Path | str | xr.DataArray | xr.Dataset,
    outfile: Path | str | None = None,
    *,
    resolution: float = 0.25,
    bbox: tuple[float, float, float, float] = (0, -90, 360, 90),
    extrapolate: bool = False,
    lat: str = "lat",
    lon: str = "lon",
    compression: str = "zip",
    datatype: str = "F32",
    n_threads: int | None = None,
) -> Path | xr.DataArray | xr.Dataset:
    """Nearest neighbour. See :func:`remap`."""
    return remap(
        obj,
        outfile,
        method="remapnn",
        resolution=resolution,
        bbox=bbox,
        extrapolate=extrapolate,
        lat=lat,
        lon=lon,
        compression=compression,
        datatype=datatype,
        n_threads=n_threads,
    )


def remapdis(
    obj: Path | str | xr.DataArray | xr.Dataset,
    outfile: Path | str | None = None,
    *,
    resolution: float = 0.25,
    bbox: tuple[float, float, float, float] = (0, -90, 360, 90),
    extrapolate: bool = False,
    lat: str = "lat",
    lon: str = "lon",
    compression: str = "zip",
    datatype: str = "F32",
    n_threads: int | None = None,
) -> Path | xr.DataArray | xr.Dataset:
    """Distance-weighted average. See :func:`remap`."""
    return remap(
        obj,
        outfile,
        method="remapdis",
        resolution=resolution,
        bbox=bbox,
        extrapolate=extrapolate,
        lat=lat,
        lon=lon,
        compression=compression,
        datatype=datatype,
        n_threads=n_threads,
    )


def remapcon(
    obj: Path | str | xr.DataArray | xr.Dataset,
    outfile: Path | str | None = None,
    *,
    resolution: float = 0.25,
    bbox: tuple[float, float, float, float] = (0, -90, 360, 90),
    extrapolate: bool = False,
    lat: str = "lat",
    lon: str = "lon",
    compression: str = "zip",
    datatype: str = "F32",
    n_threads: int | None = None,
) -> Path | xr.DataArray | xr.Dataset:
    """First-order conservative. See :func:`remap`."""
    return remap(
        obj,
        outfile,
        method="remapcon",
        resolution=resolution,
        bbox=bbox,
        extrapolate=extrapolate,
        lat=lat,
        lon=lon,
        compression=compression,
        datatype=datatype,
        n_threads=n_threads,
    )


def remaplaf(
    obj: Path | str | xr.DataArray | xr.Dataset,
    outfile: Path | str | None = None,
    *,
    resolution: float = 0.25,
    bbox: tuple[float, float, float, float] = (0, -90, 360, 90),
    extrapolate: bool = False,
    lat: str = "lat",
    lon: str = "lon",
    compression: str = "zip",
    datatype: str = "F32",
    n_threads: int | None = None,
) -> Path | xr.DataArray | xr.Dataset:
    """Largest area fraction. See :func:`remap`."""
    return remap(
        obj,
        outfile,
        method="remaplaf",
        resolution=resolution,
        bbox=bbox,
        extrapolate=extrapolate,
        lat=lat,
        lon=lon,
        compression=compression,
        datatype=datatype,
        n_threads=n_threads,
    )
