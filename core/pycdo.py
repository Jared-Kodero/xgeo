"""Minimal Python wrapper for the Climate Data Operators (CDO).

Functions
---------
remap      Regrid a NetCDF file or xarray object onto a regular lon-lat grid.
mergetime  Concatenate NetCDF files along time.
run        Run any CDO command.

Examples
--------
>>> import cdo
>>> cdo.remap("in.nc", "out.nc", method="remapcon", resolution=0.5)
>>> da_025 = cdo.remap(da, resolution=0.25)  # xarray in, xarray out
>>> cdo.mergetime(["a.nc", "b.nc"], "merged.nc")
>>> cdo.run(["-timmean", "in.nc", "out.nc"])

CDO reference: https://code.mpimet.mpg.de/projects/cdo/embedded/cdo.pdf
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

import xarray as xr

__all__ = ["mergetime", "remap", "run"]


def run(
    cmd: list[str], n_threads: int | None = None, env: dict[str, str] | None = None
):
    """Run a CDO command, e.g. ``run(["-timmean", "in.nc", "out.nc"])``.

    Options ``-s -O -P <n>`` are added. A leading ``"cdo"`` in ``cmd`` is ignored.
    ``env`` is applied to the subprocess only. Raises RuntimeError with the
    captured stderr if CDO is missing or fails.
    """
    if not shutil.which("cdo"):
        raise RuntimeError("CDO executable not found in PATH.")
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
        raise RuntimeError(f"CDO failed: {exc.stderr.strip()}") from exc


def remap(
    obj: Path | str | xr.DataArray | xr.Dataset,
    outfile: Path | str | None = None,
    *,
    method: Literal[
        "remapbil", "remapbic", "remapnn", "remapdis", "remapcon", "remaplaf"
    ] = "remapbil",
    resolution: float = 0.25,
    bbox: tuple[float, float, float, float] = (0, -90, 360, 90),
    extrapolate: bool = False,
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
    compression : CDO ``-z`` value.
    datatype : CDO ``-b`` value.
    """
    if method not in {
        "remapbil",
        "remapbic",
        "remapnn",
        "remapdis",
        "remapcon",
        "remaplaf",
    }:
        raise ValueError(
            f"method must be one of {sorted({'remapbil', 'remapbic', 'remapnn', 'remapdis', 'remapcon', 'remaplaf'})}, got {method!r}."
        )
    if extrapolate and method not in {"remapbil", "remapbic", "remapnn", "remaplaf"}:
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
        if isinstance(obj, (xr.DataArray, xr.Dataset)):
            infile = tmp / "in.nc"
            if isinstance(obj, xr.DataArray):
                da_name = obj.name or "var"
                obj = obj.to_dataset(name=da_name)
            obj.to_netcdf(infile)
        else:
            infile = Path(obj)
            if not infile.exists():
                raise FileNotFoundError(f"Input file does not exist: {infile}")

        out = Path(outfile) if outfile is not None else tmp / "out.nc"
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
        return ds[da_name] if da_name else ds


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
    run(
        ["-b", datatype, "-z", compression, "-mergetime", *map(str, paths), str(out)],
        n_threads=n_threads,
    )
    return out
