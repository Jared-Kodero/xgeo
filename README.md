# xgeo

Climate-data analysis, geospatial processing, visualization, and memory-mapped storage built around [Xarray](https://xarray.dev/).

`xgeo` provides the `.xgeo` accessor, Cartopy maps, geospatial transformations, statistical tools, scientific colormaps, preprocessing, and CDO integration. MPI-distributed arrays, the MPI context, and parallel NetCDF output belong to the separate [`xrmpi`](../xrmpi/README.md) package.

## Installation

Python **3.14 or newer** is required, matching `pyproject.toml`.

```bash
python -m pip install xgeo
python -m pip install "xgeo[regrid]"  # optional xesmf support
```

For users who clone the package, run from the `xgeo` clone root:

```bash
conda env create -n xgeo -f env/env.yaml
conda activate xgeo
python -m pip install --no-deps -e .
```

The package's own [`env/README.md`](env/README.md) documents creation, updates, optional features, notebook kernels, and installation into another environment. Its specification is [`env/env.yaml`](env/env.yaml).

CDO workflows require `cdo` and `nco` on `PATH`; animations require `ffmpeg`. The Conda specification includes these executables and `xesmf`.

## Quick start

```python
import xarray as xr

import xgeo as xg


ds = xr.open_dataset("climate.nc")
t2m = ds["t2m"]
plot = t2m.xgeo.geoplot(
    method="contourf",
    cmap=xg.cmaps.temp_div(),
    levels=21,
    gridlines=True,
)
```

The functional plotting entry point is `xgeo.plot.geoplot(t2m, ...)`.

## Geospatial and statistical tools

```python
regridded = ds.xgeo.regrid(target_grid, method="bilinear")
wrapped = ds.xgeo.wrap_lon(convention="-180/180")
with_lst = ds.xgeo.add_local_solar_time()
trend = ds["t2m"].xgeo.trends(dim="time")
correlation = ds["t2m"].xgeo.corr(other, dim="time")
```

The functional APIs include `xgeo.regrid`, `xgeo.mask`, `xgeo.sel_transect`, `xgeo.wrap_lon`, `xgeo.add_local_solar_time`, and `xgeo.stats`. Regridding imports `xesmf` on first use. See [`core/accessors.py`](core/accessors.py), [`core/xr_utils.py`](core/xr_utils.py), and [`core/stats.py`](core/stats.py) for signatures.

Serial NetCDF I/O uses Xarray's native interface, such as `ds.to_netcdf("output.nc")`. The former MPI-backed `.xgeo.append()` and `.xgeo.to_netcdf()` methods and top-level MPI aliases are removed. Use [`xrmpi`](../xrmpi/README.md) directly for distributed arrays and collective NetCDF I/O; xgeo does not import or install it.

## Plotting and colormaps

[`viz/plot.py`](viz/plot.py) implements geographic plotting and overlays. [`viz/cmaps.py`](viz/cmaps.py) provides palettes from the local IPCC tables, Matplotlib, and cmocean.

```python
from xgeo import cmaps

cmap = cmaps.temp_div()
```

## Storage

XNpy stores save NumPy, pandas, and Xarray objects as `.npy` payloads with a JSON manifest. The matching readers support memory mapping.

```python
import xgeo as xg

xg.to_xnpy(ds, "run_001")
restored = xg.open_xnpy_dataset("run_001")
```

See [`core/xnpy.py`](core/xnpy.py) for storage options.

## Package overview

| Namespace | Purpose |
| --- | --- |
| `xgeo` | Geospatial helpers, preprocessing, plotting entry points, and storage |
| `xgeo.plot` | Cartopy geographic plotting |
| `xgeo.stats` | Trends, correlations, and significance testing |
| `xgeo.cmaps` | Scientific colormaps |
| `xgeo.cdo.pycdo` | CDO/NCO command-line wrapper |
| `xrmpi` | Separate package for distributed arrays and parallel I/O |

## Links

* [Environment specification](env/env.yaml)
* [Xarray accessors](core/accessors.py)
* [Plotting source](viz/plot.py)
* [MPI package](../xrmpi/README.md)
* [License](LICENSE)

Distributed under the [MIT License](LICENSE).
