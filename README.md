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

Colorbar tick labels are formatted with a math-text `ScalarFormatter` by default. Pass `cbar_scalar_formatter=False` to `geoplot` or `animate` (accessor or functional form) to keep Matplotlib's default formatter:

```python
ds["topo"].xgeo.geoplot(cmap="terrain", cbar_scalar_formatter=False)
```

Enable hillshade to draw grayscale relief beneath a semitransparent scalar field:

```python
ds["topo"].xgeo.geoplot(
    hillshade=True, hillshade_vert_exag=10, hillshade_alpha=1.0,
    cmap="terrain", robust=True,
)
ds["topo"].xgeo.animate(
    dim="time", hillshade=True, hillshade_vert_exag=10, hillshade_alpha=1.0,
    cmap="terrain", robust=True,
)
```

These arguments also work with `xgeo.plot.geoplot` and `xgeo.plot.animate`,
including facets. Hillshade defaults to `False`, and vertical exaggeration to
`5.0`. Shading uses Matplotlib's `LightSource` with azimuth 315 degrees, altitude
45 degrees, and unit pixel spacing. The original field retains its colormap and
colorbar and uses `alpha=0.4` unless explicitly supplied. `hillshade_alpha`
controls the grayscale layer's opacity and defaults to `1.0`. No additional
dependency is required. This provides visual relief without converting
horizontal grid spacing to physical distances.

The computation is also available separately for a two-dimensional `(y, x)` field:

```python
from xgeo.viz.plot_utils import compute_hillshade_matplotlib

shaded = compute_hillshade_matplotlib(ds["topo"], z_factor=10)
```

## Geospatial and statistical tools

```python
regridded = ds.xgeo.regrid(target_grid, method="bilinear")
coarse = ds.xgeo.regrid(1.0, method="conservative")
curvilinear = ds_2d.xgeo.regrid(0.25, lat="lat", lon="lon", lat_b="lat_b", lon_b="lon_b")
wrapped = ds.xgeo.wrap_lon(convention="-180/180")
with_lst = ds.xgeo.add_local_solar_time()
trend = ds["t2m"].xgeo.trends(dim="time")
correlation = ds["t2m"].xgeo.corr(other, dim="time")
```

The functional APIs include `xgeo.regrid`, `xgeo.mask`, `xgeo.sel_transect`, `xgeo.wrap_lon`, `xgeo.add_local_solar_time`, and `xgeo.stats`. `regrid` takes a target grid as a Dataset or DataArray with one-dimensional `lat` and `lon` coordinates, or as a resolution in degrees spanning the source extent. Coordinate names are set with `lat` and `lon`. Passing both `lat_b` and `lon_b` marks the source as a two-dimensional (curvilinear) grid; the bounds may be cell corners `(ny + 1, nx + 1)` or CF bounds `(ny, nx, 4)`. The output is always a one-dimensional latitude-longitude grid with the bounds dropped. Regridding imports `xesmf` on first use. See [`core/accessors.py`](core/accessors.py), [`core/xr_utils.py`](core/xr_utils.py), and [`core/stats.py`](core/stats.py) for signatures.

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
