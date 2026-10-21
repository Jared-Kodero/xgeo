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

Frequently used options (`method`, `cmap`, `vmin`, `vmax`, `levels`, `projection`, `colorbar`, `colorbar_label`, `colorbar_orientation` and others) are plain keywords. Rarely changed settings are grouped into small frozen dataclasses, available from the `xg.plot_options` namespace (`xg.plot_options.Hillshade`, `xg.plot_options.MapFeatures` and so on), and accepted by `geoplot` and `animate` in accessor and functional form. Grouped parameters take an instance of the class, `None`, or a bool where a default group exists (`hillshade=True`, `map_ticks=True`, `clabel=True`). Plain dictionaries are not accepted and raise `TypeError`; construct the class with keyword arguments instead. All fields are keyword-only, named and typed, the classes take no open-ended keyword arguments, and an unknown field raises `TypeError`.

| Parameter | Class | Fields |
| --- | --- | --- |
| `colorbar_style` | `ColorbarStyle` | `drawedges`, `minimal_ticks`, `scalar_formatter`, `powerlimits`, `ticks`, `tick_labels`; advanced: `thickness`, `pad`, `pad_with_labels`, `facet_length`, `minimal_ticks_threshold` |
| `features` | `MapFeatures` | `coastlines`, `borders`, `states`, `ocean`, `land`, `lakes`, `rivers`, `global_extent`, `extent`, `grid_bounds`; advanced: `resolution`, `coastline_color`, `coastline_linewidth`, `states_alpha`, `borders_alpha`, `boundary_linestyle`, `boundary_color`, `boundary_linewidth`, `boundary_zorder`, `water_color`, `water_zorder`, `land_color`, `ocean_color`, `background_zorder`, `grid_bounds_linewidth`, `grid_bounds_color`, `grid_bounds_zorder` |
| `map_ticks` | `MapTicks` | `xbins`, `ybins`; advanced: `direction`, `minor_ticks`, `degree_symbol`, `direction_label`, `top_labels`, `right_labels`, `gridlines` |
| `clabel` | `ContourLabels` | `fmt`, `fontsize`, `inline`, `colors`, `levels`, `inline_spacing`, `rightside_up`, `use_clabeltext`, `manual`, `zorder` |
| `hillshade` | `Hillshade` | `vert_exag`, `azimuth`, `altitude`, `alpha`, `field_alpha`, `cmap`, `spacing`, `method`, `levels`, `vmin`, `vmax`, `fraction`, `zorder_offset`, `rasterized` |
| `significance` | `Significance` | `pvalues`, `level`, `color`, `alpha`, `marker`, `edgecolors`, `subsample`, `size`, `zorder` (`geoplot` only) |
| `vectors` | `VectorField` | `u`, `v`, `subsample`, `add_key`, `key_magnitude`, `key_units`, `powerlimits`, `scale`, `scale_units`, `units`, `color`, `width`, `headwidth`, `headlength`, `headaxislength`, `minshaft`, `minlength`, `pivot`, `alpha`, `zorder`, `angles`, `key_fontsize`, `key_gap` |

The fields marked advanced, and the `None`-valued fields of `VectorField`, keep the plotting defaults when left unset, so existing calls are unchanged.

```python
ds["topo"].xgeo.geoplot(
    cmap="terrain",
    features=xg.plot_options.MapFeatures(states=False, lakes=True),
    colorbar_style=xg.plot_options.ColorbarStyle(scalar_formatter=False),
    map_ticks=xg.plot_options.MapTicks(xbins=4),
)
t2m.xgeo.geoplot(
    method="contour",
    clabel=xg.plot_options.ContourLabels(fmt="%.1f", fontsize=7),
    vectors=xg.plot_options.VectorField(u=ds["u10"], v=ds["v10"], subsample=(4, 4), color="k"),
)
```

Enable hillshade to draw grayscale relief beneath a semitransparent scalar field:

```python
ds["topo"].xgeo.geoplot(hillshade=True, cmap="terrain", robust=True)
ds["topo"].xgeo.geoplot(
    hillshade=xg.plot_options.Hillshade(azimuth=270, altitude=35, vert_exag=2), cmap="terrain"
)
ds["topo"].xgeo.animate(dim="time", hillshade=True, cmap="terrain", robust=True)
```

`Hillshade` defaults to azimuth 315 and altitude 45 degrees, vertical exaggeration 1 and `spacing="auto"`. The surface gradient needs the horizontal cell size in the unit of the elevation. With `"auto"`, a degree grid is converted to metres from the mean cell step and the mean latitude, and projected coordinates in metres or kilometres are used as metres, so the elevation is assumed to be in metres. Grids with other or unknown units fall back to `"pixel"` (unit spacing), which makes the shading depend on grid resolution; pass `spacing=(dx, dy)` to set it explicitly. `alpha` is the opacity of the grayscale layer (default 1) and `field_alpha` the opacity given to the field (default 0.65, `None` to leave it unchanged, an explicit `alpha=` argument takes precedence). The field keeps its colormap and colorbar. Shading is computed per facet and animation frame with Matplotlib's `LightSource`.

The relief layer has its own plotting method, independent of the field: `Hillshade(method="pcolormesh")` (the default, `"default"`, gives the same result), `"imshow"` or `"contourf"`, with `levels` setting the bands of `"contourf"`. `vmin` and `vmax` (default 0 and 1) set the illumination range mapped to `cmap`, `fraction` the contrast of the shading, `zorder_offset` how far below the field the relief is drawn (default 0.1) and `rasterized` overrides the field's setting for the relief. On a map whose projection is centred away from 0 degrees longitude, `"imshow"` is placed correctly only when `MapFeatures(extent=...)` is set, which also holds for `method="imshow"` of the field itself.

The computation is also available separately for a two-dimensional `(y, x)` field:

```python
from xgeo.viz.primitives import compute_hillshade, hillshade_spacing

dx, dy = hillshade_spacing(ds["topo"], x="lon", y="lat")
shaded = compute_hillshade(ds["topo"], z_factor=1, dx=dx, dy=dy)
```

## Geospatial and statistical tools

```python
regridded = ds.xgeo.regrid(target_grid, method="bilinear")
coarse = ds.xgeo.regrid(1.0, method="conservative")
curvilinear = ds_2d.xgeo.regrid(
    0.25, lat="lat", lon="lon", lat_b="lat_b", lon_b="lon_b"
)
wrapped = ds.xgeo.wrap_lon(convention="-180/180")
with_lst = ds.xgeo.add_local_solar_time()
trend = ds["t2m"].xgeo.trends(dim="time")
correlation = ds["t2m"].xgeo.corr(other, dim="time")
```

The functional APIs include `xgeo.regrid`, `xgeo.mask`, `xgeo.sel_transect`, `xgeo.wrap_lon`, `xgeo.add_local_solar_time`, and `xgeo.stats`. `regrid` takes a target grid as a Dataset or DataArray with one-dimensional `lat` and `lon` coordinates, or as a resolution in degrees spanning the source extent. Coordinate names are set with `lat` and `lon`. Passing both `lat_b` and `lon_b` marks the source as a two-dimensional (curvilinear) grid; the bounds may be cell corners `(ny + 1, nx + 1)` or CF bounds `(ny, nx, 4)`. The output is always a one-dimensional latitude-longitude grid with the bounds dropped. Regridding imports `xesmf` on first use. See [`core/accessors.py`](core/accessors.py), [`core/xr_utils.py`](core/xr_utils.py), and [`core/stats.py`](core/stats.py) for signatures.

Serial NetCDF I/O uses Xarray's native interface, such as `ds.to_netcdf("output.nc")`. The former MPI-backed `.xgeo.append()` and `.xgeo.to_netcdf()` methods and top-level MPI aliases are removed. Use [`xrmpi`](../xrmpi/README.md) directly for distributed arrays and collective NetCDF I/O; xgeo does not import or install it.

## Plotting and colormaps

Plotting is split into three modules under [`viz/`](viz/):

* [`viz/plot.py`](viz/plot.py) holds the stateful classes and entry points: `GeoPlot` (figure, axes and every primitive it creates), `FacetedPlot` (facet layout), `Adder` (overlays, available as `GeoPlot.add`), `Animate` and the functions `geoplot`, `animate`, `theme`, `colorbar` and `create_figure`.
* [`viz/primitives.py`](viz/primitives.py) holds the stateless drawing functions (`plot_field`, `plot_quiver`, `plot_significance`) and the input and colormap helpers. `plot_field(method, data, fig, ax, ...)` is the one function behind the base layer and every scalar overlay, for `"default"`, `"pcolormesh"`, `"contourf"`, `"contour"`, `"imshow"` and `"scatter"`. It sends `levels`, `extend`, `robust` and `rasterized` only to the methods that use them and forwards every other keyword to xarray or Matplotlib. `"scatter"` also accepts an `(x, y)` tuple of arrays.
* [`viz/options.py`](viz/options.py) holds the option dataclasses described above.

Overlays are added to an existing plot with `GeoPlot.add` and can be chained. Each call draws on every panel of a faceted plot, records its primitives in `GeoPlot.layers` and returns the plot:

```python
plot = t2m.xgeo.geoplot(method="pcolormesh", features=xg.plot_options.MapFeatures(states=False))
plot.add.contour(t2m, levels=5, clabel=xg.plot_options.ContourLabels(fmt="%.0f")).add.grid_boundary()
```

The scalar overlays are `add.plot`, `add.pcolormesh`, `add.contourf`, `add.contour`, `add.imshow` and `add.scatter`, which accepts an `xarray.DataArray` or an `(x, y)` tuple of arrays. `add.quiver`, `add.significance`, `add.colorbar` and `add.grid_boundary` complete the set.

[`viz/cmaps.py`](viz/cmaps.py) provides palettes from the local IPCC tables, Matplotlib, and cmocean.

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
