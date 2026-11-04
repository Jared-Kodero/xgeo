# xgeo

Climate-data analysis, geospatial processing, visualization, and memory-mapped storage built around [Xarray](https://xarray.dev/).

`xgeo` provides the `.xgeo` accessor, Cartopy maps, geospatial transformations, statistical tools, scientific colormaps, and preprocessing. MPI-distributed arrays, the MPI context, and parallel NetCDF output belong to the separate [`xrmpi`](../xrmpi/README.md) package.

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

Animations require `ffmpeg` on `PATH`. The Conda specification includes `ffmpeg` and `xesmf`.

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
| `features` | `MapFeatures` | `coastlines`, `borders`, `states`, `ocean`, `land`, `lakes`, `rivers`, `global_extent`, `extent`, `grid_bounds`; advanced: `resolution`, `coastline_color`, `coastline_linewidth`, `coastline_alpha`, `coastline_zorder`, `states_alpha`, `borders_alpha`, `boundary_linestyle`, `boundary_color`, `boundary_linewidth`, `boundary_zorder`, `states_color`, `states_linestyle`, `states_linewidth`, `states_zorder`, `borders_color`, `borders_linestyle`, `borders_linewidth`, `borders_zorder`, `lakes_linewidth`, `rivers_linewidth`, `water_color`, `water_zorder`, `land_color`, `ocean_color`, `background_zorder`, `grid_bounds_linewidth`, `grid_bounds_color`, `grid_bounds_zorder` |
| `map_ticks` | `MapTicks` | `xbins`, `ybins`; advanced: `direction`, `minor_ticks`, `degree_symbol`, `direction_label`, `top_labels`, `right_labels`, `gridlines`, `gridline_color`, `gridline_linestyle`, `gridline_linewidth`, `gridline_alpha` |
| `clabel` | `ContourLabels` | `fmt`, `fontsize`, `inline`, `colors`, `levels`, `inline_spacing`, `rightside_up`, `use_clabeltext`, `manual`, `zorder` |
| `hillshade` | `Hillshade` | `vert_exag`, `azimuth`, `altitude`, `alpha`, `field_alpha`, `cmap`, `spacing`, `method`, `levels`, `vmin`, `vmax`, `fraction`, `zorder_offset`, `rasterized` |
| `significance` | `xg.plot.Significance` | `pvalues`, `level`, `color`, `alpha`, `marker`, `edgecolors`, `subsample`, `size`, `zorder`, `hatch` |
| `vectors` | `xg.plot.VectorField` | `u`, `v`, `subsample`, `add_key`, `key_magnitude`, `key_units`, `powerlimits`, `scale`, `scale_units`, `units`, `color`, `width`, `headwidth`, `headlength`, `headaxislength`, `minshaft`, `minlength`, `pivot`, `alpha`, `zorder`, `angles`, `key_fontsize`, `key_gap`, `cmap`, `vmin`, `vmax`, `key_color`, `key_align` |

`VectorField` and `Significance` carry data (the vector components, the p-values) rather than options, so they are `xg.plot.VectorField` and `xg.plot.Significance`, not part of `xg.plot_options`. The fields marked advanced, and the `None`-valued fields of `VectorField`, keep the plotting defaults when left unset, so existing calls are unchanged.

Fields of `MapFeatures`, `MapTicks` and `ColorbarStyle` default to `None`, meaning "not set". Values are layered, later layers winning: the class defaults, the map theme preset, `set_options`, then the keyword in the plotting call. A field you do not set therefore never overrides the theme, so `MapFeatures(lakes=True)` leaves `borders=False` of the `ferret`, `grads` and `gmt` themes in place.

### Standing options

Style defaults that do not depend on the data are set once with `xg.plot.set_options`, globally or for a `with` block, in the manner of `xarray.set_options`. There is no `map_theme` keyword in `geoplot` or `animate`.

```python
xg.plot.set_options(map_theme="ncl")            # until changed again

with xg.plot.set_options(map_theme="gmt", features=xg.plot_options.MapFeatures(lakes=True)):
    xg.plot.geoplot(da)                         # gmt theme, lakes on

xg.plot.get_options()                           # current values
```

| Option | Value | Effect |
| --- | --- | --- |
| `map_theme` | `"default"`, `"ncl"`, `"ferret"`, `"grads"`, `"gmt"` or a `MapTheme` | Style preset. A `MapTheme` is a custom preset, for example `dataclasses.replace(xg.plot_options.MapTheme.ncl(), rc={"font.family": "serif"})`. |
| `features`, `colorbar_style`, `map_ticks` | option instance or `None` | Standing defaults applied over the preset and overridden by the plotting call. |
| `rc` | mapping | Matplotlib rc parameters applied only while a plot is built or extended (`plot.add.*`), then restored. They override the preset's `rc`. |

The `ncl` and `gmt` presets request a Helvetica-like font through a `font.family` list (Liberation Sans, Arial, Helvetica, DejaVu Sans). Only installed families are used and nothing is downloaded. A plot keeps the rc parameters it was created with, so a colorbar added later matches the plot even if `set_options` has changed since. `animate` carries the current options into its worker processes.

The `states_*` and `borders_*` fields style the two boundary layers separately and fall back to the shared `boundary_*` value when `None`. `gridline_*` apply when `gridlines` is true. `Significance(hatch="///")` replaces stippling with a hatch drawn over the region where `pvalues < level` (all non-missing `pvalues` when `level` is `None`); `color` and `edgecolors` set the hatch colour and `marker`, `size` and `subsample` are then ignored. `VectorField(cmap=...)` colours the arrows by magnitude on a normalisation shared by all panels and frames (`vmin` defaults to 0, `vmax` to the maximum magnitude), is mutually exclusive with `color`, and gives a mappable for `plot.add.colorbar(plot.quiver)`. `key_color` sets the key arrow and label colour and `key_align` (`"left"`, `"center"` or `"right"`) its horizontal position under the axes.

`animate` accepts `significance=` as well as `vectors=`, with `pvalues` sharing the animation dimension of the data, and `title` as a string or a dictionary whose `label` is the title text and whose other keys are Matplotlib text properties. `GeoPlot.add.colorbar(style=ColorbarStyle(...))` overrides the plot-level `colorbar_style` for one colorbar, including `thickness`, `pad`, `pad_with_labels`, `facet_length` and `minimal_ticks_threshold`. `xg.plot.theme(font_family="serif")` sets the font family of the theme.

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
    vectors=xg.plot.VectorField(u=ds["u10"], v=ds["v10"], subsample=(4, 4), color="k"),
)
plot = t2m.xgeo.geoplot(
    map_ticks=xg.plot_options.MapTicks(gridlines=True, gridline_color="k", gridline_alpha=0.3),
    significance=xg.plot.Significance(pvalues=pval, level=0.05, hatch="///"),
    vectors=xg.plot.VectorField(u=ds["u10"], v=ds["v10"], subsample=(4, 4), cmap="viridis"),
)
plot.add.colorbar(plot.quiver, label="m s-1", style=xg.plot_options.ColorbarStyle(thickness=0.03))
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

The functional APIs live in `xgeo.xr_utils` (`regrid`, `mask`, `fillgaps`, `sel_transect`, `wrap_lon`, `add_cyclic_point`, `add_local_solar_time`) and `xgeo.stats`. `regrid` takes a target grid as a Dataset or DataArray with one-dimensional `lat` and `lon` coordinates, or as a resolution in degrees spanning the source extent. Coordinate names are set with `lat` and `lon`. Passing both `lat_b` and `lon_b` marks the source as a two-dimensional (curvilinear) grid; the bounds may be cell corners `(ny + 1, nx + 1)` or CF bounds `(ny, nx, 4)`. The output is always a one-dimensional latitude-longitude grid with the bounds dropped. Regridding imports `xesmf` on first use. See [`core/accessors.py`](core/accessors.py), [`core/xr_utils.py`](core/xr_utils.py), and [`core/stats.py`](core/stats.py) for signatures.

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

xg.xnpy.save(ds, "run_001")
restored = xg.xnpy.open_dataset("run_001")
```

See [`core/xnpy.py`](core/xnpy.py) for storage options.

## Package overview

| Namespace | Purpose |
| --- | --- |
| `xgeo` | Light shared utilities only (`LockedLogger`, `LockFile`, `RedirectStreams`, `locked_print`, `exclude_key`, `nproc`) |
| `xgeo.plot` | Cartopy geographic plotting |
| `xgeo.stats` | Trends, correlations, and significance testing |
| `xgeo.cmaps` | Scientific colormaps |
| `xgeo.xr_utils` | Regridding, masking, gap filling, transects, longitude wrapping, local solar time |
| `xgeo.xnpy` | Memory-mapped array storage (`save`, `open_dataset`, `open_dataframe`, `load_array`) |
| `xgeo.preprocess` | ERA5, IMERG, CMORPH, and GPCP preprocessing |
| `xgeo.grib_io` | GRIB input and output |
| `xrmpi` | Separate package for distributed arrays and parallel I/O |

## Links

* [Environment specification](env/env.yaml)
* [Xarray accessors](core/accessors.py)
* [Plotting source](viz/plot.py)
* [MPI package](../xrmpi/README.md)
* [License](LICENSE)

Distributed under the [MIT License](LICENSE).
