# xgeo

Xarray tools for climate data: Cartopy maps, regridding, statistics, colormaps, preprocessing and memory-mapped storage. Everything is available through the `.xgeo` accessor. MPI arrays and parallel NetCDF I/O live in the separate `xrmpi` package.

## Installation

Requires Python 3.14 or newer.

```bash
python -m pip install xgeo
python -m pip install "xgeo[regrid]"   # adds xesmf
```

From a clone (the Conda spec also installs `ffmpeg` for animations and `xesmf`):

```bash
conda env create -n xgeo -f env/env.yaml
conda activate xgeo
python -m pip install --no-deps -e .
```

See [`env/README.md`](env/README.md) for updates and notebook kernels.

## Plotting

```python
import xarray as xr
import xgeo as xg

t2m = xr.open_dataset("climate.nc")["t2m"]

t2m.xgeo.geoplot(method="contourf", cmap=xg.cmaps.temp_div(), levels=21)
t2m.xgeo.geoplot(col="time", col_wrap=3)      # facets
t2m.xgeo.animate(dim="time")
```

`xg.plot.geoplot(t2m, ...)` is the functional form. Common options (`method`, `cmap`, `vmin`, `vmax`, `levels`, `projection`, `colorbar`, `colorbar_label`, `colorbar_orientation`) are plain keywords. If `projection` is omitted, it is chosen from the data extent.

Facets share axis limits. On `PlateCarree` and `Mercator` maps, only the outer panels get tick labels; set `sharex=False` or `sharey=False` to label every panel.

### Option groups

Less common settings are frozen dataclasses in `xg.plot_options`. Pass an instance, `None`, or `True` where a default exists (`hillshade=True`, `map_ticks=True`, `clabel=True`). Dictionaries raise `TypeError`.

| Keyword | Class | Controls |
| --- | --- | --- |
| `features` | `MapFeatures` | coastlines, borders, states, ocean, land, lakes, rivers, extent, grid boundary |
| `map_ticks` | `MapTicks` | tick bins, minor ticks and intervals (`xminor`, `yminor`), gridlines, label placement |
| `colorbar_style` | `ColorbarStyle` | thickness, padding, tick thinning, formatter |
| `clabel` | `ContourLabels` | contour label format and placement |
| `hillshade` | `Hillshade` | sun angle, exaggeration, relief method and opacity |
| `significance` | `xg.plot.Significance` | stippling or hatching where `pvalues < level` |
| `vectors` | `xg.plot.VectorField` | quiver arrows, key and magnitude colormap |

`Significance` and `VectorField` carry data, so they sit in `xg.plot`. All fields are keyword-only and documented in the docstrings.

Unset fields never override a lower layer. Values resolve in this order, later wins: field defaults, map theme, `set_options`, the plotting call. An explicit `None` is a value that selects the callee default.

```python
t2m.xgeo.geoplot(
    features=xg.plot_options.MapFeatures(states=False, lakes=True),
    map_ticks=xg.plot_options.MapTicks(xbins=4),
    colorbar_style=xg.plot_options.ColorbarStyle(thickness=0.03),
    significance=xg.plot.Significance(pvalues=pval, level=0.05, hatch="///"),
    vectors=xg.plot.VectorField(u=u10, v=v10, subsample=(4, 4)),
)
```

### Standing options

```python
xg.plot.set_options(map_theme="ncl")                  # until changed
with xg.plot.set_options(map_theme="gmt", rc={"font.family": "serif"}):
    xg.plot.geoplot(t2m)
xg.plot.get_options()
```

| Option | Value |
| --- | --- |
| `map_theme` | `"default"`, `"ncl"`, `"ferret"`, `"grads"`, `"gmt"` or a `MapTheme` |
| `features`, `colorbar_style`, `map_ticks` | default option instances |
| `rc` | Matplotlib rc parameters, applied while a plot is built; they override the theme |

### Overlays

`GeoPlot.add` draws on every panel, records the layers in `GeoPlot.layers` and can be chained.

```python
plot = t2m.xgeo.geoplot(method="pcolormesh")
plot.add.contour(t2m, levels=5, clabel=xg.plot_options.ContourLabels(fmt="%.0f")).add.grid_boundary()
```

Available: `plot`, `pcolormesh`, `contourf`, `contour`, `imshow`, `scatter`, `quiver`, `significance`, `colorbar`, `grid_boundary`.

### Hillshade

```python
topo.xgeo.geoplot(hillshade=True, cmap="terrain", robust=True)
topo.xgeo.geoplot(hillshade=xg.plot_options.Hillshade(azimuth=270, altitude=35, vert_exag=2))
```

The default sun is at azimuth 315 and altitude 45 degrees. With `spacing="auto"`, degree grids are converted to metres and the elevation is assumed to be in metres. Other grids fall back to unit spacing, so pass `spacing=(dx, dy)` for correct shading. `compute_hillshade` and `hillshade_spacing` in `xgeo.viz.primitives` give the array directly.

### Layout

| Module | Contents |
| --- | --- |
| [`viz/plot.py`](viz/plot.py) | `GeoPlot`, `FacetedPlot`, `Adder`, `Animate`, `geoplot`, `animate`, `colorbar`, `theme` |
| [`viz/primitives.py`](viz/primitives.py) | stateless drawing functions (`plot_field`, `plot_quiver`, `plot_significance`) and helpers |
| [`viz/options.py`](viz/options.py) | option dataclasses, `set_options`, `resolve_options` |
| [`viz/cmaps.py`](viz/cmaps.py) | IPCC, Matplotlib and cmocean palettes |

## Geospatial and statistical tools

```python
ds.xgeo.regrid(target_grid, method="bilinear")
ds.xgeo.regrid(1.0, method="conservative")               # 1 degree grid
ds_2d.xgeo.regrid(0.25, lat="lat", lon="lon", lat_b="lat_b", lon_b="lon_b")   # curvilinear
ds.xgeo.wrap_lon(convention="-180/180")
ds.xgeo.add_local_solar_time()
ds["t2m"].xgeo.trends(dim="time")
ds["t2m"].xgeo.corr(other, dim="time")
```

`regrid` needs `xesmf`. The target is a grid with 1D `lat` and `lon` or a resolution in degrees. Giving both `lat_b` and `lon_b` marks the source as curvilinear, with corner `(ny + 1, nx + 1)` or CF `(ny, nx, 4)` bounds. The output is always a 1D latitude-longitude grid. Functional versions are in `xgeo.xr_utils` (`regrid`, `mask`, `fillgaps`, `sel_transect`, `wrap_lon`, `add_cyclic_point`, `add_local_solar_time`) and `xgeo.stats`.

## Storage

XNpy stores NumPy, pandas and Xarray objects as `.npy` files with a JSON manifest, and reads them memory-mapped.

```python
xg.xnpy.save(ds, "run_001")
ds = xg.xnpy.open_dataset("run_001")
```

Serial NetCDF I/O uses Xarray directly (`ds.to_netcdf(...)`). For parallel I/O use `xrmpi`, which xgeo does not import.

## Namespaces

| Namespace | Purpose |
| --- | --- |
| `xgeo.plot` | Cartopy plotting |
| `xgeo.stats` | trends, correlations, significance tests |
| `xgeo.cmaps` | colormaps |
| `xgeo.xr_utils` | regridding, masking, gap filling, transects, longitude wrapping, local solar time |
| `xgeo.xnpy` | memory-mapped storage (`save`, `open_dataset`, `open_dataframe`, `load_array`) |
| `xgeo.preprocess` | ERA5, IMERG, CMORPH and GPCP preprocessing |
| `xgeo.grib_io` | GRIB input and output |

Distributed under the [MIT License](LICENSE).
