# Environment for an xgeo clone

This directory contains the environment for users who clone `xgeo`. It is independent of the author's `.workspace` environment and of `xrmpi/env/`.

## Create the environment

Install Conda or Mamba first, then run from the cloned `xgeo` repository root, where `pyproject.toml` and `__init__.py` are located:

```bash
conda env create -n xgeo -f env/env.yaml
conda activate xgeo
python -m pip install --no-deps -e .
python -m pip check
```

The specification selects Python 3.14 or newer, installs `xgeo`'s runtime dependencies, and includes the native libraries needed by optional regridding and GRIB features. The editable install exposes the clone as `xgeo`, so source edits are available without reinstalling.

For a different environment name, replace `xgeo` in the first two commands. With Mamba, use `mamba env create` with the same specification.

## What the environment includes

The numerical and plotting stack includes NumPy, pandas, SciPy, Xarray, Dask, Matplotlib, Cartopy, cf-xarray, scientific colormaps, and statistical helpers. NetCDF uses the ordinary serial backend.

`xesmf` and its native ESMF dependencies support regridding. `cfgrib` and the ecCodes Python bindings support GRIB input/output. `cdo`, `nco`, and `ffmpeg` are installed as external executables. JupyterLab, a kernel, and widgets are included for interactive analysis.

For a notebook kernel:

```bash
python -m ipykernel install --user --name xgeo --display-name "Python (xgeo)"
jupyter lab
```

## Update an existing environment

From the clone root:

```bash
conda env update -n xgeo -f env/env.yaml
conda activate xgeo
python -m pip install --no-deps -e .
python -m pip check
```

If the clone moves, rerun the editable install so its source path is updated.

## Use another environment

To let pip install the ordinary dependencies into an existing environment:

```bash
python -m pip install -e .
```

Optional extras are `regrid` for xESMF and `grib` for GRIB bindings:

```bash
python -m pip install -e ".[regrid,grib]"
```

Conda is the provided route for the native regridding/GRIB stack and command-line tools. pip installing the package does not install `cdo`, `nco`, or `ffmpeg`.

## MPI workflows

Distributed arrays and parallel NetCDF output are in the separate `xrmpi` package. Clone that package and follow its `env/env.md`; this serial `xgeo` environment does not rebuild an MPI stack.
