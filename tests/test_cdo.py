"""CDO wrapper: CF tagging of plain xarray input, wrappers, and file handling."""

import shutil

import numpy as np
import pytest
import xarray as xr
from xgeo.core import pycdo as cdo

pytestmark = pytest.mark.skipif(shutil.which("cdo") is None, reason="cdo not on PATH")


@pytest.fixture
def da() -> xr.DataArray:
    lat = np.arange(30.0, 50.01, 1.0)
    lon = np.arange(-100.0, -59.99, 1.0)
    data = np.random.default_rng(0).random((3, lat.size, lon.size))
    return xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={"time": np.arange(3), "lat": lat, "lon": lon},
        name="p",
        attrs={"units": "mm"},
    )


BBOX = (-80.0, 35.0, -65.0, 47.0)


def test_plain_dataarray_is_accepted(da):
    out = cdo.remapbil(da, resolution=0.5, bbox=BBOX)
    assert out.name == "p"
    assert out.sizes == {"time": 3, "lat": 25, "lon": 31}
    assert out.lat.values[0] == 35.0 and out.lon.values[0] == -80.0


def test_unnamed_dataarray_stays_unnamed(da):
    assert cdo.remapnn(da.rename(None), resolution=1.0, bbox=BBOX).name is None


def test_wrapper_fixes_method(da):
    ds = cdo.remapcon(da.to_dataset(), resolution=2.0, bbox=(-100, 30, -60, 50))
    assert isinstance(ds, xr.Dataset) and ds.sizes["lat"] == 11


def test_alternate_coordinate_names(da):
    alt = da.rename(lat="latitude", lon="longitude")
    out = cdo.remapbil(alt, resolution=1.0, bbox=BBOX)
    assert {"latitude", "longitude"} <= set(out.dims)


def test_outfile_creates_parent(da, tmp_path):
    out = cdo.remapbil(da, tmp_path / "a" / "b.nc", resolution=1.0, bbox=BBOX)
    assert out.exists()


def test_run_accepts_string():
    assert cdo.run("-V").returncode == 0


def test_invalid_arguments(da):
    with pytest.raises(ValueError):
        cdo.remap(da, method="nope")
    with pytest.raises(ValueError):
        cdo.remapcon(da, extrapolate=True)
    with pytest.raises(ValueError):
        cdo.remapbil(da, resolution=0.5, bbox=(-80.25, 35, -65, 47))
