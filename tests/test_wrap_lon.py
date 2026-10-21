"""wrap_lon must return a new object and leave its input untouched."""

import numpy as np
import pytest
import xarray as xr
from xgeo.core.xr_utils import wrap_lon


@pytest.fixture
def da() -> xr.DataArray:
    lon = np.array([0.0, 90.0, 180.0, 270.0])
    return xr.DataArray(
        np.arange(4.0),
        dims="lon",
        coords={"lon": ("lon", lon, {"units": "degrees_east"})},
    )


def test_input_is_not_modified(da):
    before = da["lon"].values.copy()
    out = wrap_lon(da, "-180/180")
    np.testing.assert_array_equal(da["lon"].values, before)
    np.testing.assert_array_equal(out["lon"].values, [-180.0, -90.0, 0.0, 90.0])
    assert out["lon"].attrs["units"] == "degrees_east"


def test_round_trip_dataset(da):
    ds = da.to_dataset(name="v")
    back = wrap_lon(wrap_lon(ds, "-180/180"), "0/360")
    np.testing.assert_array_equal(back["lon"].values, ds["lon"].values)
    np.testing.assert_array_equal(back["v"].values, ds["v"].values)
