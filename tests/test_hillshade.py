"""Hillshade rendering and argument propagation through maps and animations."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from matplotlib.collections import QuadMesh
from matplotlib.colors import LightSource
from xgeo.viz import plot
from xgeo.viz.options import Hillshade, MapFeatures
from xgeo.viz.primitives import compute_hillshade, hillshade_spacing, plot_field


@pytest.fixture
def terrain() -> xr.DataArray:
    lon = np.linspace(10, 16, 7)
    lat = np.linspace(8, 3, 6)
    xx, yy = np.meshgrid(lon, lat)
    return xr.DataArray(
        (xx - 12) ** 2 + (yy - 5) ** 2,
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name="topo",
        attrs={"units": "m", "long_name": "Elevation"},
    )


@pytest.fixture
def map_options() -> dict:
    return {
        "x": "lon",
        "y": "lat",
        "projection": "PlateCarree",
        "features": MapFeatures(
            coastlines=False, borders=False, states=False, ocean=False, land=False
        ),
        "figsize": (4, 3),
    }


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize(
    "method", ["default", "pcolormesh", "contourf", "contour", "imshow", "scatter"]
)
def test_geoplot_hillshade(terrain, map_options, method):
    original = terrain.copy(deep=True)
    result = terrain.xgeo.geoplot(
        hillshade=Hillshade(vert_exag=10, alpha=0.65, spacing="pixel"),
        method=method,
        cmap="terrain",
        robust=True,
        **map_options,
    )
    shade = result.axes.collections[0]
    assert isinstance(shade, QuadMesh)
    np.testing.assert_allclose(
        shade.get_array(),
        LightSource(315, 45).hillshade(terrain.values, vert_exag=10),
    )
    assert shade.get_cmap().name == "gray"
    assert shade.get_clim() == (0, 1)
    assert shade.get_alpha() == 0.65
    assert shade.get_zorder() < result.artist.get_zorder()
    assert result.artist.get_alpha() == 0.4
    if method != "contour":
        assert result.colorbar.mappable is result.artist
        assert "[m]" in result.colorbar.ax.get_ylabel()
    xr.testing.assert_identical(terrain, original)


@pytest.mark.parametrize("alpha", [None, 0.7])
def test_alpha_override(terrain, map_options, alpha):
    result = plot.geoplot(terrain, hillshade=True, alpha=alpha, **map_options)
    assert result.artist.get_alpha() == alpha


def test_disabled_hillshade(terrain, map_options):
    result = plot.geoplot(terrain, hillshade=False, **map_options)
    assert len(result.axes.collections) == 1
    assert result.artist.get_alpha() is None


@pytest.mark.parametrize("reverse_lat", [False, True])
@pytest.mark.parametrize("reverse_lon", [False, True])
def test_coordinate_order_and_transpose(terrain, reverse_lat, reverse_lon):
    field = terrain.isel(
        lat=slice(None, None, -1 if reverse_lat else 1),
        lon=slice(None, None, -1 if reverse_lon else 1),
    ).transpose("lon", "lat")
    fig, ax = plt.subplots()
    plot_field(
        "default",
        field,
        fig,
        ax,
        x="lon",
        y="lat",
        hillshade=Hillshade(vert_exag=5, spacing="pixel"),
    )
    actual = field.transpose("lat", "lon").copy(data=ax.collections[0].get_array())
    expected = terrain.copy(
        data=LightSource(315, 45).hillshade(terrain.values, vert_exag=5)
    )
    xr.testing.assert_allclose(
        actual.sortby(["lat", "lon"]), expected.sortby(["lat", "lon"])
    )


def test_nan_mask(terrain, map_options):
    field = terrain.copy()
    field[2, 3] = np.nan
    result = plot.geoplot(field, hillshade=True, **map_options)
    shaded = result.axes.collections[0].get_array()
    assert np.ma.is_masked(shaded[2, 3])
    assert np.ma.count(shaded) > 0


def test_hillshade_helper(terrain):
    original = terrain.copy(deep=True)
    shaded = compute_hillshade(terrain, z_factor=2, azimuth=90, altitude=30)
    np.testing.assert_allclose(
        shaded.values, LightSource(90, 30).hillshade(terrain.values, vert_exag=2)
    )
    assert shaded.name == terrain.name
    assert shaded.attrs == terrain.attrs
    xr.testing.assert_identical(shaded.coords.to_dataset(), terrain.coords.to_dataset())
    xr.testing.assert_identical(terrain, original)


@pytest.mark.parametrize("shape", [(3,), (1, 3), (3, 3, 3)])
def test_hillshade_helper_requires_2d_grid(shape):
    with pytest.raises(ValueError, match="2D field"):
        compute_hillshade(xr.DataArray(np.zeros(shape)))


def test_hillshade_helper_missing_data(terrain):
    field = xr.full_like(terrain, np.nan)
    assert compute_hillshade(field).isnull().all()


def test_facets(terrain, map_options):
    field = xr.concat([terrain, terrain * 2], dim=xr.IndexVariable("panel", [0, 1]))
    result = plot.geoplot(
        field,
        col="panel",
        hillshade=Hillshade(vert_exag=10, spacing="pixel"),
        **map_options,
    )
    for index, (ax, _) in enumerate(result.iter_axes()):
        np.testing.assert_allclose(
            ax.collections[0].get_array(),
            LightSource(315, 45).hillshade(
                field.isel(panel=index).values, vert_exag=10
            ),
        )
        assert result.base_artists[index].get_alpha() == 0.4
    assert result.colorbar.mappable is result.base_artists[-1]


@pytest.mark.parametrize("accessor", [False, True])
def test_animation_frames(terrain, map_options, tmp_path, monkeypatch, accessor):
    field = xr.concat([terrain, terrain * 2], dim=xr.IndexVariable("time", [0, 1]))
    original_frame = plot.plot_animation_frame
    original_hillshade = LightSource.hillshade
    calls = []

    def hillshade(self, elevation, **kwargs):
        calls.append((elevation.copy(), kwargs["vert_exag"]))
        return original_hillshade(self, elevation, **kwargs)

    def render_frame(*args):
        args = list(args)
        args[4] = 40
        original_frame(*args)

    def encode(self, input_pattern, outfile, *, fps, session_tmp_dir):
        assert len(list(Path(session_tmp_dir).glob("*.png"))) == 2

    monkeypatch.setattr(LightSource, "hillshade", hillshade)
    monkeypatch.setattr(plot, "plot_animation_frame", render_frame)
    monkeypatch.setattr(plot.Animate, "to_ffmpeg", encode)
    monkeypatch.setattr(plot, "SerialProgressBar", lambda tasks, **kwargs: tasks)
    options = {
        **map_options,
        "hillshade": Hillshade(vert_exag=10, alpha=0.65, spacing="pixel"),
        "parallel": False,
        "display_inline": False,
        "indices": [1, 0],
        "outfile": tmp_path / "terrain.mp4",
    }
    result = (
        field.xgeo.animate(**options) if accessor else plot.animate(field, **options)
    )
    assert result.geo_options["hillshade"].alpha == 0.65
    assert len(calls) == 2
    for (actual, exaggeration), index in zip(calls, [1, 0], strict=True):
        np.testing.assert_array_equal(actual, field.isel(time=index).values)
        assert exaggeration == 10


def test_default_hillshade_uses_physical_spacing(terrain, map_options):
    """The default converts a degree grid to metres instead of unit pixels."""
    result = plot.geoplot(terrain, hillshade=True, **map_options)
    dx, dy = hillshade_spacing(terrain, x="lon", y="lat")
    expected = LightSource(315, 45).hillshade(terrain.values, vert_exag=1, dx=dx, dy=dy)
    np.testing.assert_allclose(result.axes.collections[0].get_array(), expected)
    assert dx != 1.0 and dy != 1.0


def test_hillshade_light_and_colormap_overrides(terrain, map_options):
    options = Hillshade(azimuth=270, altitude=30, cmap="viridis", field_alpha=None)
    result = plot.geoplot(terrain, hillshade=options, **map_options)
    shade = result.axes.collections[0]
    dx, dy = hillshade_spacing(terrain, x="lon", y="lat")
    np.testing.assert_allclose(
        shade.get_array(),
        LightSource(270, 30).hillshade(terrain.values, vert_exag=1, dx=dx, dy=dy),
    )
    assert shade.get_cmap().name == "viridis"
    assert result.artist.get_alpha() is None


def test_hillshade_helper_spacing(terrain):
    dx, dy = 2.0, 3.0
    shaded = compute_hillshade(terrain, z_factor=2, dx=dx, dy=dy)
    np.testing.assert_allclose(
        shaded.values,
        LightSource(315, 45).hillshade(terrain.values, vert_exag=2, dx=dx, dy=dy),
    )
    with pytest.raises(ValueError, match="positive"):
        compute_hillshade(terrain, dx=0)
