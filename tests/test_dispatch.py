"""Scalar dispatch, ``Adder`` layers and the title and feature helpers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from matplotlib.collections import PathCollection, QuadMesh
from matplotlib.contour import QuadContourSet
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from xarray.plot.accessor import DataArrayPlotAccessor
from xgeo.viz import plot
from xgeo.viz.options import ContourLabels, MapFeatures
from xgeo.viz.primitives import plot_field


@pytest.fixture
def field() -> xr.DataArray:
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
def stacked(field) -> xr.DataArray:
    return xr.concat([field, field * 2], dim=xr.IndexVariable("time", [0, 1]))


@pytest.fixture
def quiet() -> MapFeatures:
    return MapFeatures(
        coastlines=False, borders=False, states=False, ocean=False, land=False
    )


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def base(quiet):
    return {
        "x": "lon",
        "y": "lat",
        "projection": "PlateCarree",
        "figsize": (4, 3),
        "features": quiet,
    }


class TestDrawScalar:
    @pytest.fixture
    def axis(self, quiet):
        result = plot.geoplot(
            xr.DataArray(
                np.zeros((2, 2)),
                dims=("lat", "lon"),
                coords={"lat": [0, 1], "lon": [0, 1]},
            ),
            **base(quiet),
        )
        return result.figure, result.axes

    @pytest.mark.parametrize(
        ("method", "primitive"),
        [
            ("default", QuadMesh),
            ("pcolormesh", QuadMesh),
            ("contourf", QuadContourSet),
            ("contour", QuadContourSet),
            ("imshow", AxesImage),
        ],
    )
    def test_generic_keys_are_dropped_when_not_declared(
        self, field, axis, method, primitive
    ):
        fig, ax = axis
        artist = plot_field(
            method,
            field,
            fig,
            ax,
            x="lon",
            y="lat",
            levels=4,
            extend="both",
            robust=True,
            rasterized=True,
            zorder=2.0,
            add_labels=False,
        )
        assert isinstance(artist, primitive)

    def test_unknown_method(self, field, axis):
        fig, ax = axis
        with pytest.raises(ValueError, match="unsupported plot method 'surface'"):
            plot_field("surface", field, fig, ax, x="lon", y="lat")

    def test_scatter_receives_only_the_keys_it_applies(self, field, axis, monkeypatch):
        captured = {}

        def fake(self, *args, **kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(DataArrayPlotAccessor, "scatter", fake)
        fig, ax = axis
        plot_field(
            "scatter",
            field,
            fig,
            ax,
            x="lon",
            y="lat",
            levels=4,
            extend="both",
            robust=True,
            rasterized=True,
            zorder=5.0,
            size=3.0,
            alpha=0.5,
        )
        assert {"levels", "extend", "robust", "rasterized", "size"}.isdisjoint(captured)
        assert captured["x"] == "lon" and captured["y"] == "lat"
        assert captured["zorder"] == 5.0 and captured["s"] == 3.0
        assert captured["alpha"] == 0.5
        assert captured["add_colorbar"] is False and captured["add_labels"] is False

    def test_tuple_scatter(self, axis):
        fig, ax = axis
        points = (np.array([11.0, 13.0]), np.array([5.0, 6.0]))
        artist = plot_field(
            "scatter", points, fig, ax, x="lon", levels=3, robust=True, s=10, size=99
        )
        assert isinstance(artist, PathCollection)
        assert artist.get_sizes()[0] == 10
        alias = plot_field("scatter", points, fig, ax, size=7)
        assert alias.get_sizes()[0] == 7

    def test_tuple_data_requires_scatter(self, field, axis):
        fig, ax = axis
        with pytest.raises(TypeError, match="scatter"):
            plot_field("pcolormesh", (np.zeros(2), np.zeros(2)), fig, ax)

    def test_default_zorder_is_one(self, field, axis):
        fig, ax = axis
        kwargs = {"x": "lon", "y": "lat"}
        for method in ("pcolormesh", "contour"):
            artist = plot_field(method, field, fig, ax, levels=3, **kwargs)
            assert artist.get_zorder() == 1.0

    def test_non_generic_keys_pass_through_to_the_primitive(self, field, axis):
        fig, ax = axis
        # Unknown keywords are not filtered here: Matplotlib rejects them.
        with pytest.raises((TypeError, AttributeError), match="not_a_keyword"):
            plot_field("pcolormesh", field, fig, ax, x="lon", y="lat", not_a_keyword=1)


class TestAdder:
    def test_layers_chain_and_register(self, field, quiet):
        result = plot.geoplot(field, **base(quiet))
        returned = (
            result.add.plot(field)
            .add.pcolormesh(field)
            .add.contourf(field, levels=3)
            .add.contour(field, levels=3)
            .add.imshow(field)
        )
        assert returned is result
        assert [layer["kind"] for layer in result.layers] == [
            "default",
            "default",
            "pcolormesh",
            "contourf",
            "contour",
            "imshow",
        ]

    def test_only_contour_layers_carry_labels(self, field, quiet):
        result = plot.geoplot(field, **base(quiet))
        result.add.pcolormesh(field)
        assert "labels" not in result.layers[-1]
        result.add.contour(field, levels=3)
        assert result.layers[-1]["labels"] == []
        result.add.contour(field, levels=3, clabel=ContourLabels(fmt="%.0f"))
        assert result.layers[-1]["labels"]

    def test_faceted_layer_has_one_artist_per_panel(self, stacked, quiet):
        result = plot.geoplot(stacked, col="time", **base(quiet))
        result.add.contourf(stacked, levels=3)
        assert len(result.layers[-1]["artists"]) == 2

    def test_explicit_coordinate_names_are_used(self, field, quiet):
        result = plot.geoplot(field, **base(quiet))
        result.add.pcolormesh(field, x="lon", y="lat")
        assert isinstance(result.layers[-1]["artists"][0], QuadMesh)

    def test_contour_zorder_stays_promoted(self, field, quiet):
        result = plot.geoplot(field, **base(quiet))
        result.add.contour(field, levels=3).add.pcolormesh(field)
        contour = result.layers[1]["artists"][0]
        assert contour.get_zorder() >= 100

    def test_grid_boundary_registers_once(self, stacked, quiet):
        result = plot.geoplot(stacked, col="time", **base(quiet))
        before = len(result.layers)
        result.add.grid_boundary()
        assert len(result.layers) == before + 1
        artists = result.layers[-1]["artists"]
        assert len(artists) == 2 and all(isinstance(line, Line2D) for line in artists)

    def test_scatter_tuple_ignores_xarray_only_arguments(self, field, quiet):
        result = plot.geoplot(field, **base(quiet))
        points = (np.array([11.0, 13.0]), np.array([5.0, 6.0]))
        result.add.scatter(points, s=12, hue="ignored", markersize="ignored")
        layer = result.layers[-1]
        assert layer["kind"] == "scatter"
        assert isinstance(layer["artists"][0], PathCollection)
        assert "labels" not in layer

    def test_scatter_rejects_invalid_data(self, field, quiet):
        result = plot.geoplot(field, **base(quiet))
        with pytest.raises(TypeError):
            result.add.scatter(np.zeros(2))
        with pytest.raises(TypeError, match="numpy.ndarray"):
            result.add.scatter(([1.0], [2.0]))
        with pytest.raises(ValueError, match="matching shapes"):
            result.add.scatter((np.zeros(2), np.zeros(3)))


class TestTitles:
    def test_dict_title_single_axis(self, field, quiet):
        result = plot.geoplot(
            field, title={"label": "Relief", "loc": "left"}, **base(quiet)
        )
        assert result.axes.get_title(loc="left") == "Relief"

    def test_string_title_single_axis(self, field, quiet):
        result = plot.geoplot(field, title="Relief", **base(quiet))
        assert result.axes.get_title() == "Relief"

    def test_faceted_titles(self, stacked, quiet):
        result = plot.geoplot(stacked, col="time", title="Relief", **base(quiet))
        assert result.figure._suptitle.get_text() == "Relief"
        result = plot.geoplot(
            stacked, col="time", title={"label": "Dict", "fontsize": 9}, **base(quiet)
        )
        assert result.figure._suptitle.get_text() == "Dict"
        assert result.figure._suptitle.get_fontsize() == 9
