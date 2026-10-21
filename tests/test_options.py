"""Grouped option dataclasses, their use in plots and the spacing helpers."""

import dataclasses
import pickle

import matplotlib

matplotlib.use("Agg")

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from cartopy.mpl.gridliner import Gridliner
from xgeo.viz import plot
from xgeo.viz.options import (
    ColorbarStyle,
    ContourLabels,
    Hillshade,
    MapFeatures,
    MapTicks,
    Significance,
    VectorField,
)
from xgeo.viz.primitives import (
    EARTH_RADIUS_M,
    add_map_features,
    get_cax,
    hillshade_spacing,
)


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
def quiet() -> MapFeatures:
    return MapFeatures(
        coastlines=False, borders=False, states=False, ocean=False, land=False
    )


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def base(quiet):
    return {"x": "lon", "y": "lat", "projection": "PlateCarree", "figsize": (4, 3), "features": quiet}


class TestCoerce:
    def test_bool_none_instance(self):
        assert Hillshade.coerce(None) is None
        assert Hillshade.coerce(False) is None
        assert Hillshade.coerce(True) == Hillshade()
        option = Hillshade(altitude=10)
        assert Hillshade.coerce(option) is option

    def test_invalid_value(self):
        with pytest.raises(TypeError, match="Hillshade"):
            Hillshade.coerce(3)

    @pytest.mark.parametrize(
        "cls", [ColorbarStyle, ContourLabels, Hillshade, MapFeatures, MapTicks]
    )
    def test_mapping_rejected(self, cls):
        with pytest.raises(TypeError, match="got dict"):
            cls.coerce({})

    def test_option_fields_are_keyword_only(self):
        with pytest.raises(TypeError):
            Hillshade(270.0)
        with pytest.raises(TypeError):
            MapTicks(3, 4)

    def test_required_fields_reject_true(self, field):
        with pytest.raises(TypeError, match="required"):
            VectorField.coerce(True)

    def test_no_open_ended_keywords(self, field):
        for cls, required in (
            (ContourLabels, {}),
            (Significance, {"pvalues": field}),
            (VectorField, {"u": field, "v": field}),
        ):
            names = {item.name for item in dataclasses.fields(cls)}
            assert "kwargs" not in names
            with pytest.raises(TypeError):
                cls(**required, kwargs={"color": "k"})
            with pytest.raises(TypeError):
                cls(**required, not_a_field=1)

    def test_arguments_drop_unset_fields(self, field):
        vectors = VectorField(u=field, v=field, color="k", scale=2.0)
        style = vectors.style()
        assert style["color"] == "k" and style["scale"] == 2.0
        assert "u" not in style and "v" not in style
        assert "pivot" not in style and "key_units" not in style

    def test_field_validation(self, field):
        with pytest.raises(ValueError, match="level"):
            Significance(pvalues=field, level=2.0)
        with pytest.raises(ValueError, match="alpha"):
            VectorField(u=field, v=field, alpha=1.5)

    def test_frozen_and_picklable(self):
        option = Hillshade(vert_exag=3, cmap="gray")
        with pytest.raises(AttributeError):
            option.vert_exag = 2
        assert pickle.loads(pickle.dumps(option)) == option
        assert pickle.loads(pickle.dumps(MapFeatures(extent=(0, 1, 2, 3)))).extent == (0, 1, 2, 3)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"azimuth": 400},
            {"altitude": -1},
            {"alpha": 1.5},
            {"field_alpha": 2},
            {"vert_exag": float("nan")},
            {"spacing": "metres"},
            {"spacing": (1.0, 0.0)},
            {"spacing": (1.0,)},
        ],
    )
    def test_hillshade_validation(self, kwargs):
        with pytest.raises(ValueError):
            Hillshade(**kwargs)

    def test_extent_validation(self):
        with pytest.raises(ValueError, match="extent"):
            MapFeatures(extent=(1, 2, 3))


class TestSpacing:
    def test_degrees(self, field):
        dx, dy = hillshade_spacing(field, x="lon", y="lat")
        step = np.deg2rad(1.0)
        assert dy == pytest.approx(EARTH_RADIUS_M * step)
        assert dx == pytest.approx(EARTH_RADIUS_M * np.cos(np.deg2rad(5.5)) * step)

    def test_degrees_from_attrs(self, field):
        renamed = field.rename(lat="a", lon="b")
        renamed["a"].attrs["units"] = "degrees_north"
        renamed["b"].attrs["units"] = "degrees_east"
        assert hillshade_spacing(renamed, x="b", y="a") == pytest.approx(
            hillshade_spacing(field, x="lon", y="lat")
        )

    def test_metres_and_kilometres(self, field):
        projected = field.rename(lat="ny", lon="nx").assign_coords(
            ny=np.arange(6) * 30.0, nx=np.arange(7) * 30.0
        )
        projected["ny"].attrs["units"] = "m"
        projected["nx"].attrs["units"] = "m"
        assert hillshade_spacing(projected, x="nx", y="ny") == (30.0, 30.0)
        projected["ny"].attrs["units"] = "km"
        projected["nx"].attrs["units"] = "km"
        assert hillshade_spacing(projected, x="nx", y="ny") == (30000.0, 30000.0)

    def test_unknown_units_fall_back_to_pixels(self, field):
        unknown = field.rename(lat="ny", lon="nx")
        assert hillshade_spacing(unknown, x="nx", y="ny") == (1.0, 1.0)

    def test_explicit(self, field):
        assert hillshade_spacing(field, x="lon", y="lat", spacing="pixel") == (1.0, 1.0)
        assert hillshade_spacing(field, x="lon", y="lat", spacing=(5.0, 7.0)) == (5.0, 7.0)


class TestGroupsInPlots:
    def test_colorbar_group(self, field, quiet):
        result = plot.geoplot(
            field,
            colorbar_label="Elevation",
            colorbar_orientation="horizontal",
            colorbar_style=ColorbarStyle(ticks=[0, 10], tick_labels=["lo", "hi"]),
            **base(quiet),
        )
        assert result.colorbar.orientation == "horizontal"
        assert result.colorbar.ax.get_xlabel() == "Elevation"
        assert [t.get_text() for t in result.colorbar.ax.get_xticklabels()] == ["lo", "hi"]

    def test_colorbar_disabled(self, field, quiet):
        assert plot.geoplot(field, colorbar=False, **base(quiet)).colorbar is None

    def test_features_group_and_ticks(self, field):
        result = plot.geoplot(
            field,
            x="lon",
            y="lat",
            projection="PlateCarree",
            figsize=(4, 3),
            features=MapFeatures(coastlines=False, borders=False, states=False, ocean=False, land=False),
            map_ticks=MapTicks(xbins=3, ybins=2),
        )
        assert result.map_features == []
        assert result.axes.get_xticks().size > 0

    def test_features_dict_rejected(self, field):
        with pytest.raises(TypeError, match="MapFeatures"):
            plot.geoplot(
                field,
                x="lon",
                y="lat",
                projection="PlateCarree",
                figsize=(4, 3),
                features={"coastlines": False},
            )

    def test_significance_group(self, field, quiet):
        p = xr.where(field < 3, 0.01, 0.9)
        result = plot.geoplot(field, significance=Significance(pvalues=p, level=0.05, color="k", size=0.5), **base(quiet))
        assert any(layer["kind"] == "significance" for layer in result.layers)

    def test_vector_group(self, field, quiet):
        u = xr.ones_like(field)
        result = plot.geoplot(field, vectors=VectorField(u=u, v=u, color="k", scale_units="xy", scale=2.0, pivot="mid"), **base(quiet))
        assert result.quiver is not None

    def test_facets_use_groups(self, field, quiet):
        stacked = xr.concat([field, field * 2], dim=xr.IndexVariable("panel", [0, 1]))
        result = plot.geoplot(
            stacked,
            col="panel",
            method="contour",
            clabel=ContourLabels(fmt="%.1f"),
            hillshade=Hillshade(spacing="pixel"),
            **base(quiet),
        )
        assert len(result.contour_labels) == 2
        # Labels are only placed where a contour segment is long enough.
        assert sum(len(labels) for labels in result.contour_labels) > 0

    def test_adder_contour_labels(self, field, quiet):
        result = plot.geoplot(field, **base(quiet))
        result.add.contour(field, clabel=ContourLabels(fmt="%.1f"))
        layer = result.layers[-1]
        assert layer["kind"] == "contour" and layer["labels"]
        result.add.contour(field, clabel=True)
        assert result.layers[-1]["labels"]

    def test_contour_labels_group(self, field, quiet):
        result = plot.geoplot(
            field, method="contour", clabel=ContourLabels(fmt="%.2f"), **base(quiet)
        )
        assert len(result.contour_labels[0].get_text().split(".")[1]) == 2

    def test_contour_label_named_fields(self, field, quiet):
        options = ContourLabels(
            fmt="%.1f", fontsize=6.0, inline_spacing=3.0, rightside_up=False, zorder=7.0
        )
        result = plot.geoplot(field, method="contour", clabel=options, **base(quiet))
        assert result.contour_labels
        assert all(text.get_fontsize() == 6.0 for text in result.contour_labels)
        assert all(text.get_zorder() == 7.0 for text in result.contour_labels)

    def test_contour_is_always_labelled(self, field, quiet):
        result = plot.geoplot(field, method="contour", **base(quiet))
        assert result.contour_labels

    def test_disabled_groups(self, field, quiet):
        result = plot.geoplot(field, **base(quiet))
        assert result.quiver is None
        assert not any(layer["kind"] == "significance" for layer in result.layers)


class TestAnimateOptions:
    def make(self, field, quiet, tmp_path, monkeypatch, **extra):
        stacked = xr.concat([field, field * 2], dim=xr.IndexVariable("time", [0, 1]))
        captured = []
        original = plot.plot_animation_frame

        def render(*args):
            args = list(args)
            args[4] = 40
            captured.append(args)
            original(*args)

        monkeypatch.setattr(plot, "plot_animation_frame", render)
        monkeypatch.setattr(plot.Animate, "to_ffmpeg", lambda self, *a, **k: None)
        monkeypatch.setattr(plot, "SerialProgressBar", lambda tasks, **kwargs: tasks)
        result = plot.animate(
            stacked,
            parallel=False,
            display_inline=False,
            outfile=tmp_path / "a.mp4",
            **base(quiet),
            **extra,
        )
        return stacked, result, captured

    def test_geo_options_filtered_by_signature(self, field, quiet, tmp_path, monkeypatch):
        _, result, _ = self.make(field, quiet, tmp_path, monkeypatch, hillshade=Hillshade(spacing="pixel"))
        options = result.geo_options
        for name in ("da", "title", "dim", "indices", "outfile", "quality", "fps", "parallel", "frame_id", "display_inline", "vectors"):
            assert name not in options
        assert options["method"] == "default"
        assert options["hillshade"].spacing == "pixel"
        assert options["features"] is quiet
        assert options["vector_style"] is None

    def test_vectors_per_frame(self, field, quiet, tmp_path, monkeypatch):
        stacked = xr.concat([field, field * 2], dim=xr.IndexVariable("time", [0, 1]))
        _, result, captured = self.make(
            field,
            quiet,
            tmp_path,
            monkeypatch,
            vectors=VectorField(u=xr.ones_like(stacked), v=xr.ones_like(stacked), color="k"),
        )
        assert result.geo_options["vector_style"] == {"color": "k", "zorder": 4.0, "subsample": (1, 1), "add_key": True}
        assert len(captured) == 2
        assert all(args[7] is not None and args[8] is not None for args in captured)
        assert all(args[7].ndim == 2 for args in captured)

    def test_geo_options_pickle(self, field, quiet, tmp_path, monkeypatch):
        _, result, _ = self.make(field, quiet, tmp_path, monkeypatch, hillshade=True)
        restored = pickle.loads(pickle.dumps(result.geo_options))
        assert restored["hillshade"] is True
        assert restored["features"] == quiet


class TestExports:
    names = (
        "ColorbarStyle",
        "ContourLabels",
        "Hillshade",
        "MapFeatures",
        "MapTicks",
        "Significance",
        "VectorField",
    )

    def test_plot_options_namespace(self):
        import xgeo
        from xgeo.viz import options

        assert xgeo.plot_options is options
        assert "plot_options" in xgeo.__all__
        for name in self.names:
            assert getattr(xgeo.plot_options, name) is getattr(options, name)
            assert name in options.__all__
            assert name not in xgeo.__all__
            assert not hasattr(xgeo, name)


class TestAdvancedMapFeatures:
    @pytest.fixture
    def axis(self):
        figure = plt.figure()
        return figure, figure.add_subplot(projection=ccrs.PlateCarree())

    def test_unset_fields_are_not_passed(self):
        assert MapFeatures().feature_arguments() == {
            "coastlines": True,
            "borders": True,
            "states": True,
            "ocean": True,
            "land": True,
            "lakes": False,
            "rivers": False,
            "global_extent": False,
        }
        assert MapFeatures().boundary_arguments() == {}

    def test_arguments_split_between_features_and_outline(self):
        options = MapFeatures(
            extent=(0, 1, 2, 3),
            resolution="50m",
            states_alpha=0.5,
            grid_bounds=True,
            grid_bounds_linewidth=2.0,
            grid_bounds_color="red",
            grid_bounds_zorder=4.0,
        )
        arguments = options.feature_arguments()
        assert arguments["resolution"] == "50m" and arguments["states_alpha"] == 0.5
        assert not {"extent", "grid_bounds"} & set(arguments)
        assert not any(name.startswith("grid_bounds_") for name in arguments)
        assert options.boundary_arguments() == {
            "linewidth": 2.0,
            "color": "red",
            "zorder": 4.0,
        }

    def test_defaults_reproduce_the_fixed_styling(self, axis):
        figure, ax = axis
        artists = add_map_features(
            figure, ax, lakes=True, rivers=True, ocean=True, land=False
        )
        kwargs = [artist._kwargs for artist in artists]
        assert kwargs == [
            {},
            {"linestyle": "-", "zorder": 3, "alpha": 0.3},
            {"linestyle": "-", "zorder": 3, "alpha": 0.3},
            {"zorder": 2},
            {"zorder": 2},
            {"facecolor": "#54585f", "zorder": 0},
        ]
        assert {artist._feature.scale for artist in artists} == {"110m"}

    def test_ocean_mask_default(self, axis):
        figure, ax = axis
        (artist,) = add_map_features(
            figure,
            ax,
            coastlines=False,
            states=False,
            borders=False,
            ocean=False,
            land=True,
        )
        assert artist._kwargs == {"zorder": 0}

    def test_custom_styling(self, axis):
        figure, ax = axis
        artists = add_map_features(
            figure,
            ax,
            lakes=True,
            rivers=True,
            ocean=False,
            land=True,
            resolution="50m",
            coastline_color="red",
            coastline_linewidth=0.5,
            states_alpha=0.6,
            borders_alpha=0.1,
            boundary_linestyle=":",
            boundary_color="blue",
            boundary_linewidth=2.0,
            boundary_zorder=7,
            water_color="cyan",
            water_zorder=5,
            ocean_color="navy",
            background_zorder=-1,
        )
        coast, states, borders, lakes, rivers, ocean = (a._kwargs for a in artists)
        assert coast == {"edgecolor": "red", "linewidth": 0.5}
        assert states["alpha"] == 0.6 and borders["alpha"] == 0.1
        for boundary in (states, borders):
            assert boundary["linestyle"] == ":" and boundary["zorder"] == 7
            assert boundary["edgecolor"] == "blue" and boundary["linewidth"] == 2.0
        assert lakes == {"zorder": 5, "facecolor": "cyan", "edgecolor": "cyan"}
        assert rivers == {"zorder": 5, "edgecolor": "cyan"}
        assert ocean == {"zorder": -1, "facecolor": "navy"}
        assert {artist._feature.scale for artist in artists} == {"50m"}

    def test_land_mask_color(self, axis):
        figure, ax = axis
        (artist,) = add_map_features(
            figure,
            ax,
            coastlines=False,
            states=False,
            borders=False,
            ocean=True,
            land=False,
            land_color="tan",
            background_zorder=1,
        )
        assert artist._kwargs == {"facecolor": "tan", "zorder": 1}

    def test_options_reach_the_plot(self, field, quiet, monkeypatch):
        captured = {}
        original = plot.add_map_features

        def spy(*args, **kwargs):
            captured.update(kwargs)
            return original(*args, **kwargs)

        monkeypatch.setattr(plot, "add_map_features", spy)
        features = MapFeatures(
            coastlines=False,
            borders=False,
            states=False,
            ocean=False,
            land=False,
            resolution="10m",
            boundary_color="grey",
        )
        plot.geoplot(field, **{**base(quiet), "features": features})
        assert captured["resolution"] == "10m"
        assert captured["boundary_color"] == "grey"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"resolution": "30m"},
            {"states_alpha": 1.5},
            {"borders_alpha": -0.1},
            {"coastline_linewidth": -1.0},
            {"boundary_linewidth": float("nan")},
            {"grid_bounds_linewidth": -2.0},
        ],
    )
    def test_validation(self, kwargs):
        with pytest.raises(ValueError):
            MapFeatures(**kwargs)

    def test_grid_boundary_defaults_and_overrides(self, field):
        plain = MapFeatures(
            coastlines=False,
            borders=False,
            states=False,
            ocean=False,
            land=False,
            grid_bounds=True,
        )
        line = plot.geoplot(field, **{**base(plain), "features": plain}).axes.lines[-1]
        assert (line.get_linewidth(), line.get_color(), line.get_zorder()) == (
            1.0,
            "black",
            1,
        )
        styled = dataclasses.replace(
            plain,
            grid_bounds_linewidth=3.0,
            grid_bounds_color="red",
            grid_bounds_zorder=5.0,
        )
        line = plot.geoplot(field, **{**base(styled), "features": styled}).axes.lines[-1]
        assert (line.get_linewidth(), line.get_color(), line.get_zorder()) == (
            3.0,
            "red",
            5.0,
        )

    def test_grid_boundary_in_facets(self, field):
        stacked = xr.concat([field, field * 2], dim=xr.IndexVariable("panel", [0, 1]))
        features = MapFeatures(
            coastlines=False,
            borders=False,
            states=False,
            ocean=False,
            land=False,
            grid_bounds=True,
            grid_bounds_linewidth=2.5,
        )
        result = plot.geoplot(
            stacked, col="panel", **{**base(features), "features": features}
        )
        assert all(axis.lines[-1].get_linewidth() == 2.5 for axis in result.axes.flat)


class TestAdvancedMapTicks:
    def draw(self, field, quiet, ticks, **extra):
        return plot.geoplot(field, map_ticks=ticks, **{**base(quiet), **extra})

    def test_tick_arguments(self):
        assert MapTicks().tick_arguments() == {"xticks_bins": 5, "yticks_bins": 5}
        options = MapTicks(xbins=3, ybins=2, direction="in", gridlines=True)
        assert options.tick_arguments() == {
            "xticks_bins": 3,
            "yticks_bins": 2,
            "direction": "in",
            "gridlines": True,
        }

    def test_defaults_are_unchanged(self, field, quiet):
        ax = self.draw(field, quiet, MapTicks()).axes
        assert [t.get_text() for t in ax.get_xticklabels()][:3] == ["9", "10.5", "12"]
        assert type(ax.xaxis.get_minor_locator()).__name__ == "NullLocator"
        assert ax.xaxis.get_tick_params()["direction"] == "out"
        assert not ax.xaxis.get_tick_params()["labeltop"]
        assert not ax.yaxis.get_tick_params()["labelright"]
        assert not [a for a in ax.artists if isinstance(a, Gridliner)]

    def test_rectangular_projection_overrides(self, field, quiet):
        options = MapTicks(
            degree_symbol="\u00b0",
            direction_label=True,
            direction="in",
            top_labels=True,
            right_labels=True,
            minor_ticks=True,
            gridlines=True,
        )
        ax = self.draw(field, quiet, options).axes
        assert [t.get_text() for t in ax.get_xticklabels()][:1] == ["9\u00b0E"]
        assert [t.get_text() for t in ax.get_yticklabels()][:1] == ["3\u00b0N"]
        assert ax.xaxis.get_tick_params()["direction"] == "in"
        assert ax.xaxis.get_tick_params()["labeltop"]
        assert ax.yaxis.get_tick_params()["labelright"]
        assert type(ax.xaxis.get_minor_locator()).__name__ == "AutoMinorLocator"
        assert len([a for a in ax.artists if isinstance(a, Gridliner)]) == 1

    def test_gridliner_projection_defaults_and_overrides(self, field, quiet):
        ax = self.draw(field, quiet, MapTicks(), projection="Robinson").axes
        (gridliner,) = [a for a in ax.artists if isinstance(a, Gridliner)]
        assert not (gridliner.xlines or gridliner.ylines)
        assert not (gridliner.top_labels or gridliner.right_labels)
        options = MapTicks(gridlines=True, top_labels=True, right_labels=True)
        ax = self.draw(field, quiet, options, projection="Robinson").axes
        (gridliner,) = [a for a in ax.artists if isinstance(a, Gridliner)]
        assert gridliner.xlines and gridliner.ylines
        assert gridliner.top_labels and gridliner.right_labels

    def test_custom_coordinate_names(self, field, quiet):
        renamed = field.rename(lon="longitude", lat="latitude")
        result = plot.geoplot(
            renamed,
            x="longitude",
            y="latitude",
            projection="PlateCarree",
            figsize=(4, 3),
            features=quiet,
            map_ticks=True,
        )
        assert [t.get_text() for t in result.axes.get_xticklabels()][:1] == ["9"]

    def test_custom_coordinate_names_in_facets(self, field, quiet):
        renamed = field.rename(lon="longitude", lat="latitude")
        stacked = xr.concat(
            [renamed, renamed * 2], dim=xr.IndexVariable("panel", [0, 1])
        )
        result = plot.geoplot(
            stacked,
            x="longitude",
            y="latitude",
            col="panel",
            projection="PlateCarree",
            figsize=(6, 3),
            features=quiet,
            map_ticks=MapTicks(direction="inout"),
        )
        assert all(
            axis.xaxis.get_tick_params()["direction"] == "inout"
            for axis in result.axes.flat
        )

    def test_validation(self):
        with pytest.raises(ValueError, match="direction"):
            MapTicks(direction="sideways")


class TestAdvancedColorbarStyle:
    def test_defaults_are_unset(self):
        assert ColorbarStyle().advanced_arguments() == {}

    def test_advanced_arguments(self):
        options = ColorbarStyle(
            thickness=0.05, pad=0.1, facet_length=0.8, minimal_ticks_threshold=4
        )
        assert options.advanced_arguments() == {
            "thickness": 0.05,
            "pad": 0.1,
            "facet_length": 0.8,
            "minimal_ticks_threshold": 4,
        }

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"thickness": 0.0},
            {"thickness": -0.1},
            {"pad": -0.1},
            {"pad_with_labels": float("nan")},
            {"facet_length": 0.0},
            {"facet_length": 1.5},
            {"minimal_ticks_threshold": 0},
            {"minimal_ticks_threshold": True},
            {"minimal_ticks_threshold": 2.5},
        ],
    )
    def test_validation(self, kwargs):
        with pytest.raises(ValueError):
            ColorbarStyle(**kwargs)

    def geometry(self, field, quiet, style):
        result = plot.geoplot(field, colorbar_style=style, **base(quiet))
        axes = result.axes.get_position()
        bar = result.colorbar.ax.get_position()
        return bar.width, bar.x0 - axes.x1

    def test_vertical_thickness_and_pad_reach_the_colorbar(self, field, quiet):
        width, gap = self.geometry(field, quiet, None)
        wide, far = self.geometry(
            field, quiet, ColorbarStyle(thickness=0.06, pad=0.08)
        )
        assert wide == pytest.approx(2 * width)
        assert far == pytest.approx(2 * gap)

    @pytest.mark.parametrize(
        ("kwargs", "gap"),
        [
            ({"pad_bottom": False}, 0.05),
            ({"pad_bottom": True}, 0.1),
            ({"pad_bottom": False, "pad": 0.07}, 0.07),
            ({"pad_bottom": True, "pad": 0.07}, 0.12),
            ({"pad_bottom": True, "pad": 0.07, "pad_with_labels": 0.2}, 0.2),
        ],
    )
    def test_horizontal_pad(self, kwargs, gap):
        figure, ax = plt.subplots(figsize=(5, 4.8))
        cax = get_cax(
            fig=figure, axes=ax, orientation="horizontal", adjust=False, **kwargs
        )
        assert ax.get_position().y0 - cax.get_position().y0 == pytest.approx(gap)
        assert cax.get_position().height == pytest.approx(0.04)

    def test_horizontal_thickness(self):
        figure, ax = plt.subplots(figsize=(5, 4.8))
        cax = get_cax(
            fig=figure,
            axes=ax,
            orientation="horizontal",
            adjust=False,
            thickness=0.1,
        )
        assert cax.get_position().height == pytest.approx(0.1)

    @pytest.mark.parametrize("length", [0.5, 0.8])
    def test_facet_length(self, length):
        figure, axes = plt.subplots(2, 1, figsize=(5, 6))
        cax = get_cax(
            fig=figure,
            axes=axes,
            orientation="vertical",
            adjust=False,
            facet_length=length,
        )
        grid = axes[0].get_position().y1 - axes[1].get_position().y0
        assert cax.get_position().height == pytest.approx(length * grid)

    def test_minimal_ticks_threshold(self, field, quiet):
        levels = np.linspace(0, 26, 10)
        kwargs = {"method": "contourf", "levels": levels, **base(quiet)}
        thinned = plot.geoplot(field, **kwargs).colorbar
        kept = plot.geoplot(
            field, colorbar_style=ColorbarStyle(minimal_ticks_threshold=20), **kwargs
        ).colorbar
        assert len(thinned.get_ticks()) == 5
        assert len(kept.get_ticks()) == 10

    def test_adder_colorbar_uses_the_plot_style(self, field, quiet):
        result = plot.geoplot(
            field,
            colorbar=False,
            colorbar_style=ColorbarStyle(thickness=0.06),
            **base(quiet),
        )
        result.add.colorbar()
        reference = plot.geoplot(field, colorbar=False, **base(quiet))
        reference.add.colorbar()
        assert result.colorbar.ax.get_position().width == pytest.approx(
            2 * reference.colorbar.ax.get_position().width
        )


class TestAdvancedVectorField:
    def test_defaults_are_unset(self, field):
        style = VectorField(u=field, v=field).style()
        assert not {"angles", "key_fontsize", "key_gap"} & set(style)

    def test_style_carries_set_fields(self, field):
        style = VectorField(
            u=field, v=field, angles="uv", key_fontsize=12.0, key_gap=6.0
        ).style()
        assert style["angles"] == "uv"
        assert style["key_fontsize"] == 12.0 and style["key_gap"] == 6.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"angles": "polar"},
            {"key_fontsize": 0.0},
            {"key_fontsize": float("nan")},
            {"key_gap": -1.0},
        ],
    )
    def test_validation(self, field, kwargs):
        with pytest.raises(ValueError):
            VectorField(u=field, v=field, **kwargs)

    def test_defaults_are_unchanged(self, field, quiet):
        ones = xr.ones_like(field)
        result = plot.geoplot(field, vectors=VectorField(u=ones, v=ones), **base(quiet))
        assert result.quiver.angles == "xy"
        assert result.quiver_key.text.get_fontsize() == 10.0
        assert result.quiver_key.gap_points == 4.0

    def test_overrides_reach_the_quiver(self, field, quiet):
        ones = xr.ones_like(field)
        vectors = VectorField(
            u=ones, v=ones, angles="uv", key_fontsize=14.0, key_gap=9.0
        )
        result = plot.geoplot(field, vectors=vectors, **base(quiet))
        assert result.quiver.angles == "uv"
        assert result.quiver_key.text.get_fontsize() == 14.0
        assert result.quiver_key.gap_points == 9.0

    def test_adder_accepts_the_new_arguments(self, field, quiet):
        ones = xr.ones_like(field)
        result = plot.geoplot(field, **base(quiet))
        result.add.quiver(ones, ones, key_fontsize=7.0, key_gap=2.0, angles="uv")
        assert result.quiver.angles == "uv"
        assert result.quiver_key.gap_points == 2.0
