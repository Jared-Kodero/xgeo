"""Grouped option dataclasses, their use in plots and the spacing helpers."""

import dataclasses
import pickle

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
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
from xgeo.viz.plot_utils import EARTH_RADIUS_M, hillshade_spacing


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
    def test_bool_none_mapping_instance(self):
        assert Hillshade.coerce(None) is None
        assert Hillshade.coerce(False) is None
        assert Hillshade.coerce(True) == Hillshade()
        assert Hillshade.coerce({"azimuth": 270}).azimuth == 270
        option = Hillshade(altitude=10)
        assert Hillshade.coerce(option) is option

    def test_invalid_value(self):
        with pytest.raises(TypeError, match="Hillshade"):
            Hillshade.coerce(3)

    def test_unknown_mapping_key(self):
        with pytest.raises(TypeError):
            Hillshade.coerce({"azimut": 1})

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
                cls.coerce({**required, "not_a_field": 1})

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

    def test_features_dict(self, field):
        result = plot.geoplot(
            field,
            x="lon",
            y="lat",
            projection="PlateCarree",
            figsize=(4, 3),
            features={"coastlines": False, "borders": False, "states": False, "ocean": False, "land": False},
        )
        assert result.map_features == []

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
        assert result.geo_options["vector_style"] == {"color": "k", "zorder": 4.0, "subsample": (1, 1), "add_key": True, "powerlimits": (-3, 3)}
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

    def test_plot_namespace_and_top_level(self):
        import xgeo

        for name in self.names:
            assert name in plot.__all__
            assert getattr(plot, name) is getattr(xgeo, name)
            assert name in xgeo.__all__
