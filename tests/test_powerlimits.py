"""Shared ``powerlimits`` for colorbar tick labels and the quiver key label."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from xgeo.viz import plot
from xgeo.viz.options import ColorbarStyle, MapFeatures, VectorField
from xgeo.viz.primitives import format_scalar, make_scalar_formatter


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def quiet() -> MapFeatures:
    return MapFeatures(
        coastlines=False, borders=False, states=False, ocean=False, land=False
    )


def make_field(scale: float) -> xr.DataArray:
    lon = np.linspace(10, 16, 7)
    lat = np.linspace(8, 3, 6)
    xx, yy = np.meshgrid(lon, lat)
    return xr.DataArray(
        ((xx - 12) ** 2 + (yy - 5) ** 2 + 1.0) * scale,
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name="f",
        attrs={"units": "m", "long_name": "Field"},
    )


def draw(field, quiet, **kwargs):
    return plot.geoplot(
        field,
        x="lon",
        y="lat",
        projection="PlateCarree",
        figsize=(4, 3),
        features=quiet,
        **kwargs,
    )


def offset_text(result) -> str:
    result.figure.canvas.draw()
    return result.colorbar.ax.yaxis.get_offset_text().get_text()


def key_label(result) -> str:
    return result.quiver_key.text.get_text()


class TestFormatScalar:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (200.0, "200"),
            (2.5, "2.5"),
            (0.05, "0.05"),
            (1.0, "1"),
            (500.0, "500"),
            (2.5e-5, r"$2.5\times10^{-5}$"),
            (5e-4, r"$5\times10^{-4}$"),
            (2e5, r"$2\times10^{5}$"),
            (1e3, r"$1\times10^{3}$"),
        ],
    )
    def test_default_limits(self, value, expected):
        assert format_scalar(value) == expected

    def test_limits_move_the_switch(self):
        assert format_scalar(5000.0) == r"$5\times10^{3}$"
        assert format_scalar(5000.0, (-3, 5)) == "5000"
        assert format_scalar(0.002, (-3, 3)) == r"$2\times10^{-3}$"
        assert format_scalar(0.002, (-5, 3)) == "0.002"

    def test_zero_and_nonfinite(self):
        assert format_scalar(0.0) == "0"
        assert format_scalar(float("nan")) == "nan"

    def test_matches_scalar_formatter_threshold(self):
        # The key switches to scientific notation at the same order of magnitude
        # as a ScalarFormatter configured with the same limits.
        for exponent in range(-6, 7):
            value = 3.0 * 10.0**exponent
            formatter = make_scalar_formatter((-3, 3), use_offset=False)
            formatter.create_dummy_axis()
            formatter.axis.set_view_interval(0.0, value)
            formatter.set_locs([value])
            assert ("times" in format_scalar(value)) == bool(formatter.orderOfMagnitude)


class TestColorbarStyle:
    def test_default_and_validation(self):
        assert ColorbarStyle().powerlimits == (-3, 3)
        assert ColorbarStyle(powerlimits=(-2, 4)).powerlimits == (-2, 4)
        for bad in [(3, -3), (1,), 3, (0.5, 2), (True, 2)]:
            with pytest.raises(ValueError, match="powerlimits"):
                ColorbarStyle(powerlimits=bad)

    def test_vector_field_default_is_unset(self):
        field = make_field(1.0)
        vectors = VectorField(u=field, v=field)
        assert vectors.powerlimits is None
        assert "powerlimits" not in vectors.style()
        with pytest.raises(ValueError, match="powerlimits"):
            VectorField(u=field, v=field, powerlimits=(4, -4))


class TestColorbar:
    def test_default_limits_use_offset_text(self, quiet):
        result = draw(make_field(1e4), quiet)
        assert "10^{5}" in offset_text(result)

    def test_colorbar_style_limits_reach_the_colorbar(self, quiet):
        field = make_field(1e5)
        result = draw(field, quiet, colorbar_style=ColorbarStyle(powerlimits=(-3, 9)))
        assert offset_text(result) == ""
        formatter = result.colorbar.formatter
        assert tuple(formatter._powerlimits) == (-3, 9)

    def test_scalar_formatter_off_ignores_limits(self, quiet):
        result = draw(
            make_field(1e5),
            quiet,
            colorbar_style=ColorbarStyle(scalar_formatter=False, powerlimits=(-3, 9)),
        )
        assert tuple(result.colorbar.formatter._powerlimits) != (-3, 9)

    def test_adder_colorbar_inherits_and_overrides(self, quiet):
        result = draw(
            make_field(1e5),
            quiet,
            colorbar=False,
            colorbar_style=ColorbarStyle(powerlimits=(-3, 9)),
        )
        result.add.colorbar()
        assert tuple(result.colorbar.formatter._powerlimits) == (-3, 9)
        result.add.colorbar(powerlimits=(-1, 2))
        assert tuple(result.colorbar.formatter._powerlimits) == (-1, 2)


class TestQuiverKey:
    def vectors(self, field, magnitude, **kwargs):
        u = xr.full_like(field, magnitude)
        kwargs.setdefault("key_units", "")
        return VectorField(u=u, v=xr.zeros_like(field), key_magnitude=magnitude, **kwargs)

    def test_small_magnitude_is_scientific(self, quiet):
        field = make_field(1.0)
        result = draw(field, quiet, vectors=self.vectors(field, 2.5e-5, key_units="m s-1"))
        assert key_label(result) == r"$2.5\times10^{-5}$ m s-1"

    def test_mid_magnitude_is_plain(self, quiet):
        field = make_field(1.0)
        result = draw(field, quiet, vectors=self.vectors(field, 20.0, key_units="m s-1"))
        assert key_label(result) == "20 m s-1"

    def test_inherits_colorbar_style_limits(self, quiet):
        field = make_field(1.0)
        vectors = self.vectors(field, 5000.0)
        assert key_label(draw(field, quiet, vectors=vectors)) == r"$5\times10^{3}$"
        result = draw(
            field,
            quiet,
            vectors=vectors,
            colorbar_style=ColorbarStyle(powerlimits=(-3, 6)),
        )
        assert key_label(result) == "5000"

    def test_vector_limits_override_colorbar_style(self, quiet):
        field = make_field(1.0)
        result = draw(
            field,
            quiet,
            vectors=self.vectors(field, 5000.0, powerlimits=(-3, 3)),
            colorbar_style=ColorbarStyle(powerlimits=(-3, 6)),
        )
        assert key_label(result) == r"$5\times10^{3}$"

    def test_adder_quiver_inherits_plot_limits(self, quiet):
        field = make_field(1.0)
        result = draw(field, quiet, colorbar_style=ColorbarStyle(powerlimits=(-3, 6)))
        u = xr.full_like(field, 5000.0)
        result.add.quiver(u, xr.zeros_like(field), key_magnitude=5000.0, key_units="")
        assert key_label(result) == "5000"

    def test_colorbar_and_key_switch_together(self, quiet):
        field = make_field(1e4)
        vectors = self.vectors(field, 1e4)
        scientific = draw(field, quiet, vectors=vectors)
        assert offset_text(scientific) != ""
        assert "times" in key_label(scientific)
        plain = draw(
            field,
            quiet,
            vectors=vectors,
            colorbar_style=ColorbarStyle(powerlimits=(-3, 6)),
        )
        assert offset_text(plain) == ""
        assert "times" not in key_label(plain)
