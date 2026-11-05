"""Option groups shared by ``GeoPlot``, ``Animate``, their wrappers and accessors.

Frequently changed plotting arguments (``cmap``, ``vmin``, ``projection``, ...)
remain ordinary keywords. Rarely changed settings are grouped into small frozen
dataclasses so that a signature stays short, defaults live in one place, and an
IDE can complete the fields::

    da.xgeo.geoplot(hillshade=True)
    da.xgeo.geoplot(hillshade=Hillshade(azimuth=270, vert_exag=2))
    da.xgeo.geoplot(features=MapFeatures(states=False, lakes=True))
    da.xgeo.geoplot(hillshade=True, clabel=True)

Grouped parameters accept only an option instance, ``None`` and, where a
default group exists, a bool. Plain dictionaries are not accepted. All fields
are keyword-only, named and typed; no class accepts open-ended keyword
arguments.

Fields of :class:`MapFeatures`, :class:`MapTicks` and :class:`ColorbarStyle`
are declared with ``option(default)``: they start as ``UNSET`` and take
``default`` only if no layer sets them. Resolution layers them, later layers
winning: the declared defaults, the :class:`MapTheme` preset,
:func:`set_options`, and the keyword given in the plotting call. A field a user
does not set therefore never overrides the preset, while an explicit ``None``
does and selects the callee default.

Standing, style-only defaults are set once with :func:`set_options`, either
globally or for a ``with`` block::

    xgeo.plot.set_options(map_theme="ncl")
    with xgeo.plot.set_options(map_theme="gmt", features=MapFeatures(lakes=True)):
        xgeo.plot.geoplot(da)
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from numbers import Integral
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Self,
    TypedDict,
    Unpack,
    get_args,
)

if TYPE_CHECKING:
    from matplotlib.colors import Colormap

__all__ = [
    "ColorbarStyle",
    "ContourLabels",
    "Hillshade",
    "MapFeatures",
    "MapTheme",
    "MapThemeName",
    "MapTicks",
    "Options",
    "ResolvedOptions",
    "get_options",
    "map_rc",
    "resolve_options",
    "set_options",
]


class _Unset:
    """Marks a layered field that no layer has set."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSET"

    def __bool__(self) -> bool:
        return False

    def __reduce__(self) -> str:
        return "UNSET"


UNSET: Any = _Unset()


def option(default: Any = None) -> Any:
    """Declare a layered field: ``UNSET`` until given, ``default`` once resolved."""
    return field(default=UNSET, metadata={"default": default})


class Option:
    """Shared construction logic for option dataclasses."""

    __slots__ = ()

    @classmethod
    def defaults(cls) -> Self:
        """Return the instance with every layered field at its declared default."""
        return cls(
            **{
                f.name: f.metadata["default"]
                for f in fields(cls)
                if "default" in f.metadata
            }
        )

    def given(self) -> dict[str, Any]:
        """Return the fields that were set, including those set to ``None``."""
        return {
            f.name: value
            for f in fields(self)
            if (value := getattr(self, f.name)) is not UNSET
        }

    def over(self, other: Self | None) -> Self:
        """Return a copy with the fields that ``other`` sets applied on top."""
        return self if other is None else replace(self, **other.given())

    def get(self, name: str) -> Any:
        """Return a field, mapping ``UNSET`` to ``None``."""
        return None if (value := getattr(self, name)) is UNSET else value

    @classmethod
    def coerce(cls, value: bool | Self | None) -> Self | None:
        """Convert a user value to an option instance.

        ``None`` and ``False`` disable the group, ``True`` selects the defaults
        and an instance is returned unchanged. Any other value, including a
        mapping, raises ``TypeError``.
        """
        if value is None or value is False:
            return None
        if value is True:
            try:
                return cls()
            except TypeError as error:
                raise TypeError(
                    f"{cls.__name__} has required fields, so True is not accepted"
                ) from error
        if isinstance(value, cls):
            return value
        raise TypeError(
            f"expected a bool, {cls.__name__} or None, got {type(value).__name__}"
        )

    def arguments(self, *, exclude: Sequence[str] = ()) -> dict[str, Any]:
        """Return the fields that are neither ``None`` nor ``UNSET`` as keyword arguments.

        ``None`` marks "use the callee default", so omitting it keeps the
        callee in control of every unset value. ``exclude`` drops fields that
        are passed separately.
        """
        return {
            f.name: value
            for f in fields(self)
            if f.name not in exclude
            and (value := getattr(self, f.name)) is not None
            and value is not UNSET
        }


def check_powerlimits(value: Sequence[int]) -> None:
    """Raise ``ValueError`` unless ``value`` is ``(lower, upper)`` with lower <= upper.

    Both exponents must be integers; booleans are rejected because ``True`` would
    otherwise pass as the exponent 1.
    """
    if (
        isinstance(value, str)
        or not isinstance(value, Sequence)
        or len(value) != 2
        or not all(
            isinstance(item, Integral) and not isinstance(item, bool) for item in value
        )
    ):
        raise ValueError("powerlimits must be a pair of integers (lower, upper)")
    if value[0] > value[1]:
        raise ValueError("powerlimits must satisfy lower <= upper")


@dataclass(frozen=True, kw_only=True, slots=True)
class Hillshade(Option):
    """Grayscale relief drawn beneath a scalar field.

    Parameters
    ----------
    vert_exag : float, default 1.0
        Vertical exaggeration. With physical spacing (``spacing="auto"`` on a
        degree or metre grid) and elevation in metres, 1.0 is unexaggerated.
        Use a different value to compensate for other elevation units or to
        emphasize relief.
    azimuth : float, default 315.0
        Light direction in degrees clockwise from north, in [0, 360].
    altitude : float, default 45.0
        Light height in degrees above the horizon, in [0, 90].
    alpha : float, default 1.0
        Opacity of the grayscale layer, in [0, 1].
    field_alpha : float or None, default 0.65
        Opacity given to the plotted field unless ``alpha`` is passed explicitly.
        ``None`` leaves the field opacity unchanged.
    cmap : str or matplotlib.colors.Colormap, default "gray"
        Colormap of the relief layer, applied to illumination in [0, 1].
    spacing : {"auto", "pixel"} or tuple of float, default "auto"
        Horizontal cell spacing used for the surface gradient, in the same
        units as the elevation. ``"auto"`` converts a degree grid to metres
        (mean cell step, central latitude) and uses projected coordinates given
        in metres or kilometres as metres. Grids with other or unknown units
        fall back to ``"pixel"``, which uses unit spacing and therefore makes the
        shading depend on grid resolution. A tuple sets ``(dx, dy)`` directly.
    method : {"default", "pcolormesh", "imshow", "contourf"}, default "default"
        Plotting method of the relief layer. It is independent of the ``method``
        of the field drawn above it. ``"default"`` lets xarray choose, which
        gives a ``QuadMesh`` for a two-dimensional grid, the same as
        ``"pcolormesh"``. ``"imshow"`` requires a regular grid, and on a map
        whose projection is centred away from 0 degrees longitude it is placed
        correctly only when ``MapFeatures.extent`` is set (as for
        ``method="imshow"`` of the field itself). ``"contourf"`` draws banded
        relief; set ``levels`` to control the number of bands.
    levels : int or sequence of float, optional
        Illumination levels of the relief layer. Used only by
        ``method="contourf"``. Chosen by xarray when omitted.
    vmin, vmax : float, default 0.0, 1.0
        Illumination values mapped to the ends of ``cmap``. Illumination lies in
        [0, 1]; a narrower range stretches the contrast of the relief.
    fraction : float, default 1.0
        Contrast of the illumination, passed to
        :meth:`matplotlib.colors.LightSource.hillshade`. Values above one darken
        and increase the contrast of the shaded slopes, values below one brighten
        them.
    zorder_offset : float, default 0.1
        Amount by which the relief layer is drawn below the field. A positive
        value keeps the relief beneath the field.
    rasterized : bool, optional
        Rasterize the relief layer. The setting of the field is used when
        omitted.
    """

    vert_exag: float = 1.0
    azimuth: float = 315.0
    altitude: float = 45.0
    alpha: float = 1.0
    field_alpha: float | None = 0.65
    cmap: str | Colormap = "gray"
    spacing: Literal["auto", "pixel"] | tuple[float, float] = "auto"
    method: Literal["default", "pcolormesh", "imshow", "contourf"] = "default"
    levels: int | Sequence[float] | None = None
    vmin: float = 0.0
    vmax: float = 1.0
    fraction: float = 1.0
    zorder_offset: float = 0.1
    rasterized: bool | None = None

    def __post_init__(self) -> None:
        if self.method not in ("default", "pcolormesh", "imshow", "contourf"):
            raise ValueError(
                "method must be 'default', 'pcolormesh', 'imshow' or 'contourf'"
            )
        if not (math.isfinite(self.vmin) and math.isfinite(self.vmax)):
            raise ValueError("vmin and vmax must be finite")
        if self.vmin >= self.vmax:
            raise ValueError("vmin must be smaller than vmax")
        if not (math.isfinite(self.fraction) and self.fraction > 0.0):
            raise ValueError("fraction must be positive and finite")
        if not math.isfinite(self.zorder_offset):
            raise ValueError("zorder_offset must be finite")
        if not math.isfinite(self.vert_exag):
            raise ValueError("vert_exag must be finite")
        if not 0.0 <= self.azimuth <= 360.0:
            raise ValueError("azimuth must lie in [0, 360]")
        if not 0.0 <= self.altitude <= 90.0:
            raise ValueError("altitude must lie in [0, 90]")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        if self.field_alpha is not None and not 0.0 <= self.field_alpha <= 1.0:
            raise ValueError("field_alpha must be None or lie in [0, 1]")
        if isinstance(self.spacing, str):
            if self.spacing not in ("auto", "pixel"):
                raise ValueError("spacing must be 'auto', 'pixel' or (dx, dy)")
        else:
            if len(self.spacing) != 2 or not all(
                math.isfinite(step) and step > 0 for step in self.spacing
            ):
                raise ValueError("spacing must contain two positive finite values")


@dataclass(frozen=True, kw_only=True, slots=True)
class ContourLabels(Option):
    """Line-contour label settings.

    Parameters
    ----------
    fmt : str or mapping of float to str, default "%1.0f"
        Label format or per-level labels.
    fontsize : float, default 8.0
        Label font size.
    inline : bool, default True
        Break the contour line under each label.
    colors : str or sequence of str, optional
        Label colors.
    levels : sequence of float, optional
        Contour levels to label. All levels are labelled when omitted.
    inline_spacing : float, default 5.0
        Gap in points left on each side of a label when ``inline`` is true.
    rightside_up : bool, default True
        Rotate labels so that the text is never upside down.
    use_clabeltext : bool, default False
        Use ``ClabelText`` so that labels follow the line when the axes are
        redrawn with a different aspect ratio.
    manual : bool or sequence of (float, float), default False
        ``True`` places labels by mouse click. A sequence places them at the
        given data coordinates.
    zorder : float, optional
        Drawing order of the labels.
    """

    fmt: str | Mapping[float, str] = "%1.0f"
    fontsize: float = 8.0
    inline: bool = True
    colors: str | Sequence[str] | None = None
    levels: Sequence[float] | None = None
    inline_spacing: float = 5.0
    rightside_up: bool = True
    use_clabeltext: bool = False
    manual: bool | Sequence[tuple[float, float]] = False
    zorder: float | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class MapTicks(Option):
    """Longitude and latitude tick settings.

    Fields that are not given are unset: they take the :class:`MapTheme` preset
    value or, failing that, the default given below.

    Parameters
    ----------
    xbins, ybins : int, default 5
        Maximum number of longitude and latitude tick intervals.
    direction : {"out", "in", "inout"}, optional
        Tick-mark direction. Default ``"out"``.
    minor_ticks : bool, optional
        Show minor ticks. Default ``False``.
    xminor, yminor : int, optional
        Minor intervals per major interval; ``3`` gives two minor ticks between
        majors. Requires ``minor_ticks``.
    major_tick_length, minor_tick_length : float, optional
        Tick lengths in points. Use Matplotlib defaults when omitted.
    major_tick_width, minor_tick_width : float, optional
        Tick widths in points. Use Matplotlib defaults when omitted.
    top_ticks, right_ticks : bool, optional
        Enable tick marks on the top and right edges when provided.
    degree_symbol : str, optional
        Text appended to every tick label. Default ``""``, so labels are plain
        signed numbers.
    direction_label : bool, optional
        Append the hemisphere letter (N, S, E, W) to the labels instead of using
        signed numbers. Default ``False``.
    top_labels, right_labels : bool, optional
        Label the top and right edges as well as the bottom and left edges.
        Default ``False``.
    gridlines : bool, optional
        Draw grid lines through the ticks. Default ``False``.
    gridline_color : str, optional
        Grid-line color. Matplotlib's grid color when omitted.
    gridline_linestyle : str, optional
        Grid-line style, for example ``"--"`` or ``":"``. Matplotlib's grid
        style when omitted.
    gridline_linewidth : float, optional
        Grid-line width in points. Matplotlib's grid width when omitted.
    gridline_alpha : float, optional
        Grid-line opacity, in [0, 1]. Opaque when omitted.

    The ``gridline_*`` fields take effect only when ``gridlines`` is true.
    """

    xbins: int = option(5)
    ybins: int = option(5)
    direction: Literal["out", "in", "inout"] | None = option()
    minor_ticks: bool | None = option()
    xminor: int | None = option()
    yminor: int | None = option()
    major_tick_length: float | None = option()
    major_tick_width: float | None = option()
    minor_tick_length: float | None = option()
    minor_tick_width: float | None = option()
    top_ticks: bool | None = option()
    right_ticks: bool | None = option()
    degree_symbol: str | None = option()
    direction_label: bool | None = option()
    top_labels: bool | None = option()
    right_labels: bool | None = option()
    gridlines: bool | None = option()
    gridline_color: str | None = option()
    gridline_linestyle: str | None = option()
    gridline_linewidth: float | None = option()
    gridline_alpha: float | None = option()

    def __post_init__(self) -> None:
        if self.get("direction") not in (None, "out", "in", "inout"):
            raise ValueError("direction must be 'out', 'in' or 'inout'")
        for name in ("xbins", "ybins", "xminor", "yminor"):
            value = self.get(name)
            if value is not None and not (
                isinstance(value, Integral)
                and not isinstance(value, bool)
                and value > 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "major_tick_length",
            "major_tick_width",
            "minor_tick_length",
            "minor_tick_width",
        ):
            value = self.get(name)
            if value is not None and not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"{name} must be non-negative and finite")
        alpha = self.get("gridline_alpha")
        if alpha is not None and not 0.0 <= alpha <= 1.0:
            raise ValueError("gridline_alpha must be None or lie in [0, 1]")
        if (width := self.get("gridline_linewidth")) is not None and not (
            math.isfinite(width) and width >= 0.0
        ):
            raise ValueError("gridline_linewidth must be non-negative and finite")

    def tick_arguments(self) -> dict[str, Any]:
        """Return the keyword arguments of :func:`~xgeo.viz.primitives.add_xy_ticks`.

        ``xbins`` and ``ybins`` become ``xticks_bins`` and ``yticks_bins``; unset
        fields are omitted so that the callee defaults apply.
        """
        arguments = self.arguments(exclude=("xbins", "ybins"))
        arguments["xticks_bins"] = self.get("xbins")
        arguments["yticks_bins"] = self.get("ybins")
        return arguments


@dataclass(frozen=True, kw_only=True, slots=True)
class ColorbarStyle(Option):
    """Colorbar styling beyond ``colorbar``, ``colorbar_label`` and ``colorbar_orientation``.

    Fields that are not given are unset: they take the :class:`MapTheme` preset
    value or, failing that, the default given below.

    Parameters
    ----------
    drawedges : bool, default True
        Draw colorbar interval edges. Contour-filled plots always draw them.
    minimal_ticks : bool, default True
        Skip every other tick when the labels would be crowded.
    scalar_formatter : bool, default True
        Format automatic tick labels with a math-text ``ScalarFormatter``.
    powerlimits : tuple of int, default (-3, 3)
        Decimal exponents ``(lower, upper)`` for the ``ScalarFormatter``: tick
        labels whose order of magnitude is at most ``lower`` or at least
        ``upper`` use scientific notation. Also the default for the quiver key
        (``VectorField.powerlimits``), so both switch at the same magnitude.
    ticks : sequence of float, optional
        Explicit tick locations.
    tick_labels : sequence of str, optional
        Labels for ``ticks``.
    thickness : float, optional
        Thickness of an automatically placed colorbar, as a figure fraction on a
        5 by 4.8 inch figure (it scales with the figure size so the physical
        thickness stays constant). Default 0.03 for a vertical and 0.04 for a
        horizontal colorbar.
    pad : float, optional
        Gap between the axes and the colorbar, in the same units as
        ``thickness``. Default 0.04 for a vertical and 0.05 for a horizontal
        colorbar.
    pad_with_labels : float, optional
        Gap below the axes of a horizontal colorbar when tick labels or an
        x-axis label sit there. Default ``pad + 0.05``, which is 0.1 for the
        default ``pad``. Ignored for a vertical colorbar.
    facet_length : float, optional
        Length of the colorbar as a fraction of the panel grid when the grid has
        more than one row (vertical colorbar) or more than one column (horizontal
        colorbar), in (0, 1]. Default 0.5.
    minimal_ticks_threshold : int, optional
        Number of colour boundaries above which ``minimal_ticks`` thins the
        labels. Default 6.
    linewidth : float, optional
        Colorbar outline width in points; use Matplotlib defaults if omitted.
    edgecolor : str, optional
        Colorbar outline edge color; use Matplotlib defaults if omitted.
    major_tick_length, major_tick_width : float, optional
        Colorbar major tick dimensions in points; default to Matplotlib settings.
    ticks_visible : bool, optional
        ``False`` hides all colorbar tick marks, major and minor, and keeps the
        labels. Unchanged when omitted.
    """

    drawedges: bool = option(True)
    minimal_ticks: bool = option(True)
    scalar_formatter: bool = option(True)
    powerlimits: tuple[int, int] = option((-3, 3))
    ticks: Sequence[float] | None = option()
    tick_labels: Sequence[str] | None = option()
    thickness: float | None = option()
    pad: float | None = option()
    pad_with_labels: float | None = option()
    facet_length: float | None = option()
    minimal_ticks_threshold: int | None = option()
    linewidth: float | None = option()
    edgecolor: str | None = option()
    major_tick_length: float | None = option()
    major_tick_width: float | None = option()
    ticks_visible: bool | None = option()

    def __post_init__(self) -> None:
        if (limits := self.get("powerlimits")) is not None:
            check_powerlimits(limits)
        for name in (
            "linewidth",
            "major_tick_length",
            "major_tick_width",
        ):
            value = self.get(name)
            if value is not None and not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"{name} must be non-negative and finite")
        for name in ("thickness", "pad", "pad_with_labels"):
            value = self.get(name)
            if value is not None and not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"{name} must be non-negative and finite")
        if self.get("thickness") == 0.0:
            raise ValueError("thickness must be positive")
        if (length := self.get("facet_length")) is not None and not 0.0 < length <= 1.0:
            raise ValueError("facet_length must lie in (0, 1]")
        if (threshold := self.get("minimal_ticks_threshold")) is not None and (
            isinstance(threshold, bool)
            or not isinstance(threshold, Integral)
            or threshold < 1
        ):
            raise ValueError("minimal_ticks_threshold must be a positive integer")

    def advanced_arguments(self) -> dict[str, Any]:
        """Return the set layout and thinning fields as keyword arguments.

        These are the fields that :func:`~xgeo.viz.primitives.colorbar` accepts
        beyond the common ones. Unset fields are omitted so that the callee
        defaults apply.
        """
        return {
            name: value
            for name in (
                "thickness",
                "pad",
                "pad_with_labels",
                "facet_length",
                "minimal_ticks_threshold",
                "linewidth",
                "edgecolor",
                "major_tick_length",
                "major_tick_width",
                "ticks_visible",
            )
            if (value := self.get(name)) is not None
        }


@dataclass(frozen=True, kw_only=True, slots=True)
class MapFeatures(Option):
    """Cartopy map features, extent and grid outline.

    Fields that are not given are unset: they take the :class:`MapTheme` preset
    value or, failing that, the default given below.

    Parameters
    ----------
    coastlines, borders, states : bool, default True
        Boundary features.
    ocean, land : bool, default True
        Background fills.
    lakes, rivers : bool, default False
        Inland-water features.
    global_extent : bool, default False
        Use a global map extent.
    extent : tuple of float, optional
        Explicit extent ``(lon_min, lon_max, lat_min, lat_max)``.
    grid_bounds : bool, default False
        Outline the outer perimeter of the plotted grid domain.

    The remaining fields are advanced settings. Each one left as ``None`` keeps
    the default given below.

    resolution : {"110m", "50m", "10m"}, optional
        Natural Earth scale of every feature. Default ``"110m"``. Finer scales
        are downloaded by Cartopy on first use. Note that state boundaries at
        110m cover the United States only.
    coastline_color : str, optional
        Coastline color. Cartopy default when omitted.
    coastline_linewidth : float, optional
        Coastline width in points. Matplotlib default when omitted.
    edgecolor : str, optional
        Map frame edge color. Unchanged when omitted.
    linewidth : float, optional
        Map frame spine width in points. Unchanged when omitted.
    states_alpha, borders_alpha : float, optional
        Opacity of the state and country boundaries, in [0, 1]. Default 0.3.
    boundary_linestyle : str, optional
        Line style of the state and country boundaries. Default ``"-"``.
    boundary_color : str, optional
        Color of state and country boundaries. Cartopy default when omitted.
    boundary_linewidth : float, optional
        Width of state and country boundaries in points. Matplotlib default when omitted.
    boundary_zorder : float, optional
        Drawing order of the state and country boundaries. Default 3.
    states_color, borders_color : str, optional
        Color of the state or the country boundaries alone. Overrides
        ``boundary_color`` for that feature.
    states_linestyle, borders_linestyle : str, optional
        Line style of the state or the country boundaries alone. Overrides
        ``boundary_linestyle`` for that feature.
    states_linewidth, borders_linewidth : float, optional
        Width in points of the state or the country boundaries alone. Overrides
        ``boundary_linewidth`` for that feature.
    states_zorder, borders_zorder : float, optional
        Drawing order of the state or the country boundaries alone. Overrides
        ``boundary_zorder`` for that feature.
    coastline_alpha : float, optional
        Coastline opacity, in [0, 1]. Opaque when omitted.
    coastline_zorder : float, optional
        Drawing order of the coastlines. Cartopy's default is used when
        omitted.
    lakes_linewidth, rivers_linewidth : float, optional
        Outline width of the lakes and width of the rivers in points.
        Cartopy's default is used when omitted.
    water_color : str, optional
        Color of lakes and rivers. Default is Cartopy's water color.
    water_zorder : float, optional
        Drawing order of lakes and rivers. Default 2.
    land_color : str, optional
        Color of the land fill that masks the land when ``ocean`` is true and
        ``land`` is false. Default ``"#54585f"``.
    ocean_color : str, optional
        Color of the ocean fill used when ``land`` is true and ``ocean`` is
        false. Default is Cartopy's ocean color.
    background_zorder : float, optional
        Drawing order of the land and ocean fills. Default 0.
    grid_bounds_linewidth : float, optional
        Width of the ``grid_bounds`` outline in points. Default 1.
    grid_bounds_color : str, optional
        Color of the ``grid_bounds`` outline. Default ``"black"``.
    grid_bounds_zorder : float, optional
        Drawing order of the ``grid_bounds`` outline. Default 1.
    """

    coastlines: bool = option(True)
    borders: bool = option(True)
    states: bool = option(True)
    ocean: bool = option(True)
    land: bool = option(True)
    lakes: bool = option(False)
    rivers: bool = option(False)
    global_extent: bool = option(False)
    extent: tuple[float, float, float, float] | None = option()
    grid_bounds: bool = option(False)
    resolution: Literal["110m", "50m", "10m"] | None = option()
    coastline_color: str | None = option()
    coastline_linewidth: float | None = option()
    edgecolor: str | None = option()
    linewidth: float | None = option()
    states_alpha: float | None = option()
    borders_alpha: float | None = option()
    boundary_linestyle: str | None = option()
    boundary_color: str | None = option()
    boundary_linewidth: float | None = option()
    boundary_zorder: float | None = option()
    states_color: str | None = option()
    states_linestyle: str | None = option()
    states_linewidth: float | None = option()
    states_zorder: float | None = option()
    borders_color: str | None = option()
    borders_linestyle: str | None = option()
    borders_linewidth: float | None = option()
    borders_zorder: float | None = option()
    coastline_alpha: float | None = option()
    coastline_zorder: float | None = option()
    lakes_linewidth: float | None = option()
    rivers_linewidth: float | None = option()
    water_color: str | None = option()
    water_zorder: float | None = option()
    land_color: str | None = option()
    ocean_color: str | None = option()
    background_zorder: float | None = option()
    grid_bounds_linewidth: float | None = option()
    grid_bounds_color: str | None = option()
    grid_bounds_zorder: float | None = option()

    def __post_init__(self) -> None:
        if (extent := self.get("extent")) is not None and len(extent) != 4:
            raise ValueError("extent must be (lon_min, lon_max, lat_min, lat_max)")
        if self.get("resolution") not in (None, "110m", "50m", "10m"):
            raise ValueError("resolution must be '110m', '50m' or '10m'")
        for name in ("states_alpha", "borders_alpha", "coastline_alpha"):
            value = self.get(name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be None or lie in [0, 1]")
        for name in (
            "coastline_linewidth",
            "linewidth",
            "boundary_linewidth",
            "states_linewidth",
            "borders_linewidth",
            "lakes_linewidth",
            "rivers_linewidth",
            "grid_bounds_linewidth",
        ):
            value = self.get(name)
            if value is not None and not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"{name} must be non-negative and finite")

    _GRID_BOUNDS = (
        "grid_bounds",
        "grid_bounds_linewidth",
        "grid_bounds_color",
        "grid_bounds_zorder",
    )

    def feature_arguments(self) -> dict[str, Any]:
        """Return the keyword arguments of :func:`~xgeo.viz.primitives.add_map_features`.

        The extent (passed as ``set_extent``) and the grid-boundary fields are
        excluded, and unset fields are omitted so that the callee defaults apply.
        """
        return self.arguments(exclude=("extent", *self._GRID_BOUNDS))

    def boundary_arguments(self) -> dict[str, Any]:
        """Return the set fields of the grid-boundary outline.

        The keys are ``linewidth``, ``color`` and ``zorder``, the keyword
        arguments of :func:`~xgeo.viz.primitives.add_grid_boundary`.
        """
        names = {
            "linewidth": self.get("grid_bounds_linewidth"),
            "color": self.get("grid_bounds_color"),
            "zorder": self.get("grid_bounds_zorder"),
        }
        return {name: value for name, value in names.items() if value is not None}


type MapThemeName = Literal["default", "ncl", "ferret", "grads", "gmt"]
_HELVETICA = {"font.family": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"]}


@dataclass(frozen=True, slots=True)
class MapTheme:
    """Cartopy-only feature, tick and colorbar style presets.

    Explicit :class:`MapFeatures`, :class:`MapTicks`, and :class:`ColorbarStyle`
    values override preset fields. ``rc`` holds Matplotlib rc parameters that are
    applied only while a plot is built and drawn, then restored. No font is
    downloaded or installed: ``font.family`` is a fallback list that Matplotlib
    resolves against the installed fonts. While a plot is built the preset's
    ``rc`` overrides a global :func:`xgeo.plot.theme`, and ``set_options(rc=...)``
    overrides the preset. The colorbar remains vertical unless set separately.
    """

    features: MapFeatures
    ticks: MapTicks | None
    colorbar: ColorbarStyle
    rc: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def default(cls) -> Self:
        """Preserve the original plotting appearance, without colorbar ticks."""
        return cls(MapFeatures(), None, ColorbarStyle(ticks_visible=False))

    @classmethod
    def ncl(cls) -> Self:
        """Use fine gray coastlines and GeoCAT-viz style ticks on all four sides."""
        return cls(
            MapFeatures(
                linewidth=0.9,
                edgecolor="0.2",
                coastline_linewidth=0.55,
                coastline_color="0.35",
                boundary_linewidth=0.5,
                boundary_color="0.4",
            ),
            MapTicks(
                direction="out",
                minor_ticks=True,
                xminor=3,
                yminor=3,
                major_tick_length=8.0,
                major_tick_width=0.9,
                minor_tick_length=4.0,
                minor_tick_width=0.4,
                top_ticks=True,
                right_ticks=True,
                degree_symbol="°",
                direction_label=True,
            ),
            ColorbarStyle(
                linewidth=0.7,
                edgecolor="0.2",
                ticks_visible=False,
            ),
            rc=_HELVETICA,
        )

    @classmethod
    def ferret(cls) -> Self:
        """Use Ferret-inspired geographic axes and simple coastal outlines."""
        return cls(
            MapFeatures(
                linewidth=0.8,
                edgecolor="0.2",
                coastline_linewidth=0.7,
                coastline_color="0.2",
                borders=False,
                states=False,
            ),
            MapTicks(
                direction="out",
                minor_ticks=False,
                major_tick_length=5.0,
                major_tick_width=0.75,
                top_ticks=False,
                right_ticks=False,
                degree_symbol="°",
                direction_label=True,
            ),
            ColorbarStyle(
                linewidth=0.6,
                edgecolor="0.2",
                ticks_visible=False,
            ),
        )

    @classmethod
    def grads(cls) -> Self:
        """Use GrADS-inspired meteorological map outlines and axis ticks."""
        return cls(
            MapFeatures(
                linewidth=1.0,
                edgecolor="0.1",
                coastline_linewidth=0.85,
                coastline_color="0.15",
                boundary_linewidth=0.5,
                boundary_color="0.4",
                states=False,
            ),
            MapTicks(
                direction="out",
                minor_ticks=False,
                major_tick_length=4.5,
                major_tick_width=0.9,
                top_ticks=False,
                right_ticks=False,
                degree_symbol="°",
                direction_label=True,
            ),
            ColorbarStyle(
                linewidth=0.8,
                edgecolor="0.1",
                ticks_visible=False,
            ),
        )

    @classmethod
    def gmt(cls) -> Self:
        """Use GMT-inspired four-sided geographic annotations and ticks."""
        return cls(
            MapFeatures(
                linewidth=1.2,
                edgecolor="0.1",
                coastline_linewidth=0.7,
                coastline_color="0.1",
                borders=False,
                states=False,
            ),
            MapTicks(
                direction="out",
                minor_ticks=True,
                major_tick_length=6.0,
                major_tick_width=0.9,
                minor_tick_length=3.0,
                minor_tick_width=0.6,
                top_ticks=True,
                right_ticks=True,
                top_labels=True,
                right_labels=True,
                degree_symbol="°",
                direction_label=True,
            ),
            ColorbarStyle(
                linewidth=0.8,
                edgecolor="0.1",
                ticks_visible=True,
            ),
            rc=_HELVETICA,
        )

    @classmethod
    def from_name(cls, name: MapThemeName) -> Self:
        """Build a preset by name, rejecting unknown strings at runtime."""
        match name:
            case "default":
                return cls.default()
            case "ncl":
                return cls.ncl()
            case "ferret":
                return cls.ferret()
            case "grads":
                return cls.grads()
            case "gmt":
                return cls.gmt()
            case _:
                raise ValueError(f"unknown map_theme: {name!r}")


class Options(TypedDict, total=False):
    """Keys accepted by :func:`set_options`."""

    map_theme: MapThemeName | MapTheme
    features: MapFeatures | None
    colorbar_style: ColorbarStyle | None
    map_ticks: MapTicks | None
    rc: Mapping[str, Any]


OPTIONS: dict[str, Any] = {
    "map_theme": "default",
    "features": None,
    "colorbar_style": None,
    "map_ticks": None,
    "rc": {},
}

_VALIDATORS: dict[str, Any] = {
    "map_theme": lambda value: (
        isinstance(value, MapTheme) or value in get_args(MapThemeName.__value__)
    ),
    "features": lambda value: value is None or isinstance(value, MapFeatures),
    "colorbar_style": lambda value: value is None or isinstance(value, ColorbarStyle),
    "map_ticks": lambda value: value is None or isinstance(value, MapTicks),
    "rc": lambda value: isinstance(value, Mapping),
}


class set_options:
    """Set standing plot options, globally or for a ``with`` block.

    The options are style defaults that are independent of the data. A keyword
    given in a plotting call overrides them; they override the theme preset.
    Options are read when a plot is created, so a block only affects plots
    created inside it. Used as a function, the new values stay in force until
    changed again.

    Parameters
    ----------
    map_theme : {"default", "ncl", "ferret", "grads", "gmt"} or MapTheme
        Style preset. Change a preset's font or any other rc parameter with
        ``rc`` rather than building a new :class:`MapTheme`.
    features : MapFeatures or None
        Default map features, applied over the preset.
    colorbar_style : ColorbarStyle or None
        Default colorbar styling, applied over the preset.
    map_ticks : MapTicks or None
        Default geographic ticks, applied over the preset. ``None`` keeps the
        preset's ticks.
    rc : mapping
        Matplotlib rc parameters, including the font, applied only while a plot
        is built, drawn or extended, then restored. They override the preset's
        ``rc``, which in turn overrides a global :func:`xgeo.plot.theme`.

    Examples
    --------
    >>> xgeo.plot.set_options(map_theme="ncl")
    >>> with xgeo.plot.set_options(map_theme="ncl", rc={"font.family": "serif"}):
    ...     xgeo.plot.geoplot(da)
    """

    def __init__(self, **options: Unpack[Options]) -> None:
        unknown = options.keys() - OPTIONS.keys()
        if unknown:
            raise ValueError(
                f"unknown option(s) {sorted(unknown)}; valid options are "
                f"{sorted(OPTIONS)}"
            )
        for key, value in options.items():
            if not _VALIDATORS[key](value):
                raise ValueError(f"invalid value for option {key!r}: {value!r}")
        self.old = {key: OPTIONS[key] for key in options}
        OPTIONS.update(options)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        OPTIONS.update(self.old)


def get_options() -> dict[str, Any]:
    """Return a copy of the options currently set by :func:`set_options`."""
    return dict(OPTIONS)


def map_rc() -> dict[str, Any]:
    """Return the rc parameters of the current theme, then ``set_options(rc=...)``.

    A ``font.family`` list is cut to the families that are installed, so a
    preset's preference list never makes Matplotlib warn about a missing font.
    """
    theme = OPTIONS["map_theme"]
    theme = MapTheme.from_name(theme) if isinstance(theme, str) else theme
    rc = {**theme.rc, **OPTIONS["rc"]}
    family = rc.get("font.family")
    if isinstance(family, list):
        from matplotlib import font_manager

        available = {font.name for font in font_manager.fontManager.ttflist}
        generic = {"serif", "sans-serif", "monospace", "cursive", "fantasy"}
        rc["font.family"] = [
            name for name in family if name in available or name in generic
        ]
    return rc


@dataclass(frozen=True, slots=True, eq=False)
class ResolvedOptions:
    """Grouped options after layering. ``None`` marks a disabled optional group."""

    colorbar_style: ColorbarStyle
    features: MapFeatures
    hillshade: Hillshade | None
    contour_labels: ContourLabels | None
    map_ticks: MapTicks | None
    rc: Mapping[str, Any]


def resolve_options(
    *,
    method: Literal[
        "default", "pcolormesh", "contourf", "contour", "imshow", "scatter"
    ],
    colorbar_style: ColorbarStyle | None,
    features: MapFeatures | None,
    hillshade: bool | Hillshade,
    clabel: bool | ContourLabels,
    map_ticks: bool | MapTicks | None,
    cbar_ticks: bool | None,
) -> ResolvedOptions:
    """Layer class defaults, the theme preset, :func:`set_options` and the call.

    Later layers win. ``map_ticks=None`` inherits the preset and the
    ``set_options`` ticks, ``False`` disables geographic ticks and ``True``
    enables them with the preset's settings or, for the default theme, the
    :class:`MapTicks` defaults.
    """
    theme = OPTIONS["map_theme"]
    preset = MapTheme.from_name(theme) if isinstance(theme, str) else theme
    resolved_features = (
        MapFeatures.defaults()
        .over(preset.features)
        .over(OPTIONS["features"])
        .over(features)
    )
    resolved_colorbar = (
        ColorbarStyle.defaults()
        .over(preset.colorbar)
        .over(OPTIONS["colorbar_style"])
        .over(colorbar_style)
    )
    if cbar_ticks is not None:
        resolved_colorbar = replace(resolved_colorbar, ticks_visible=cbar_ticks)
    inherited_ticks = (
        None
        if preset.ticks is None and OPTIONS["map_ticks"] is None
        else MapTicks.defaults().over(preset.ticks).over(OPTIONS["map_ticks"])
    )
    if map_ticks is False:
        resolved_ticks = None
    elif map_ticks is None:
        resolved_ticks = inherited_ticks
    else:
        resolved_ticks = (inherited_ticks or MapTicks.defaults()).over(
            MapTicks.coerce(map_ticks)
        )

    contour_labels = ContourLabels.coerce(clabel)
    if contour_labels is None and method == "contour":
        contour_labels = ContourLabels()
    return ResolvedOptions(
        colorbar_style=resolved_colorbar,
        features=resolved_features,
        hillshade=Hillshade.coerce(hillshade),
        contour_labels=contour_labels,
        map_ticks=resolved_ticks,
        rc=map_rc(),
    )
