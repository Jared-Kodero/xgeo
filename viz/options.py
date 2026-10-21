"""Option groups shared by ``GeoPlot``, ``Animate``, their wrappers and accessors.

Frequently changed plotting arguments (``cmap``, ``vmin``, ``projection``, ...)
remain ordinary keywords. Rarely changed settings are grouped into small frozen
dataclasses so that a signature stays short, defaults live in one place, and an
IDE can complete the fields::

    da.xgeo.geoplot(hillshade=True)
    da.xgeo.geoplot(hillshade=Hillshade(azimuth=270, vert_exag=2))
    da.xgeo.geoplot(features=MapFeatures(states=False, lakes=True))
    da.xgeo.geoplot(vectors=VectorField(u=u, v=v, color="k", subsample=(4, 4)))

Grouped parameters accept only an option instance, ``None`` and, where a
default group exists, a bool. Plain dictionaries are not accepted. All fields
are keyword-only, named and typed; no class accepts open-ended keyword
arguments.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from numbers import Integral
from typing import TYPE_CHECKING, Any, Literal, Self

import xarray as xr

if TYPE_CHECKING:
    from matplotlib.colors import Colormap

__all__ = [
    "ColorbarStyle",
    "ContourLabels",
    "Hillshade",
    "MapFeatures",
    "MapTicks",
    "ResolvedOptions",
    "Significance",
    "VectorField",
    "resolve_options",
]


class _Option:
    """Shared construction logic for option dataclasses."""

    __slots__ = ()

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
        """Return the fields that are not ``None`` as keyword arguments.

        ``None`` marks "use the callee default", so omitting it keeps the
        callee in control of every unset value. ``exclude`` drops fields that
        are passed separately.
        """
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name not in exclude and getattr(self, field.name) is not None
        }


def _check_powerlimits(value: Sequence[int]) -> None:
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
class Hillshade(_Option):
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
    hillshade_only: bool = False
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
class ContourLabels(_Option):
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
class MapTicks(_Option):
    """Longitude and latitude tick settings.

    Fields left as ``None`` use the default given below.

    Parameters
    ----------
    xbins, ybins : int, default 5
        Maximum number of longitude and latitude tick intervals.
    direction : {"out", "in", "inout"}, optional
        Tick-mark direction. Default ``"out"``.
    minor_ticks : bool, optional
        Show minor ticks. Default ``False``.
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
    """

    xbins: int = 5
    ybins: int = 5
    direction: Literal["out", "in", "inout"] | None = None
    minor_ticks: bool | None = None
    degree_symbol: str | None = None
    direction_label: bool | None = None
    top_labels: bool | None = None
    right_labels: bool | None = None
    gridlines: bool | None = None

    def __post_init__(self) -> None:
        if self.direction is not None and self.direction not in (
            "out",
            "in",
            "inout",
        ):
            raise ValueError("direction must be 'out', 'in' or 'inout'")

    def tick_arguments(self) -> dict[str, Any]:
        """Return the keyword arguments of :func:`~xgeo.viz.primitives.add_xy_ticks`.

        ``xbins`` and ``ybins`` become ``xticks_bins`` and ``yticks_bins``; unset
        fields are omitted so that the callee defaults apply.
        """
        arguments = self.arguments(exclude=("xbins", "ybins"))
        arguments["xticks_bins"] = self.xbins
        arguments["yticks_bins"] = self.ybins
        return arguments


@dataclass(frozen=True, kw_only=True, slots=True)
class ColorbarStyle(_Option):
    """Colorbar styling beyond ``colorbar``, ``colorbar_label`` and ``colorbar_orientation``.

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
    """

    drawedges: bool = True
    minimal_ticks: bool = True
    scalar_formatter: bool = True
    powerlimits: tuple[int, int] = (-3, 3)
    ticks: Sequence[float] | None = None
    tick_labels: Sequence[str] | None = None
    thickness: float | None = None
    pad: float | None = None
    pad_with_labels: float | None = None
    facet_length: float | None = None
    minimal_ticks_threshold: int | None = None

    def __post_init__(self) -> None:
        _check_powerlimits(self.powerlimits)
        for name in ("thickness", "pad", "pad_with_labels"):
            value = getattr(self, name)
            if value is not None and not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"{name} must be non-negative and finite")
        if self.thickness is not None and self.thickness == 0.0:
            raise ValueError("thickness must be positive")
        if self.facet_length is not None and not 0.0 < self.facet_length <= 1.0:
            raise ValueError("facet_length must lie in (0, 1]")
        if self.minimal_ticks_threshold is not None and (
            isinstance(self.minimal_ticks_threshold, bool)
            or not isinstance(self.minimal_ticks_threshold, Integral)
            or self.minimal_ticks_threshold < 1
        ):
            raise ValueError("minimal_ticks_threshold must be a positive integer")

    def advanced_arguments(self) -> dict[str, Any]:
        """Return the set layout and thinning fields as keyword arguments.

        These are the fields that :func:`~xgeo.viz.primitives.colorbar` accepts
        beyond the common ones. Unset fields are omitted so that the callee
        defaults apply.
        """
        return {
            name: getattr(self, name)
            for name in (
                "thickness",
                "pad",
                "pad_with_labels",
                "facet_length",
                "minimal_ticks_threshold",
            )
            if getattr(self, name) is not None
        }


@dataclass(frozen=True, kw_only=True, slots=True)
class MapFeatures(_Option):
    """Cartopy map features, extent and grid outline.

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
        Coastline color. Default is Cartopy's black.
    coastline_linewidth : float, optional
        Coastline width in points. Default is the Matplotlib line width.
    states_alpha, borders_alpha : float, optional
        Opacity of the state and country boundaries, in [0, 1]. Default 0.3.
    boundary_linestyle : str, optional
        Line style of the state and country boundaries. Default ``"-"``.
    boundary_color : str, optional
        Color of the state and country boundaries. Default is Cartopy's black.
    boundary_linewidth : float, optional
        Width of the state and country boundaries in points. Default is the
        Matplotlib line width.
    boundary_zorder : float, optional
        Drawing order of the state and country boundaries. Default 3.
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

    coastlines: bool = True
    borders: bool = True
    states: bool = True
    ocean: bool = True
    land: bool = True
    lakes: bool = False
    rivers: bool = False
    global_extent: bool = False
    extent: tuple[float, float, float, float] | None = None
    grid_bounds: bool = False
    resolution: Literal["110m", "50m", "10m"] | None = None
    coastline_color: str | None = None
    coastline_linewidth: float | None = None
    states_alpha: float | None = None
    borders_alpha: float | None = None
    boundary_linestyle: str | None = None
    boundary_color: str | None = None
    boundary_linewidth: float | None = None
    boundary_zorder: float | None = None
    water_color: str | None = None
    water_zorder: float | None = None
    land_color: str | None = None
    ocean_color: str | None = None
    background_zorder: float | None = None
    grid_bounds_linewidth: float | None = None
    grid_bounds_color: str | None = None
    grid_bounds_zorder: float | None = None

    def __post_init__(self) -> None:
        if self.extent is not None and len(self.extent) != 4:
            raise ValueError("extent must be (lon_min, lon_max, lat_min, lat_max)")
        if self.resolution is not None and self.resolution not in (
            "110m",
            "50m",
            "10m",
        ):
            raise ValueError("resolution must be '110m', '50m' or '10m'")
        for name in ("states_alpha", "borders_alpha"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be None or lie in [0, 1]")
        for name in (
            "coastline_linewidth",
            "boundary_linewidth",
            "grid_bounds_linewidth",
        ):
            value = getattr(self, name)
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
            "linewidth": self.grid_bounds_linewidth,
            "color": self.grid_bounds_color,
            "zorder": self.grid_bounds_zorder,
        }
        return {name: value for name, value in names.items() if value is not None}


@dataclass(frozen=True, kw_only=True, slots=True, eq=False)
class Significance(_Option):
    """Pointwise significance markers.

    Parameters
    ----------
    pvalues : xarray.DataArray
        Pointwise p-values on the plotted grid.
    level : float or None, default 0.05
        Markers are drawn where ``pvalues`` is below this threshold. ``None``
        marks every point.
    color : str, default "grey"
        Marker face color.
    alpha : float, default 0.3
        Marker opacity, in [0, 1].
    marker : str, optional
        Matplotlib marker style.
    edgecolors : str, optional
        Marker-edge color.
    subsample : int or tuple of int, default (1, 1)
        Spatial stride used to thin the markers, as ``(x, y)`` or one value for
        both directions.
    size : float, default 0.25
        Marker area in points squared.
    zorder : float, default 3.0
        Drawing order.
    """

    pvalues: xr.DataArray
    level: float | None = 0.05
    color: str = "grey"
    alpha: float = 0.3
    marker: str | None = None
    edgecolors: str | None = None
    subsample: int | tuple[int, int] = (1, 1)
    size: float = 0.25
    zorder: float = 3.0

    def __post_init__(self) -> None:
        if not isinstance(self.pvalues, xr.DataArray):
            raise TypeError("pvalues must be an xarray.DataArray")
        if self.level is not None and not 0.0 <= self.level <= 1.0:
            raise ValueError("level must be None or lie in [0, 1]")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")


@dataclass(frozen=True, kw_only=True, slots=True, eq=False)
class VectorField(_Option):
    """Vector components drawn as a quiver layer.

    Parameters
    ----------
    u, v : xarray.DataArray
        Zonal and meridional components. For animations they must contain the
        animation dimension.
    subsample : int or tuple of int, default (1, 1)
        Spatial stride used to thin the arrows, as ``(x, y)`` or one value for
        both directions.
    add_key : bool, default True
        Add one shared reference key.
    key_magnitude : float, optional
        Reference magnitude of the key. Chosen from the data when omitted.
    key_units : str, optional
        Units appended to the key label.
    powerlimits : tuple of int, optional
        Decimal exponents ``(lower, upper)`` outside which the key magnitude is
        written in scientific notation, formatted by the same
        ``ScalarFormatter`` as the colorbar. Defaults to
        ``ColorbarStyle.powerlimits`` (``(-3, 3)`` unless changed).
    scale : float, optional
        Data units per arrow length unit (see ``scale_units``). Larger values
        give shorter arrows.
    scale_units : {"width", "height", "dots", "inches", "x", "y", "xy"}, optional
        Unit of arrow length for ``scale``.
    units : {"width", "height", "dots", "inches", "x", "y", "xy"}, optional
        Unit of the arrow width and head dimensions.
    color : str, optional
        Arrow color.
    width : float, optional
        Arrow-shaft width in ``units``.
    headwidth, headlength, headaxislength : float, optional
        Arrow-head width, length and axis length in multiples of the shaft
        width.
    minshaft, minlength : float, optional
        Length below which an arrow is drawn with a shrunken head, and below
        which it is drawn as a dot, in multiples of the head length and shaft
        width.
    pivot : {"tail", "mid", "middle", "tip"}, optional
        Arrow anchor on its grid point.
    alpha : float, optional
        Arrow opacity, in [0, 1].
    zorder : float, default 4.0
        Drawing order.
    angles : {"uv", "xy"}, optional
        Arrow direction convention of :meth:`matplotlib.axes.Axes.quiver`.
        ``"xy"`` points arrows from each grid point toward the next in data
        coordinates, which keeps them tangent to the map on a projected axis.
        Default ``"xy"``.
    key_fontsize : float, optional
        Font size of the quiver-key label in points. Default 10.
    key_gap : float, optional
        Gap in points between the quiver key and the lowest axis decoration
        above it. Default 4.
    """

    u: xr.DataArray
    v: xr.DataArray
    subsample: int | tuple[int, int] = (1, 1)
    add_key: bool = True
    key_magnitude: float | None = None
    key_units: str | None = None
    powerlimits: tuple[int, int] | None = None
    scale: float | None = None
    scale_units: Literal["width", "height", "dots", "inches", "x", "y", "xy"] | None = (
        None
    )
    units: Literal["width", "height", "dots", "inches", "x", "y", "xy"] | None = None
    color: str | None = None
    width: float | None = None
    headwidth: float | None = None
    headlength: float | None = None
    headaxislength: float | None = None
    minshaft: float | None = None
    minlength: float | None = None
    pivot: Literal["tail", "mid", "middle", "tip"] | None = None
    alpha: float | None = None
    zorder: float = 4.0
    angles: Literal["uv", "xy"] | None = None
    key_fontsize: float | None = None
    key_gap: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.u, xr.DataArray) or not isinstance(self.v, xr.DataArray):
            raise TypeError("u and v must be xarray.DataArray objects")
        if self.alpha is not None and not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        if self.powerlimits is not None:
            _check_powerlimits(self.powerlimits)
        if self.angles is not None and self.angles not in ("uv", "xy"):
            raise ValueError("angles must be 'uv' or 'xy'")
        if self.key_fontsize is not None and not (
            math.isfinite(self.key_fontsize) and self.key_fontsize > 0.0
        ):
            raise ValueError("key_fontsize must be positive and finite")
        if self.key_gap is not None and not (
            math.isfinite(self.key_gap) and self.key_gap >= 0.0
        ):
            raise ValueError("key_gap must be non-negative and finite")

    def style(self) -> dict[str, Any]:
        """Return every non-``None`` field except the components ``u`` and ``v``."""
        return self.arguments(exclude=("u", "v"))


@dataclass(frozen=True, slots=True, eq=False)
class ResolvedOptions:
    """Grouped options after coercion. ``None`` marks a disabled optional group."""

    colorbar_style: ColorbarStyle
    features: MapFeatures
    hillshade: Hillshade | None
    contour_labels: ContourLabels | None
    map_ticks: MapTicks | None
    significance: Significance | None
    vectors: VectorField | None


def resolve_options(
    *,
    method: Literal[
        "default", "pcolormesh", "contourf", "contour", "imshow", "scatter"
    ],
    colorbar_style: ColorbarStyle | None,
    features: MapFeatures | None,
    hillshade: bool | Hillshade,
    clabel: bool | ContourLabels,
    map_ticks: bool | MapTicks,
    significance: Significance | None,
    vectors: VectorField | None,
) -> ResolvedOptions:
    """Coerce every grouped option of a plot call.

    ``colorbar_style`` and ``features`` fall back to their defaults. The other
    groups are ``None`` when disabled. Line contours are always labelled, with
    the label settings if given.
    """
    contour_labels = ContourLabels.coerce(clabel)
    if contour_labels is None and method == "contour":
        contour_labels = ContourLabels()
    return ResolvedOptions(
        colorbar_style=ColorbarStyle.coerce(colorbar_style) or ColorbarStyle(),
        features=MapFeatures.coerce(features) or MapFeatures(),
        hillshade=Hillshade.coerce(hillshade),
        contour_labels=contour_labels,
        map_ticks=MapTicks.coerce(map_ticks),
        significance=Significance.coerce(significance),
        vectors=VectorField.coerce(vectors),
    )
