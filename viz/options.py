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
    field_alpha : float or None, default 0.4
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
    """

    vert_exag: float = 1.0
    azimuth: float = 315.0
    altitude: float = 45.0
    alpha: float = 1.0
    field_alpha: float | None = 0.4
    cmap: str | Colormap = "gray"
    spacing: Literal["auto", "pixel"] | tuple[float, float] = "auto"
    hillshade_only: bool = False

    def __post_init__(self) -> None:
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

    Parameters
    ----------
    xbins, ybins : int, default 5
        Maximum number of longitude and latitude tick intervals.
    """

    xbins: int = 5
    ybins: int = 5


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
    ticks : sequence of float, optional
        Explicit tick locations.
    tick_labels : sequence of str, optional
        Labels for ``ticks``.
    """

    drawedges: bool = True
    minimal_ticks: bool = True
    scalar_formatter: bool = True
    ticks: Sequence[float] | None = None
    tick_labels: Sequence[str] | None = None


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

    def __post_init__(self) -> None:
        if self.extent is not None and len(self.extent) != 4:
            raise ValueError("extent must be (lon_min, lon_max, lat_min, lat_max)")


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
    powerlimits : tuple of int, default (-3, 3)
        Decimal exponents outside which the key magnitude is written in
        scientific notation.
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
    """

    u: xr.DataArray
    v: xr.DataArray
    subsample: int | tuple[int, int] = (1, 1)
    add_key: bool = True
    key_magnitude: float | None = None
    key_units: str | None = None
    powerlimits: tuple[int, int] = (-3, 3)
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

    def __post_init__(self) -> None:
        if not isinstance(self.u, xr.DataArray) or not isinstance(self.v, xr.DataArray):
            raise TypeError("u and v must be xarray.DataArray objects")
        if self.alpha is not None and not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")

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
