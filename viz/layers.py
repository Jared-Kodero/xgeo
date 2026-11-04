"""Data layers drawn over a map: vector components and significance masks.

Unlike the style groups in :mod:`xgeo.viz.options`, these classes carry data
(``u`` and ``v``, ``pvalues``). They are public as ``xgeo.plot.VectorField`` and
``xgeo.plot.Significance`` and are passed to ``geoplot`` and ``animate``::

    xgeo.plot.geoplot(da, vectors=xgeo.plot.VectorField(u=u, v=v, subsample=(4, 4)))
    xgeo.plot.geoplot(da, significance=xgeo.plot.Significance(pvalues=p, hatch="///"))
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import xarray as xr

from .options import Option, check_powerlimits

if TYPE_CHECKING:
    from matplotlib.colors import Colormap

__all__ = ["Significance", "VectorField"]


@dataclass(frozen=True, kw_only=True, slots=True, eq=False)
class Significance(Option):
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
    hatch : str, optional
        Matplotlib hatch pattern, for example ``"///"`` or ``".."``. The
        significant region is then filled with hatching, drawn in ``color`` (or
        ``edgecolors`` when given) with the opacity ``alpha``, instead of
        markers. ``marker`` and ``size`` are ignored.
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
    hatch: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.pvalues, xr.DataArray):
            raise TypeError("pvalues must be an xarray.DataArray")
        if self.hatch is not None and not isinstance(self.hatch, str):
            raise TypeError("hatch must be a string")
        if self.level is not None and not 0.0 <= self.level <= 1.0:
            raise ValueError("level must be None or lie in [0, 1]")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")


@dataclass(frozen=True, kw_only=True, slots=True, eq=False)
class VectorField(Option):
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
    cmap : str or matplotlib.colors.Colormap, optional
        Color the arrows by vector magnitude with this colormap instead of a
        single ``color``; the two are mutually exclusive. The arrows are
        a scalar mappable, so ``plot.add.colorbar(plot.quiver)`` adds a
        colorbar (for a faceted plot pass one element of ``plot.quiver``).
    vmin, vmax : float, optional
        Magnitude limits of ``cmap``. Default to 0 and the largest magnitude of
        the plotted field, shared by every panel and animation frame.
    key_color : str, optional
        Color of the quiver-key arrow and label. Black when omitted.
    key_align : {"left", "center", "right"}, optional
        Horizontal alignment of the quiver key along the lower edge of the
        axes. Default ``"left"``. The vertical position stays automatic, below
        the axis decorations.
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
    cmap: str | Colormap | None = None
    vmin: float | None = None
    vmax: float | None = None
    key_color: str | None = None
    key_align: Literal["left", "center", "right"] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.u, xr.DataArray) or not isinstance(self.v, xr.DataArray):
            raise TypeError("u and v must be xarray.DataArray objects")
        if self.alpha is not None and not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        if self.powerlimits is not None:
            check_powerlimits(self.powerlimits)
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
        if self.cmap is not None and self.color is not None:
            raise ValueError("cmap and color are mutually exclusive")
        if self.vmin is not None and self.vmax is not None and self.vmin >= self.vmax:
            raise ValueError("vmin must be smaller than vmax")
        if self.key_align is not None and self.key_align not in (
            "left",
            "center",
            "right",
        ):
            raise ValueError("key_align must be 'left', 'center' or 'right'")

    def style(self) -> dict[str, Any]:
        """Return every non-``None`` field except the components ``u`` and ``v``."""
        return self.arguments(exclude=("u", "v"))
