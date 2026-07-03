"""Shared Plotly figure-persistence utilities.

Every chart produced across the five case studies is routed through
:func:`save_figure` so that HTML (interactive, standalone) and PNG
(static, for LaTeX embedding) artifacts are always produced together with
a consistent visual theme.
"""

from __future__ import annotations

import pathlib

import plotly.graph_objects as go
import plotly.io as pio

_TEMPLATE = "plotly_white"
_FONT = dict(family="Helvetica, Arial, sans-serif", size=13, color="#1a2332")
_ACCENT = ["#0B3D91", "#D62246", "#2E8B57", "#E8A33D", "#6A4C93", "#1B998B"]

pio.templates.default = _TEMPLATE


def apply_house_style(fig: go.Figure, title: str, height: int = 520, width: int = 980) -> go.Figure:
    """Applies the institutional chart theme to a Plotly figure in-place.

    Args:
        fig: The Plotly figure to style.
        title: Chart title, rendered in the house font.
        height: Figure height in pixels.
        width: Figure width in pixels.

    Returns:
        The same figure instance, styled and returned for chaining.
    """
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.02, xanchor="left", font=dict(size=18, color="#0B3D91")),
        font=_FONT,
        height=height,
        width=width,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=80, b=50),
        colorway=_ACCENT,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e8e8ef", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e8e8ef", zeroline=False)
    return fig


def save_figure(fig: go.Figure, name: str, out_dir: str = "charts") -> tuple[str, str]:
    """Persists a Plotly figure to both interactive HTML and static PNG.

    Args:
        fig: A styled Plotly figure (see :func:`apply_house_style`).
        name: Base filename (no extension), e.g. ``"p1_fee_forecast"``.
        out_dir: Destination directory, created if it does not exist.

    Returns:
        A ``(html_path, png_path)`` tuple of the written file paths.
    """
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    html_path = str(out / f"{name}.html")
    png_path = str(out / f"{name}.png")
    fig.write_html(html_path, include_plotlyjs="cdn")
    fig.write_image(png_path, scale=2)
    return html_path, png_path
