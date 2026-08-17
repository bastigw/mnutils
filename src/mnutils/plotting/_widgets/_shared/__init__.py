"""Shared frontend assets for MNUtils plotting widgets.

Every widget's browser code is the concatenation of a common layer
(``dom.js`` / ``theme.css``) and the widget's own ``<name>.js`` / ``<name>.css``.
Because there is no JS bundler in this project, the sharing happens in Python:
:func:`load_esm` / :func:`load_css` read both files and join them into a single
source string that feeds `_widgets._html.render_html`.

The shared JS is deliberately ``import``/``export``-free so it can sit in the
same module scope as the widget's own ``export default renderWidget``.
"""

import pathlib

_SHARED = pathlib.Path(__file__).parent


def load_esm(widget_js: pathlib.Path, adapter: pathlib.Path | None = None) -> str:
    """Return the shared JS helpers concatenated ahead of a widget's own JS.

    Parameters
    ----------
    widget_js : pathlib.Path
        Path to the widget's ``<name>.js`` (owns the `export default renderWidget`).
    adapter : pathlib.Path, optional
        A backend adapter appended *after* the widget module, so it can call
        into `renderWidget` and provide its own ``export default``. This is how
        the anywidget backend reuses the widget component unchanged instead of
        forking it; the ``export`` in ``widget_js`` is harmless alongside it.

    Returns
    -------
    str
        Combined ESM source: ``dom.js``, the widget's module, then any adapter.
    """
    parts = [
        (_SHARED / "dom.js").read_text(encoding="utf-8"),
        widget_js.read_text(encoding="utf-8"),
    ]
    if adapter is not None:
        parts.append(adapter.read_text(encoding="utf-8"))
    return "\n".join(parts)


def load_css(widget_css: pathlib.Path) -> str:
    """Return the shared theme concatenated ahead of a widget's own CSS.

    Parameters
    ----------
    widget_css : pathlib.Path
        Path to the widget's ``<name>.css`` (widget-specific / layout classes).

    Returns
    -------
    str
        Combined stylesheet: ``theme.css`` followed by the widget's overrides.
    """
    shared = (_SHARED / "theme.css").read_text(encoding="utf-8")
    return shared + "\n" + widget_css.read_text(encoding="utf-8")
