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


def load_esm(widget_js: pathlib.Path) -> str:
    """Return the shared JS helpers concatenated ahead of a widget's own JS.

    Parameters
    ----------
    widget_js : pathlib.Path
        Path to the widget's ``<name>.js`` (owns the `export default renderWidget`).

    Returns
    -------
    str
        Combined ESM source: ``dom.js`` followed by the widget's module.
    """
    shared = (_SHARED / "dom.js").read_text(encoding="utf-8")
    return shared + "\n" + widget_js.read_text(encoding="utf-8")


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
