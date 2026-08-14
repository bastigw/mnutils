import json
import uuid
from typing import Any


def render_html(js: str, css: str, data: dict[str, Any]) -> str:
    """Build a self-contained HTML/JS output embedding `data` for a widget's JS.

    Deliberately not the ipywidgets/anywidget comm protocol: every widget in
    this package is one-way (all data is baked in at construction, no
    Python-side callback is ever needed after display), so there's nothing
    to gain from a live model-sync channel -- and mystmd's execution engine
    doesn't support it anyway (it has no registered `jupyter.widget` comm
    target, so `widget-state+json` never gets captured into a static build,
    only a dangling `widget-view+json` reference). A plain HTML/JS output
    has no such dependency: it works identically in a live kernel, in myst's
    live preview, and in a static `myst build --html` site.

    `js` must export a default function `(data, el) => void` and read every
    value it needs directly from `data`.
    """
    container_id = f"mnutils-widget-{uuid.uuid4().hex}"
    # `</script>` inside the JSON payload would terminate the script block early.
    data_json = json.dumps(data).replace("<", "\\u003c")
    return f"""\
<div id="{container_id}"></div>
<style>{css}</style>
<script type="module">
{js}
renderWidget({data_json}, document.getElementById("{container_id}"));
</script>
"""
