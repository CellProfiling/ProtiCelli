"""
Tool registry for using ProtiCelli inside LLM agent loops.

``PROTICELLI_TOOLS`` uses standard JSON Schema (``"parameters"`` key), which most
agent frameworks accept directly.  Providers that need a different wrapper (e.g.
Anthropic expects ``"input_schema"`` instead of ``"parameters"``) can do so in one
line:

    # Anthropic
    tools = [{"name": t["name"], "description": t["description"],
               "input_schema": t["parameters"]} for t in PROTICELLI_TOOLS]

    # OpenAI / Groq / Mistral
    tools = [{"type": "function", "function": t} for t in PROTICELLI_TOOLS]

``run_tool`` is provider-agnostic — pass the tool name and parsed input dict
regardless of which LLM produced the call.
"""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any, Dict, List, Optional

PROTICELLI_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "validate_inputs",
        "description": (
            "Check whether protein and cell line names are in the model vocabulary. "
            "Returns resolved names, blocking errors, and any auto-corrections. "
            "Call this before predict_from_files when names may be uncertain."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "protein_names": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "cell_line_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Omit if not conditioning on cell line.",
                },
            },
            "required": ["protein_names"],
        },
    },
    {
        "name": "predict_from_files",
        "description": (
            "End-to-end ProtiCelli prediction from raw TIFF channel files. "
            "Handles assembly, normalization, and resampling internally. "
            "One set of channel files can predict multiple proteins."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "microtubules_path": {"type": "string"},
                "nucleus_path":      {"type": "string"},
                "er_path":           {"type": "string"},
                "protein_names": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "cell_line_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "One per protein, or omit to use default embedding.",
                },
                "xy_resolution": {
                    "type": "number",
                    "description": "Pixel size in µm/px (e.g. 0.1067, 0.0707, 0.3250).",
                },
                "bit_depth": {
                    "type": "integer",
                    "enum": [8, 16],
                    "description": "Default 16.",
                },
                "output_directory": {"type": "string"},
                "output_prefix":    {"type": "string"},
                "num_inference_steps": {
                    "type": "integer",
                    "description": "Denoising steps. Default 50.",
                },
                "seed": {"type": "integer"},
            },
            "required": [
                "microtubules_path",
                "nucleus_path",
                "er_path",
                "protein_names",
                "xy_resolution",
                "output_directory",
            ],
        },
    },
    {
        "name": "search_proteins",
        "description": (
            "Search the protein vocabulary by partial or informal name "
            "(e.g. 'mitochondria', 'TOMM'). Returns matching exact protein keys."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query":       {"type": "string"},
                "max_results": {"type": "integer", "description": "Default 10."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_cell_lines",
        "description": "Return all cell line names recognized by the model.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
]


def run_tool(model: Any, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch a tool call to the corresponding ProtiCelli operation.

    Parameters
    ----------
    model : Model
        An initialized ProtiCelli ``Model`` instance.
    tool_name : str
        The ``"name"`` field from the tool definition.
    tool_input : dict
        Parsed input payload from the LLM.

    Returns
    -------
    dict
        JSON-serializable result with ``"status"`` (``"ok"`` / ``"error"``)
        and ``"message"``.
    """
    handlers = {
        "validate_inputs":    _run_validate_inputs,
        "predict_from_files": _run_predict_from_files,
        "search_proteins":    _run_search_proteins,
        "list_cell_lines":    _run_list_cell_lines,
    }
    handler = handlers.get(tool_name)
    if handler is None:
        return {"status": "error", "message": f"Unknown tool: '{tool_name}'"}
    return handler(model, tool_input)


# ------------------------------------------------------------------ #
#  Tool implementations
# ------------------------------------------------------------------ #

def _run_validate_inputs(model: Any, inp: Dict[str, Any]) -> Dict[str, Any]:
    import numpy as np

    protein_names: List[str]             = inp["protein_names"]
    cell_line_names: Optional[List[str]] = inp.get("cell_line_names") or None
    n = len(protein_names)

    dummy  = [np.zeros((64, 64, 3), dtype=np.float32)] * n
    report = model.validate_inputs(dummy, protein_names, cell_line_names)

    return {
        "status":              "ok" if report["valid"] else "error",
        "valid":               report["valid"],
        "errors":              report["errors"],
        "warnings":            report["warnings"],
        "resolved_proteins":   report["resolved_proteins"],
        "resolved_cell_lines": report["resolved_cell_lines"],
        "message": (
            "All inputs are valid."
            if report["valid"]
            else f"{len(report['errors'])} error(s): {'; '.join(report['errors'])}"
        ),
    }


def _run_predict_from_files(model: Any, inp: Dict[str, Any]) -> Dict[str, Any]:
    from .data import ChannelAssembler, ImageNormalizer, ResolutionResampler

    try:
        stack = ChannelAssembler(has_protein=False).transform({
            "microtubules": inp["microtubules_path"],
            "nucleus":      inp["nucleus_path"],
            "er":           inp["er_path"],
        })
        norm  = ImageNormalizer(bit_depth=inp.get("bit_depth", 16)).transform(stack)
        ready = ResolutionResampler().transform(norm, xy_resolution=inp["xy_resolution"])

        protein_names   = inp["protein_names"]
        cell_line_names = inp.get("cell_line_names") or None
        n               = len(protein_names)

        results = model.predict(
            images=[ready] * n,
            protein_names=protein_names,
            cell_line_names=cell_line_names,
            num_inference_steps=inp.get("num_inference_steps", 50),
            seed=inp.get("seed"),
            show_progress=False,
        )

        out_dir = inp["output_directory"]
        prefix  = inp.get("output_prefix", "")
        results.save_prediction(prefix=prefix, directory=out_dir)

        saved = []
        for i, meta in enumerate(results.metadata):
            cl   = (meta.get("cell_line_name") or "unknown").replace(" ", "_")
            prot = (meta.get("protein_name")   or "unknown").replace(" ", "_")
            stem = f"{i}_{cl}_cell_{prot}"
            if prefix:
                stem = f"{prefix}_{stem}"
            saved.append(str(Path(out_dir) / f"{stem}.tif"))

        return {
            "status":      "ok",
            "message":     results.summary,
            "saved_files": saved,
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def _run_search_proteins(model: Any, inp: Dict[str, Any]) -> Dict[str, Any]:
    query       = inp["query"]
    max_results = inp.get("max_results", 10)
    keys        = model.available_proteins

    exact = [k for k in keys if query.lower() in k.lower()][:max_results]
    if len(exact) < max_results:
        fuzzy = difflib.get_close_matches(
            query, keys, n=max_results - len(exact), cutoff=0.4
        )
        matches = list(dict.fromkeys(exact + fuzzy))
    else:
        matches = exact

    return {
        "status":        "ok",
        "query":         query,
        "matches":       matches,
        "total_matches": len(matches),
        "message":       f"Found {len(matches)} protein(s) matching '{query}'.",
    }


def _run_list_cell_lines(model: Any, inp: Dict[str, Any]) -> Dict[str, Any]:
    lines = model.available_cell_lines
    return {
        "status":     "ok",
        "cell_lines": lines,
        "count":      len(lines),
        "message":    f"{len(lines)} cell line(s) available.",
    }
