"""
Log Case Validation GUI
========================
A Gradio-based interface for selecting log cases, configuring validation
options, running cross-validation, and reviewing results with issue images.

Usage:
    python app.py

Configuration:
    - LOG_CASES_FILE : Path to a text file listing log case names (one per line).
    - LOG_CASES_DIR  : Root directory containing log case folders.
    - PROJECT_ROOT   : Root directory for resolving issue_image paths in results.
"""

import json
import os
from pathlib import Path

import gradio as gr

import os
import cv2
import numpy as np
import json
from typing import Dict, List, Tuple
from src.data_validation_with_hybrid import DataValidationWithHybridStrategy

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LOG_CASES_FILE = "config/test_log_cases.txt"   # Text file with log case names (one per line)
LOG_CASES_DIR = "test_cases/RoboFlowSyntheticData"
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")

EXEMPLAR_FOLDER = "test_cases/RoboFlowConverted"
EXEMPLAR_RECOMMENDAR_LOG_CASE_FOLDER = "test_cases/RoboFlowConverted"
DATA_FOLDER = "test_cases/RoboFlowSyntheticData"
CONFIG_PATH = "config/data_validation_config.yaml"
TRAIN_LOG_CASE_LIST_PATH = "config/train_log_cases.txt"

g_hybrid_validator = DataValidationWithHybridStrategy(CONFIG_PATH,
                                                      EXEMPLAR_FOLDER,
                                                      EXEMPLAR_RECOMMENDAR_LOG_CASE_FOLDER,
                                                      TRAIN_LOG_CASE_LIST_PATH)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_log_case_list(filepath: str) -> list[str]:
    """Load log case names from a text file (one name per line)."""
    path = Path(filepath)
    if not path.is_file():
        return []
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def resolve_case_path(case_name: str) -> Path:
    """Return the absolute folder path for a given log case name."""
    return Path(LOG_CASES_DIR).resolve() / case_name


def resolve_image_path(image_ref: str, case_path: Path) -> str | None:
    """
    Resolve an issue_image path from the result JSON.

    The path in the JSON may be:
      - Absolute from the project root  (e.g. "test_cases/.../image.jpg")
      - Relative to the case folder     (e.g. "issues/image.jpg")
      - An absolute filesystem path

    We try project-root-relative first, then case-folder-relative, then
    absolute.  Returns the absolute path string if found, else None.
    """
    if not image_ref:
        return None

    # Try as project-root-relative path
    project_abs = Path(PROJECT_ROOT).resolve() / image_ref
    if project_abs.is_file():
        return str(project_abs)

    # Try as case-folder-relative path
    case_abs = case_path / image_ref
    if case_abs.is_file():
        return str(case_abs)

    # Try as absolute path
    if Path(image_ref).is_file():
        return str(Path(image_ref).resolve())

    return None


# ---------------------------------------------------------------------------
# Core validation function (placeholder)
# ---------------------------------------------------------------------------

def run_validation(case_name: str, enabled_options: list[str]) -> dict:
    """
    Execute cross-validation for the selected log case.

    Parameters
    ----------
    case_name : str
        Name of the selected log case.
    enabled_options : list[str]
        List of enabled validation options. Possible values:
        "JSON Schema", "Feature Validation", "SAM3 Validation".

    Returns
    -------
    dict
        The data-validation result dictionary.  The function also saves the
        result as ``data_validation_result.json`` inside the log case folder.

    Expected output schema
    ----------------------
    {
        "inspection_path": "...",
        "scale_factor": 1.5,
        "original_image_size": [1984, 1673],
        "ga_bins_used": 6,
        "ga_fitness": 0.084,
        "exemplars_image": "path/to/reference.jpg",
        "Labels with Issue": [
            {
                "id": 7,                            # int or "NA"
                "issue_image": "path/to/img.jpg",    # project-root-relative
                "data failure reason": "..."         # NOTE: spaces, not underscores
            }
        ],
        "False Positives": [ ... ],
        "False Negatives": [ ... ],
        "True Positives Num": 9,
        "True Negatives Num": 13,
        "False Positives Num": 1,
        "False Negatives Num": 0,
        "Precision": 0.9,
        "Recall": 1.0
    }
    """

    case_path = resolve_case_path(case_name)

    print(f"Running validation for case: {case_name}")
    print(f"Enabled options: {enabled_options}")

    log_case_folder = os.path.join(DATA_FOLDER, case_name)
    img_path = os.path.join(log_case_folder, "image.jpg")
    img = cv2.imread(img_path)
    label_path = os.path.join(log_case_folder, "label.json")
    with open(label_path, "r") as f:
        json_data = json.load(f)
    log_path = os.path.join(log_case_folder, "HybridDataValidationLog")
    result_path = os.path.join(log_case_folder, "HybridDataValidationResult")

    result_info = g_hybrid_validator(validation_case_folder=log_case_folder, 
                                       img=img,
                                       json_data=json_data,
                                       log_path=log_path, 
                                       result_path=result_path, 
                                       use_json_validator=True if "JSON Schema" in enabled_options else False, 
                                       use_feature_validator=True if "Feature Validation" in enabled_options else False,
                                       use_sam3_validator=True if "SAM3 Validation" in enabled_options else False)

    return result_info


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ISSUE_CATEGORIES = ["Labels with Issue", "False Positives", "False Negatives"]

METADATA_FIELDS = [
    ("inspection_path", "Inspection Path"),
    ("scale_factor", "Scale Factor"),
    ("original_image_size", "Original Image Size"),
    ("ga_bins_used", "GA Bins Used"),
    ("ga_fitness", "GA Fitness"),
]

SUMMARY_STATS = [
    ("True Positives Num", "True Positives"),
    ("True Negatives Num", "True Negatives"),
    ("False Positives Num", "False Positives"),
    ("False Negatives Num", "False Negatives"),
    ("Precision", "Precision"),
    ("Recall", "Recall"),
]


# ---------------------------------------------------------------------------
# GUI callback helpers
# ---------------------------------------------------------------------------

def on_case_selected(case_name: str):
    """When a log case is selected, return the preview image and label JSON."""
    if not case_name:
        return None, "{}"

    case_path = resolve_case_path(case_name)

    # Load image
    image_path = case_path / "image.jpg"
    image = str(image_path) if image_path.is_file() else None

    # Load label JSON
    label_path = case_path / "label.json"
    if label_path.is_file():
        with open(label_path, "r") as f:
            label_data = json.dumps(json.load(f), indent=2)
    else:
        label_data = '{"error": "label.json not found"}'

    return image, label_data


def render_metadata_html(result: dict, case_path: Path) -> str:
    """Render the metadata and summary statistics as an HTML panel."""

    # --- Summary stat cards ------------------------------------------------
    stats_cells = ""
    for key, label in SUMMARY_STATS:
        val = result.get(key)
        if val is None:
            continue
        if isinstance(val, float) and key in ("Precision", "Recall"):
            display = f"{val:.1%}"
            color = "#2563eb"
        else:
            display = str(val)
            color = "#333"
        stats_cells += (
            f'<div style="text-align:center; padding:10px 18px;">'
            f'<div style="font-size:24px; font-weight:700; color:{color};">{display}</div>'
            f'<div style="font-size:12px; color:#888; margin-top:2px;">{label}</div>'
            f'</div>'
        )

    # --- Issue count cards -------------------------------------------------
    issue_counts = ""
    cat_colors = {"Labels with Issue": "#e11d48", "False Positives": "#dc2626", "False Negatives": "#f59e0b"}
    for cat in ISSUE_CATEGORIES:
        count = len(result.get(cat, []))
        color = cat_colors.get(cat, "#333")
        issue_counts += (
            f'<div style="text-align:center; padding:10px 18px;">'
            f'<div style="font-size:24px; font-weight:700; color:{color};">{count}</div>'
            f'<div style="font-size:12px; color:#888; margin-top:2px;">{cat}</div>'
            f'</div>'
        )

    # --- Metadata key-value table ------------------------------------------
    meta_rows = ""
    for key, label in METADATA_FIELDS:
        val = result.get(key)
        if val is None:
            continue
        if isinstance(val, list):
            val = " × ".join(str(v) for v in val)
        elif isinstance(val, float):
            val = f"{val:.6f}"
        meta_rows += (
            f'<tr>'
            f'<td style="padding:6px 12px; font-weight:600; color:#555; white-space:nowrap;">{label}</td>'
            f'<td style="padding:6px 12px; word-break:break-all;">{val}</td>'
            f'</tr>'
        )


    html = f"""
    <div style="font-family:sans-serif;">
        <div style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:14px;
                    padding:14px; background:#f8fafc; border-radius:8px; border:1px solid #e2e8f0;">
            {stats_cells}
        </div>
        <div style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:14px;
                    padding:14px; background:#fff5f5; border-radius:8px; border:1px solid #fecdd3;">
            {issue_counts}
        </div>
        <table style="border-collapse:collapse; font-size:13px; margin-bottom:8px;">
            {meta_rows}
        </table>
        
    </div>
    """
    return html


def render_result_html(items: list[dict], case_path: Path) -> str:
    """Render a list of result items as an HTML table with embedded images."""
    if not items:
        return (
            '<p style="color:#888; font-style:italic; padding:12px;">'
            'No items in this category.</p>'
        )

    html = """
    <table style="width:100%; border-collapse:collapse; font-family:sans-serif; font-size:14px;">
        <thead>
            <tr style="background:#f0f0f0; text-align:left;">
                <th style="padding:10px; border-bottom:2px solid #ccc; width:80px;">ID</th>
                <th style="padding:10px; border-bottom:2px solid #ccc; width:280px;">Issue Image</th>
                <th style="padding:10px; border-bottom:2px solid #ccc;">Data Failure Reason</th>
            </tr>
        </thead>
        <tbody>
    """
    for idx, item in enumerate(items):
        item_id = item.get("id", "")

        # Resolve image path
        img_ref = item.get("issue_image", "")
        print(f"Resolving image for item ID {item_id}: {img_ref}")
        img_abs = resolve_image_path(img_ref, case_path)
        if img_abs:
            img_tag = (
                f'<img src="file={img_abs}" style="max-width:260px; max-height:200px; '
                f'border-radius:4px; border:1px solid #ddd;" />'
            )
        else:
            img_tag = (
                f'<em style="color:#aaa;">Image not found</em>'
                f'<br/><span style="font-size:11px; color:#bbb; word-break:break-all;">{img_ref}</span>'
            )

        # Support both "data failure reason" (real format) and "data_failure_reason"
        reason = item.get("data failure reason", "")
        if not reason:
            reason = item.get("data_failure_reason", "")

        bg = "#fff" if idx % 2 == 0 else "#fafafa"
        html += f"""
            <tr style="border-bottom:1px solid #eee; background:{bg};">
                <td style="padding:10px; vertical-align:top; font-weight:600;
                           font-family:monospace; font-size:15px;">{item_id}</td>
                <td style="padding:10px; vertical-align:top;">{img_tag}</td>
                <td style="padding:10px; vertical-align:top; line-height:1.5;">{reason}</td>
            </tr>
        """
    html += "</tbody></table>"
    return html


def on_start_validation(
    case_name: str,
    opt_json_schema: bool,
    opt_feature_validation: bool,
    opt_sam3_validation: bool,
):
    """Run validation and return metadata + category tables + visibility updates."""
    empty_html = ""
    hidden = gr.update(visible=False)

    if not case_name:
        return (
            "⚠️ Please select a log case first.",
            empty_html,
            empty_html, empty_html, empty_html,
            hidden, hidden, hidden, hidden,
        )

    # Build option list
    enabled_options: list[str] = []
    if opt_json_schema:
        enabled_options.append("JSON Schema")
    if opt_feature_validation:
        enabled_options.append("Feature Validation")
    if opt_sam3_validation:
        enabled_options.append("SAM3 Validation")

    if len(enabled_options) == 0:
        return (
            "⚠️ Please select at least one validation option.",
            empty_html,
            empty_html, empty_html, empty_html,
            hidden, hidden, hidden, hidden,
        )

    # Run validation
    result = run_validation(case_name, enabled_options)
    case_path = resolve_case_path(case_name)

    # Render metadata / summary panel
    meta_html = render_metadata_html(result, case_path)

    # Render each issue-category table
    cat_htmls = []
    cat_vis = []
    for cat in ISSUE_CATEGORIES:
        items = result.get(cat, [])
        cat_htmls.append(render_result_html(items, case_path))
        cat_vis.append(gr.update(visible=True))

    opts_str = ", ".join(enabled_options) if enabled_options else "None"
    status = f"✅ Validation complete for **{case_name}** — Options: {opts_str}"

    return (
        status,
        meta_html,
        *cat_htmls,
        gr.update(visible=True),   # metadata section
        *cat_vis,                  # 3 category sections
    )


# ---------------------------------------------------------------------------
# Build the Gradio UI
# ---------------------------------------------------------------------------

def create_app() -> gr.Blocks:
    log_cases = load_log_case_list(LOG_CASES_FILE)

    with gr.Blocks(
        title="Log Case Validation",
        theme=gr.themes.Soft(),
        css="""
            .json-preview-container {
                max-height: 360px;
                overflow-y: auto;
            }
            .json-preview-container .code-wrap,
            .json-preview-container .cm-editor {
                max-height: 320px !important;
                overflow-y: auto !important;
            }
        """,
    ) as app:

        gr.Markdown(
            "# 🔍 Log Case Cross-Validation\n"
            "Select a log case, configure validation options, and review results."
        )

        # --- Section 1: Log case selection & preview -----------------------
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📂 Log Case Selection")
                case_dropdown = gr.Dropdown(
                    choices=log_cases,
                    label="Select Log Case",
                    info="Loaded from test_log_cases.txt",
                    interactive=True,
                )
                gr.Markdown("## ⚙️ Validation Options")
                opt_json_schema = gr.Checkbox(label="JSON Schema", value=True)
                opt_feature_validation = gr.Checkbox(label="Feature Validation", value=True)
                opt_sam3 = gr.Checkbox(label="SAM3 Validation", value=True)
                start_btn = gr.Button(
                    "🚀 Start Validation",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=2):
                gr.Markdown("## 🖼️ Preview")
                with gr.Row():
                    preview_image = gr.Image(
                        label="image.jpg",
                        type="filepath",
                        interactive=False,
                        height=360,
                    )
                    preview_json = gr.Code(
                        label="label.json",
                        language="json",
                        interactive=False,
                        lines=15,
                        #max_lines=15,
                        elem_classes=["json-preview-container"],
                    )

        # --- Section 2: Results -------------------------------------------
        gr.Markdown("---")
        status_md = gr.Markdown("*Click 'Start Validation' to see results.*")

        # Metadata & summary stats
        with gr.Column(visible=False) as sec_meta:
            gr.Markdown("### 📊 Validation Summary")
            html_meta = gr.HTML()

        # Labels with Issue
        with gr.Column(visible=False) as sec_labels:
            gr.Markdown("### 🏷️ Labels with Issue")
            html_labels = gr.HTML()

        # False Positives
        with gr.Column(visible=False) as sec_fp:
            gr.Markdown("### 🔴 False Positives")
            html_fp = gr.HTML()

        # False Negatives
        with gr.Column(visible=False) as sec_fn:
            gr.Markdown("### 🟡 False Negatives")
            html_fn = gr.HTML()

        # --- Event wiring --------------------------------------------------
        case_dropdown.change(
            fn=on_case_selected,
            inputs=[case_dropdown],
            outputs=[preview_image, preview_json],
        )

        start_btn.click(
            fn=on_start_validation,
            inputs=[case_dropdown, opt_json_schema, opt_feature_validation, opt_sam3],
            outputs=[
                status_md,
                html_meta,
                html_labels, html_fp, html_fn,
                sec_meta, sec_labels, sec_fp, sec_fn,
            ],
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Collect all paths that Gradio needs permission to serve via file= refs
    allowed = list({
        str(Path(LOG_CASES_DIR).resolve()),
        str(Path(PROJECT_ROOT).resolve()),
    })
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        allowed_paths=allowed,
    )
