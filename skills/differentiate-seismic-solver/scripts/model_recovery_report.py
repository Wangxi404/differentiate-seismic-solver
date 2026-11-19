#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portable model-recovery report from saved arrays; never calls a solver.

Usage: python model_recovery_report.py models.npz metadata.json output_directory

NPZ fields: true, initial, inverted; increasing physical x/depth axes (2-D,
shape nx,nz), or x/y/depth axes (3-D, shape nx,ny,nz). No implicit transpose.
Optional fields: loss, loss_iterations, accepted_update, display_mask (boolean,
same shape), sources/receivers (N,ndim physical coordinates in axis order).

Metadata requires coordinate_unit, value_unit, accepted_updates, status
(running/completed/interrupted/failed), model_is_last_accepted=true. Completed
runs also require stopping_reason and source_checkpoint_sha256. Optional:
field_name, target_updates, coordinate_scale/coordinate_display_unit,
value_scale/value_display_unit, sections
({x,y,depth}: requested physical coordinates; nearest saved nodes are recorded),
include_3d, include_loss, loss_label, loss_log, provenance (JSON object).
A provenance.model_credit string appears with the primary model figure in
the HTML and is embedded in its downloadable PNG, PDF and SVG metadata.
Optional max_3d_points (default 100000) bounds the embedded volume size; disable
include_3d or explicitly increase that budget for a larger saved model.
Loss is one objective/stage; separate different-frequency histories upstream.

Colour limits are computed jointly over all three models on their physical
coverage. Older metadata containing limits can be read without changing this scale.
Dependencies: numpy, matplotlib. The offline 3-D view uses the browser canvas.
"""
import argparse
import base64
import hashlib
import html
import json
import operator
from pathlib import Path
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import nonsingular


BLUES = [[i / 8, color] for i, color in enumerate([
    "rgb(247,251,255)", "rgb(222,235,247)", "rgb(198,219,239)",
    "rgb(158,202,225)", "rgb(107,174,214)", "rgb(66,146,198)",
    "rgb(33,113,181)", "rgb(8,81,156)", "rgb(8,48,107)"])]
COLORS = [tuple(int(c) / 255 for c in re.findall(r"\d+", color))
          for _, color in BLUES]
CMAP = LinearSegmentedColormap.from_list("RecoveryBlues", COLORS)
ASSETS = Path(__file__).resolve().parents[1] / "assets"
COLOR_CREDIT = "This product includes color specifications and designs developed by Cynthia Brewer (http://colorbrewer.org/)."
FIELDS = ("true", "initial", "inverted")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(value):
    """Hash dtype, shape and C-order bytes, not a platform-dependent repr."""
    value = np.asarray(value)
    header = json.dumps({"dtype": value.dtype.str, "shape": list(value.shape)},
                        sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(header + b"\n" + value.tobytes(order="C")).hexdigest()


def finite(value, name):
    value = np.asarray(value)
    if value.dtype.kind not in "fiu" or not value.size or not np.isfinite(value).all():
        raise ValueError(f"{name}: require nonempty finite real numeric data")
    return value


def count(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name}: require a nonnegative integer")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name}: require a nonnegative integer") from exc
    if value < 0:
        raise ValueError(f"{name}: require a nonnegative integer")
    return value


def validate(data, metadata):
    """Validate physical geometry and evidence labels without modifying inputs."""
    meta = dict(metadata)
    arrays = {name: finite(data[name], name) for name in FIELDS}
    shape = arrays["true"].shape
    if len(shape) not in (2, 3) or any(a.shape != shape for a in arrays.values()):
        raise ValueError("true/initial/inverted require identical 2-D or 3-D shapes")
    axes = ("x", "depth") if len(shape) == 2 else ("x", "y", "depth")
    if tuple(meta.get("axis_order", axes)) != axes:
        raise ValueError(f"axis_order must explicitly match {axes}; transpose upstream")
    for index, name in enumerate(axes):
        value = finite(data[name], name)
        if value.ndim != 1 or value.size != shape[index] or value.size < 2:
            raise ValueError(f"{name}: physical axis must match its field dimension (at least 2 nodes)")
        if not np.all(value[1:] > value[:-1]):
            raise ValueError(f"{name}: physical coordinates must be strictly increasing")
        arrays[name] = value
    for name in ("coordinate_unit", "value_unit"):
        if not isinstance(meta.get(name), str) or not meta[name].strip():
            raise ValueError(f"{name} must be explicit")
    for prefix in ("coordinate", "value"):
        scale = float(meta.get(prefix + "_scale", 1))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"{prefix}_scale must be finite and positive")
        meta[prefix + "_scale"] = scale
        if scale != 1 and not meta.get(prefix + "_display_unit"):
            raise ValueError(f"{prefix}_display_unit required for a unit conversion")
        meta.setdefault(prefix + "_display_unit", meta[prefix + "_unit"])
    accepted = count(meta["accepted_updates"], "accepted_updates")
    meta["accepted_updates"] = accepted
    for key in ("accepted_update", "accepted_updates", "stage_accepted_update"):
        if key in data:
            saved = np.asarray(data[key])
            if saved.size != 1 or count(saved.item(), key) != accepted:
                raise ValueError("metadata accepted count does not match the saved model")
    if "target_updates" in meta:
        meta["target_updates"] = count(meta["target_updates"], "target_updates")
        if accepted > meta["target_updates"]:
            raise ValueError("accepted count exceeds target_updates")
    if meta.get("model_is_last_accepted") is not True:
        raise ValueError("model_is_last_accepted=true is required; do not render a trial as accepted")
    if meta.get("status") not in ("running", "completed", "interrupted", "failed"):
        raise ValueError("status must be running, completed, interrupted or failed")
    if meta["status"] == "completed":
        if not str(meta.get("stopping_reason", "")).strip():
            raise ValueError("completed report requires an actual stopping_reason")
        if not re.fullmatch(r"[a-fA-F0-9]{64}", str(meta.get("source_checkpoint_sha256", ""))):
            raise ValueError("completed report requires source_checkpoint_sha256")
    if not isinstance(meta.get("provenance", {}), dict):
        raise ValueError("provenance must be a JSON object")
    if not isinstance(meta.get("provenance", {}).get("model_credit", ""), str):
        raise ValueError("provenance.model_credit must be text")
    for key in ("include_3d", "include_loss", "loss_log"):
        if key in meta and not isinstance(meta[key], bool):
            raise ValueError(f"{key} must be a JSON boolean")
    budget = count(meta.get("max_3d_points", 100000), "max_3d_points")
    if len(shape) == 3 and meta.get("include_3d", True) and arrays["true"].size > budget:
        raise ValueError("3-D payload exceeds max_3d_points; set include_3d=false or explicitly increase the budget (no automatic downsampling)")
    if "display_mask" in data:
        mask = np.asarray(data["display_mask"])
        if mask.dtype.kind != "b" or mask.shape != shape or not mask.any():
            raise ValueError("display_mask must be a nonempty boolean mask matching the model")
        arrays["display_mask"] = mask
    else:
        mask = None
    fields = [arrays[key] if mask is None else arrays[key][mask] for key in FIELDS]
    low = min(float(field.min()) for field in fields)
    high = max(float(field.max()) for field in fields)
    if low == high:
        low, high = nonsingular(low, high)
    meta["limits"] = [low, high]
    for name in ("sources", "receivers"):
        if name in data:
            points = finite(data[name], name)
            if points.ndim != 2 or points.shape[1] != len(shape):
                raise ValueError(f"{name} requires N by {len(shape)} physical coordinates")
            arrays[name] = points
    if "loss" in data:
        loss = finite(data["loss"], "loss")
        if loss.ndim != 1:
            raise ValueError("loss must be a vector for one objective/stage")
        if "loss_iterations" in data:
            iterations = finite(data["loss_iterations"], "loss_iterations")
        elif loss.size == accepted + 1:
            iterations = np.arange(accepted + 1)
        else:
            raise ValueError("provide loss_iterations unless loss contains exactly 0..accepted")
        if iterations.shape != loss.shape or not np.all(np.diff(iterations) > 0):
            raise ValueError("loss_iterations must match loss and strictly increase")
        if np.any(iterations < 0) or np.any(iterations > accepted) or np.any(iterations % 1):
            raise ValueError("loss_iterations must be actual accepted integer counts")
        if iterations[-1] != accepted:
            raise ValueError("loss history must end at the saved accepted model; omit unavailable loss instead of showing a stale curve")
        if meta.get("loss_log", True) and np.any(loss <= 0):
            raise ValueError("log loss requires positive values; use loss_log=false for signed/zero loss")
        arrays.update(loss=loss, loss_iterations=iterations)
    sections = meta.get("sections", {})
    if not isinstance(sections, dict) or set(sections) - set(axes):
        raise ValueError("sections must map physical axis names to coordinates")
    chosen = {}
    for name in axes:
        axis = arrays[name]
        requested = float(sections.get(name, (float(axis[0]) + float(axis[-1])) / 2))
        if not np.isfinite(requested) or not axis[0] <= requested <= axis[-1]:
            raise ValueError(f"section {name} lies outside saved coordinates")
        index = int(np.argmin(np.abs(axis - requested)))
        chosen[name] = {"requested": requested, "index": index, "actual": float(axis[index])}
    return arrays, meta, chosen


def edges(axis):
    """Nearest-node cells clipped to the supplied physical domain endpoints."""
    axis = np.asarray(axis, dtype=float)
    return np.r_[axis[0], (axis[:-1] + axis[1:]) / 2, axis[-1]]


def status_text(meta):
    number = str(meta["accepted_updates"])
    if "target_updates" in meta:
        number += "/" + str(meta["target_updates"])
    label = {"running": "In progress", "completed": "Completed",
             "interrupted": "Interrupted", "failed": "Stopped after failure"}[meta["status"]]
    result = label + " " + number
    if meta.get("stopping_reason"):
        reason = str(meta["stopping_reason"])
        if reason.lower() in ("maximum stage accepted updates reached", "maximum accepted updates reached", "iteration limit"):
            reason = "Iteration limit"
        result += " · " + reason
    return result


def layout_record(fig, axes, output):
    fig.canvas.draw()
    bounds = [list(ax.get_position().bounds) for ax in axes]
    record = {"figure_inches": fig.get_size_inches().tolist(),
              "plot_area_bounds_fraction": bounds, "tolerance_pt": 1.5}
    # All comparable columns share each row's exact rectangle dimensions.
    widths = np.array([b[2] for b in bounds]).reshape(-1, 3)
    heights = np.array([b[3] for b in bounds]).reshape(-1, 3)
    if np.max(np.ptp(widths, axis=1)) * fig.get_figwidth() * 72 > 1.5 or np.max(np.ptp(heights, axis=1)) * fig.get_figheight() * 72 > 1.5:
        raise ValueError("model-panel geometry is not aligned")
    output.write_text(json.dumps(record, indent=2) + "\n")
    return record


def save_figure(fig, stem, model_credit=""):
    # Stable timestamps and font settings make exports reproducible.
    description = {"Description": model_credit} if model_credit else {}
    subject = {"Subject": model_credit} if model_credit else {}
    fig.savefig(str(stem) + ".png", dpi=300, metadata=description)
    fig.savefig(str(stem) + ".pdf", metadata={"CreationDate": None, "ModDate": None, **subject})
    fig.savefig(str(stem) + ".svg", metadata={"Date": None, **description})
    paths = [Path(str(stem) + "." + ext) for ext in ("png", "pdf", "svg")]
    return {path.name: sha256(path) for path in paths}


def draw_models(arrays, meta, sections, output):
    cs, vs = meta["coordinate_scale"], meta["value_scale"]
    unit, vunit = meta["coordinate_display_unit"], meta["value_display_unit"]
    is3d = arrays["true"].ndim == 3
    views = [("x", "depth", "y", "XZ"), ("y", "depth", "x", "YZ"),
             ("x", "y", "depth", "XY")] if is3d else [("x", "depth", None, "XZ")]
    spans = {k: float((arrays[k][-1] - arrays[k][0]) * cs) for k in ("x", "depth") + (("y",) if is3d else ())}
    colw, gap, left, right, top, bottom, rowgap = 2.35, .40, 1.00, 1.05, .65, .72, .72
    heights = [colw * spans[v] / spans[h] for h, v, _, _ in views]
    # Anisotropic grids retain equal physical scaling; tall plates grow rather
    # than silently stretch their spatial content.
    width = left + 3 * colw + 2 * gap + right
    height = top + sum(heights) + (len(views)-1) * rowgap + bottom
    fig = plt.figure(figsize=(width, height))
    axes, panel_records = [], []
    ytop = height - top
    labels = ("True", "Initial", "Inverted " + str(meta["accepted_updates"]))
    low, high = np.asarray(meta["limits"]) * vs
    for row, ((h, v, fixed, view), panelh) in enumerate(zip(views, heights)):
        ybottom = ytop - panelh
        for col, key in enumerate(FIELDS):
            ax = fig.add_axes([(left + col * (colw + gap))/width,
                               ybottom/height, colw/width, panelh/height])
            axes.append(ax)
            field = arrays[key]
            mask = arrays.get("display_mask", np.ones(field.shape, dtype=bool))
            if is3d:
                dim = ("x", "y", "depth").index(fixed)
                field = np.take(field, sections[fixed]["index"], axis=dim)
                mask = np.take(mask, sections[fixed]["index"], axis=dim)
            shown = np.ma.array(field.T * vs, mask=~mask.T)
            artist = ax.pcolormesh(edges(arrays[h] * cs), edges(arrays[v] * cs), shown,
                                   cmap=CMAP, vmin=low, vmax=high, shading="flat", rasterized=True)
            ax.set_xlim(arrays[h][0] * cs, arrays[h][-1] * cs)
            ax.set_ylim(arrays[v][-1] * cs, arrays[v][0] * cs) if v == "depth" else ax.set_ylim(arrays[v][0] * cs, arrays[v][-1] * cs)
            ax.set_aspect("equal", adjustable="box")
            ax.set_anchor("C")
            ax.set_xlabel(h + " (" + unit + ")")
            if col == 0:
                ax.set_ylabel(("Depth" if v == "depth" else v) + " (" + unit + ")")
                fixed_label = "" if fixed is None else " · " + fixed + " = " + format(sections[fixed]["actual"] * cs, ".4g") + " " + unit
                ax.set_ylabel(view + ": " + ("Depth" if v == "depth" else v) + " (" + unit + ")" + ("\n" + fixed_label[3:] if fixed else ""))
            else:
                ax.tick_params(labelleft=False)
            if row == 0:
                ax.set_title(labels[col], fontsize=9, pad=7)
            panel_records.append({"field": key, "view": view, "horizontal_axis": h,
                                  "vertical_axis": v, "fixed_axis": fixed,
                                  "shape": list(field.shape), "masked_count": int((~mask).sum()),
                                  "raw_slice_sha256": array_hash(field)})
        ytop = ybottom - rowgap
    cax = fig.add_axes([(width-right+.19)/width, bottom/height, .12/width,
                        (height-top-bottom)/height])
    fig.colorbar(artist, cax=cax, label=meta.get("field_name", "Model") + " (" + vunit + ")")
    fig.text(left/width, .13/height, status_text(meta), fontsize=8, color="#536576")
    geometry = layout_record(fig, axes, output / "model-recovery.alignment.json")
    model_credit = meta.get("provenance", {}).get("model_credit", "")
    exports = save_figure(fig, output / "model-recovery", (model_credit + "\n" + COLOR_CREDIT).strip())
    plt.close(fig)
    return {"exports": exports, "panels": panel_records, "geometry": geometry}


def draw_loss(arrays, meta, output):
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    fig.subplots_adjust(left=.15, right=.96, bottom=.22, top=.86)
    ax.plot(arrays["loss_iterations"], arrays["loss"], color="#08519c", linewidth=1.6)
    if meta.get("loss_log", True):
        ax.set_yscale("log")
    ax.set_xlabel("Accepted update")
    ax.set_ylabel(meta.get("loss_label", "Objective"))
    ax.set_title(status_text(meta), fontsize=10)
    ax.grid(True, alpha=.18)
    exports = save_figure(fig, output / "loss")
    plt.close(fig)
    return exports


def volume_payload(arrays, meta):
    cs, vs = meta["coordinate_scale"], meta["value_scale"]
    mask = arrays.get("display_mask")
    fields = {}
    for name in FIELDS:
        values = np.asarray(arrays[name], dtype=float).ravel() * vs
        if mask is None:
            fields[name] = values.tolist()
        else:
            fields[name] = [float(v) if keep else None for v, keep in zip(values, mask.ravel())]
    return {
        "axes": [(np.asarray(arrays[name], dtype=float) * cs).tolist() for name in ("x", "y", "depth")],
        "fields": fields,
        "limits": (np.asarray(meta["limits"]) * vs).tolist(),
        "palette": [matplotlib.colors.to_hex(CMAP(index)) for index in range(CMAP.N)],
        "acquisition": {name: (np.asarray(arrays[name], dtype=float) * cs).tolist()
                        for name in ("sources", "receivers") if name in arrays},
        "coordinate_unit": meta["coordinate_display_unit"],
        "value_unit": meta["value_display_unit"],
    }


def volume_view(meta, payload):
    low, high = payload["limits"]
    unit = html.escape(payload["value_unit"])
    titles = ("True", "Initial", "Inverted " + str(meta["accepted_updates"]))
    panels = "".join(
        f'<figure><figcaption>{title}</figcaption><canvas data-field="{field}" tabindex="0" '
        f'aria-label="{title} model. Rotate with the arrow keys."></canvas></figure>'
        for field, title in zip(FIELDS, titles)
    )
    field = html.escape(meta.get("field_name", "Model"))
    stops = ", ".join(color for _, color in BLUES)
    color_key = (f'<span>{field} ({unit}) &nbsp; {low:.5g} '
                 f'<span class="ramp" style="background:linear-gradient(to right,{stops})"></span> {high:.5g}</span>')
    markers = payload["acquisition"]
    legend = []
    if "receivers" in markers:
        legend.append("▲ Receivers")
    if "sources" in markers:
        legend.append('<span class="source">★</span> Sources')
    marker_control = ('<label><input id="show-acquisition" type="checkbox" checked> ' +
                      " · ".join(legend) + '</label>') if legend else ""
    return (
        '<details id="three-d-models"><summary>3D models</summary>'
        '<p class="hint">Drag or use the arrow keys to rotate all views. Scroll to zoom.</p>'
        f'<div class="controls"><label>Minimum displayed value <input id="threshold" type="range" '
        f'min="{low}" max="{high}" value="{low}" step="any"> <output id="threshold-value"></output> {unit}</label>'
        + marker_control + '<button type="button" id="reset-view">Reset view</button></div>'
        + '<div id="model-volumes">' + panels + '</div><div class="volume-key">'
        + color_key + '<span id="volume-extents"></span></div></details>'
    )


def write_html(arrays, meta, output):
    def image(name, title):
        encoded = base64.b64encode((output / name).read_bytes()).decode()
        return '<figure><img alt="'+html.escape(title, quote=True)+'" src="data:image/png;base64,'+encoded+'"></figure>'
    primary_title = "Three-direction model recovery" if arrays["true"].ndim == 3 else "Model recovery"
    sections = '<h2>'+primary_title+'</h2>' + image("model-recovery.png", primary_title)
    if "display_mask" in arrays and not arrays["display_mask"].all():
        sections += "<p><small>Blank cells lie outside the simulated domain.</small></p>"
    credit = meta.get("provenance", {}).get("model_credit", "")
    if credit:
        sections += "<p><small>" + html.escape(credit) + "</small></p>"
    asset_info, payload = None, None
    scripts = ""
    notices = (ASSETS / "LICENSE.colorbrewer").read_text(encoding="utf-8")
    if arrays["true"].ndim == 3 and meta.get("include_3d", True):
        engine = ASSETS / "volume-viewer.js"
        asset_info = {"file": engine.name, "sha256": sha256(engine)}
        payload = volume_payload(arrays, meta)
        encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
        sections += volume_view(meta, payload)
        scripts = ('<script id="volume-data" type="application/json">' + encoded + '</script>'
                   + '<script>' + engine.read_text(encoding="utf-8") + '</script>')
    if "loss" in arrays and meta.get("include_loss", True):
        sections += '<details id="loss-curve"><summary>Loss curve</summary>'+image("loss.png", "Loss curve")+'</details>'
    sections += '<details><summary>Third-party notices</summary><pre style="white-space:pre-wrap">' + html.escape(notices) + '</pre></details>'
    css = 'body{font:16px/1.5 system-ui;margin:0;background:#f5f7fa;color:#18314a}main{max-width:1250px;margin:auto;padding:24px 20px}header{display:flex;align-items:baseline;gap:20px;flex-wrap:wrap}h1{font-size:26px;margin:0 0 12px}header span{color:#60748a;font-size:14px}h2{font-size:16px;margin:4px 0 8px}figure{margin:0 0 18px;background:white;border-radius:7px;padding:10px}img{display:block;width:100%;height:auto}details{margin:12px 0;background:white;border:1px solid #dae2ea;border-radius:7px;padding:10px 14px}summary{cursor:pointer;font-size:15px}#model-volumes{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}#model-volumes figure{margin:0;padding:0;text-align:center}canvas{display:block;width:100%;aspect-ratio:1.25;touch-action:none;cursor:grab}canvas:active{cursor:grabbing}.controls{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin:8px 0 15px}.controls label,.hint{font-size:13px}.controls input{vertical-align:middle}.volume-key{display:flex;gap:20px;flex-wrap:wrap;align-items:center;margin:8px 0;color:#526679;font-size:12px}.ramp{display:inline-block;width:85px;height:9px;vertical-align:middle;border:1px solid #b4bfca}.source{color:#b02e2b}.hint{color:#526679}@media(max-width:600px){#model-volumes{grid-template-columns:1fr}}@media(max-width:700px){main{padding:14px 8px}h1{font-size:22px}}'
    document = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Model recovery</title><style>'+css+'</style></head><body><main><header><h1>Model recovery</h1><span>'+html.escape(status_text(meta))+'</span></header>'+sections+'</main>'+scripts+'</body></html>'
    (output / "model-recovery.html").write_text(document, encoding="utf-8")
    return {"asset": asset_info, "volume_payload_sha256": None if payload is None else hashlib.sha256(json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()).hexdigest()}


def render_report(npz_path, metadata_path, output):
    """Render a saved accepted-model snapshot and return its audit manifest."""
    npz_path, metadata_path, output = Path(npz_path), Path(metadata_path), Path(output)
    with np.load(npz_path, allow_pickle=False) as saved:
        data = {key: saved[key].copy() for key in saved.files}
    arrays, meta, sections = validate(data, json.loads(metadata_path.read_text()))
    # Validate first, so malformed evidence cannot leave a plausible new report.
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "sans-serif", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "svg.hashsalt": "model-recovery-report-v1"}):
        models = draw_models(arrays, meta, sections, output)
        loss = draw_loss(arrays, meta, output) if "loss" in arrays and meta.get("include_loss", True) else {}
    html_info = write_html(arrays, meta, output)
    low, high = meta["limits"]
    mask = arrays.get("display_mask")
    fields = {}
    for key in FIELDS:
        field = arrays[key]
        covered = field if mask is None else field[mask]
        clipped = int(np.count_nonzero((covered < low) | (covered > high)))
        fields[key] = {"sha256": array_hash(field), "shape": list(field.shape),
                       "min": float(field.min()), "max": float(field.max()), "count": int(field.size),
                       "covered_min": float(covered.min()), "covered_max": float(covered.max()),
                       "covered_count": int(covered.size), "clipped_count": clipped,
                       "clipped_fraction": clipped / covered.size,
                       "clipping_scope": "Complete physical coverage before section extraction"}
    truth64 = arrays["true"].astype(np.float64)
    metrics = {"scope": "All supplied controls; raw value units. No geometry/illumination mask inferred.",
               "sample_count": int(truth64.size), "unit": meta["value_unit"]}
    if mask is not None:
        metrics["physical_coverage"] = {"scope": "Controls inside the supplied native-coverage mask",
                                        "sample_count": int(np.count_nonzero(mask)), "unit": meta["value_unit"]}
    for key in ("initial", "inverted"):
        squared_error = np.square(arrays[key].astype(np.float64) - truth64)
        metrics["rmse_" + key] = float(np.sqrt(np.mean(squared_error)))
        if mask is not None:
            metrics["physical_coverage"]["rmse_" + key] = float(np.sqrt(np.mean(squared_error[mask])))
    manifest = {"schema_version": 1, "report_language": "en", "helper_sha256": sha256(__file__),
                "input_npz_sha256": sha256(npz_path), "metadata_sha256": sha256(metadata_path),
                "metadata": meta, "fields": fields,
                "arrays": {key: array_hash(value) for key, value in arrays.items()},
                "sections": sections, "metrics": metrics,
                "scale_policy": "Shared extrema of true, initial and inverted over complete physical coverage",
                "palette": {"name": "ColorBrewer Blues", "colorscale": BLUES, "reversescale": False},
                "display_mask": {"present": "display_mask" in arrays,
                                 "excluded_count": int((~arrays["display_mask"]).sum()) if "display_mask" in arrays else 0},
                "display_transforms": "Explicit unit scaling; nearest saved section; nearest-node cells. No smoothing, spatial interpolation or array modification.",
                "model_figure": models, "loss_exports": loss, "html": html_info,
                "outputs": {**models["exports"], **loss, "model-recovery.html": sha256(output / "model-recovery.html")},
                "convergence_claim": False, "solver_calls": 0,
                "status_provenance": "Caller-supplied accepted-model/status attestation; renderer checks saved count and hashes, not solver qualification or checkpoint contents.",
                "browser_interaction": "Not exercised by this renderer; static data and asset hashes retained."}
    (output / "report-manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("arrays", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = render_report(args.arrays, args.metadata, args.output)
    print(json.dumps({"html": str(args.output / "model-recovery.html"),
                      "accepted_updates": report["metadata"]["accepted_updates"],
                      "status": report["metadata"]["status"]}))


if __name__ == "__main__":
    main()
