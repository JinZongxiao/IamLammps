"""OVITO-based structure analysis for tensile workflow.

Gracefully degrades when OVITO is not available.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class OvitoAnalysisResult:
    available: bool
    unavailable_reason: str = ""
    # Keyframe images produced (label -> filename)
    keyframe_images: dict[str, str] = field(default_factory=dict)
    # Per-frame structural stats: {frame_label: {metric: value}}
    structural_stats: dict[str, dict] = field(default_factory=dict)
    # Summary text for the report
    summary: str = ""
    error: str = ""


def analyze_tensile_dump(
    run_dir: Path,
    strain_arr: list[float],
    yield_strain: Optional[float],
    uts_strain: Optional[float],
) -> OvitoAnalysisResult:
    """Run OVITO PTM/CNA analysis on tensile dump files.

    Parameters
    ----------
    run_dir:        Path to the deform stage run directory.
    strain_arr:     Engineering strain values corresponding to thermo steps.
    yield_strain:   Strain at yield point (may be None).
    uts_strain:     Strain at UTS (may be None).
    """
    try:
        import ovito  # noqa: F401 — just to check availability
    except ImportError:
        return OvitoAnalysisResult(
            available=False,
            unavailable_reason=(
                "OVITO Python package not found in the current environment. "
                "Install via: pip install ovito  or  conda install -c conda-forge ovito"
            ),
            summary="OVITO analysis unavailable",
        )

    # --- OVITO is available — run the actual analysis ---
    try:
        return _run_ovito_analysis(run_dir, strain_arr, yield_strain, uts_strain)
    except Exception as exc:
        return OvitoAnalysisResult(
            available=True,
            error=str(exc),
            summary=f"OVITO analysis failed: {exc}",
        )


# ---------------------------------------------------------------------------
# Internal — only reached when ovito is importable
# ---------------------------------------------------------------------------

def _run_ovito_analysis(
    run_dir: Path,
    strain_arr: list[float],
    yield_strain: Optional[float],
    uts_strain: Optional[float],
) -> OvitoAnalysisResult:
    import ovito.io as oio
    import ovito.modifiers as omod

    dump_files = sorted(run_dir.glob("dump.tensile.*.lammpstrj"))
    if not dump_files:
        return OvitoAnalysisResult(
            available=True,
            error="No dump.tensile.*.lammpstrj files found",
            summary="OVITO analysis skipped: no dump files",
        )

    # Build pipeline once for all files
    pipeline = oio.import_file(
        str(run_dir / "dump.tensile.*.lammpstrj"),
        sort_files=True,
    )
    pipeline.modifiers.append(omod.PolyhedralTemplateMatchingModifier(
        rmsd_cutoff=0.1,
        output_rmsd=False,
        output_interatomic_distance=False,
        output_ordering=False,
        output_orientation=False,
    ))

    n_frames = pipeline.source.num_frames

    # Identify target frame indices for 4 keyframes
    def _nearest_frame(target_strain: Optional[float]) -> int:
        if target_strain is None or not strain_arr:
            return 0
        diffs = [abs(s - target_strain) for s in strain_arr]
        idx = diffs.index(min(diffs))
        # Map thermo index → frame index (frames may be sparser)
        return min(int(idx * n_frames / max(len(strain_arr), 1)), n_frames - 1)

    keyframe_map = {
        "initial": 0,
        "yield": _nearest_frame(yield_strain),
        "uts": _nearest_frame(uts_strain),
        "final": n_frames - 1,
    }
    # Deduplicate while preserving order
    seen_frames: set[int] = set()
    unique_keyframes: dict[str, int] = {}
    for label, fidx in keyframe_map.items():
        if fidx not in seen_frames:
            unique_keyframes[label] = fidx
            seen_frames.add(fidx)

    # --- Render keyframes ---
    try:
        import ovito.vis as ovis
        _has_vis = True
    except ImportError:
        _has_vis = False

    keyframe_images: dict[str, str] = {}
    structural_stats: dict[str, dict] = {}

    for label, fidx in unique_keyframes.items():
        data = pipeline.compute(fidx)

        # Structural statistics (PTM)
        ptm_struct = data.particles_.structure_types_
        n_total = len(ptm_struct)
        n_bcc = int((ptm_struct == 1).sum())   # PTM type 1 = BCC
        n_other = n_total - n_bcc
        actual_strain = strain_arr[
            min(int(fidx * len(strain_arr) / max(n_frames, 1)), len(strain_arr) - 1)
        ] if strain_arr else None

        structural_stats[label] = {
            "frame_index": fidx,
            "strain_approx": round(actual_strain, 4) if actual_strain is not None else None,
            "n_atoms_total": n_total,
            "n_bcc": n_bcc,
            "n_non_bcc": n_other,
            "bcc_fraction": round(n_bcc / n_total, 4) if n_total > 0 else None,
        }

        # Render image if vis module available
        if _has_vis:
            img_name = f"ovito_{label}.png"
            img_path = run_dir / img_name
            vp = pipeline.source.data.cell.vis.create_viewport()
            vp.render_image(
                filename=str(img_path),
                size=(800, 600),
                frame=fidx,
                pipeline=pipeline,
            )
            keyframe_images[label] = img_name

    # Build summary
    stats_lines = []
    for label, stats in structural_stats.items():
        strain_str = f"ε≈{stats['strain_approx']:.3f}" if stats.get("strain_approx") is not None else ""
        bcc_frac = stats.get("bcc_fraction")
        bcc_str = f"BCC={bcc_frac*100:.1f}%" if bcc_frac is not None else "BCC=?"
        n_defect = stats.get("n_non_bcc", 0)
        stats_lines.append(f"  [{label} {strain_str}] {bcc_str}, 非BCC原子={n_defect}")

    summary = "PTM 结构分类完成。\n" + "\n".join(stats_lines)

    # Detect significant defect growth
    bcc_fracs = [
        v["bcc_fraction"]
        for v in structural_stats.values()
        if v.get("bcc_fraction") is not None
    ]
    if len(bcc_fracs) >= 2 and (bcc_fracs[0] - bcc_fracs[-1]) > 0.05:
        summary += "\n⚠ 检测到明显缺陷增长（BCC 比例下降 > 5%）。"

    return OvitoAnalysisResult(
        available=True,
        keyframe_images=keyframe_images,
        structural_stats=structural_stats,
        summary=summary,
    )
