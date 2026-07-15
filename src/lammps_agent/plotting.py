"""Deterministic plots derived from immutable run artifacts."""

from __future__ import annotations

import importlib
import math
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

from .run_artifacts import RunArtifactStore, RunManifest
from .thermo_parser import parse_thermo_table


class PlotError(ValueError):
    code = "PLOT_ERROR"

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": str(self)}


class UnknownPlotTypeError(PlotError):
    code = "UNKNOWN_PLOT_TYPE"


class PlotDataError(PlotError):
    code = "PLOT_DATA_ERROR"


class PlottingDependencyError(PlotError):
    code = "PLOTTING_DEPENDENCY_MISSING"


@dataclass(frozen=True, slots=True)
class PlotArtifact:
    plot_type: str
    path: str
    width: int
    height: int
    bytes: int
    source_run_id: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def plot_run(
    workspace_root: str | Path,
    run_id: str,
    plot_types: Sequence[str],
) -> list[PlotArtifact]:
    """Create validated PNGs below one run's ``artifacts/plots`` directory."""
    if not plot_types:
        raise PlotError("at least one plot type is required")
    unknown = sorted(set(plot_types) - {"thermo", "rdf_oo"})
    if unknown:
        raise UnknownPlotTypeError(f"Unknown plot type: {unknown[0]}")

    store = RunArtifactStore(workspace_root)
    manifest = store.inspect_run(run_id)
    figure_factory, pyplot, image_reader = _load_matplotlib()
    output_dir = store.resolve_manifest_path(manifest.artifact_dir) / "artifacts" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[PlotArtifact] = []
    for plot_type in dict.fromkeys(plot_types):
        destination = output_dir / f"{plot_type}.png"
        if plot_type == "thermo":
            table = _load_thermo(store, manifest)
            render = partial(_render_thermo, figure_factory, table)
        else:
            radius, rdf = _load_rdf_oo(store, manifest)
            render = partial(_render_rdf, figure_factory, radius, rdf)
        width, height, size = _save_atomic_png(
            destination, render, pyplot, image_reader
        )
        outputs.append(
            PlotArtifact(
                plot_type=plot_type,
                path=destination.relative_to(store.root).as_posix(),
                width=width,
                height=height,
                bytes=size,
                source_run_id=manifest.run_id,
            )
        )
    return outputs


def parse_rdf_oo(source: str | Path) -> tuple[list[float], list[float]]:
    """Parse the last complete LAMMPS ``fix ave/time mode vector`` RDF block."""
    text = Path(source).read_text(encoding="utf-8") if isinstance(source, Path) else source
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    latest: tuple[list[float], list[float]] | None = None
    index = 0
    while index < len(lines):
        header = lines[index].split()
        if len(header) == 2 and all(_is_integer(token) for token in header):
            row_count = int(header[1])
            if row_count <= 0 or index + row_count >= len(lines):
                index += 1
                continue
            radius: list[float] = []
            rdf: list[float] = []
            valid = True
            for row_number in range(1, row_count + 1):
                tokens = lines[index + row_number].split()
                if len(tokens) < 3 or not _is_integer(tokens[0]):
                    valid = False
                    break
                try:
                    r_value = float(tokens[1])
                    rdf_value = float(tokens[2])
                except ValueError:
                    valid = False
                    break
                if not math.isfinite(r_value) or not math.isfinite(rdf_value):
                    valid = False
                    break
                radius.append(r_value)
                rdf.append(rdf_value)
            if valid:
                latest = (radius, rdf)
                index += row_count + 1
                continue
        index += 1
    if latest is None or len(latest[0]) < 2:
        raise PlotDataError("RDF file has no complete data block with at least two rows")
    if any(right <= left for left, right in zip(latest[0], latest[0][1:])):
        raise PlotDataError("RDF radius values must be strictly increasing")
    return latest


def _load_matplotlib() -> tuple[Callable[..., Any], Any, Callable[[str], Any]]:
    try:
        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg", force=True)
        pyplot = importlib.import_module("matplotlib.pyplot")
        image = importlib.import_module("matplotlib.image")
    except ImportError as exc:
        raise PlottingDependencyError(
            "matplotlib is required for deterministic plotting; install the project dependencies"
        ) from exc
    return pyplot.subplots, pyplot, image.imread


def _load_thermo(
    store: RunArtifactStore, manifest: RunManifest
) -> dict[str, list[float]]:
    path = store.resolve_manifest_path(manifest.stdout_path)
    table = parse_thermo_table(path)
    steps = table.get("Step", [])
    series_names = [
        name
        for name in ("Temp", "TotEng", "PotEng", "KinEng", "Press", "Density")
        if len(table.get(name, [])) == len(steps)
    ]
    if len(steps) < 2 or not series_names:
        raise PlotDataError("thermo output needs Step and at least one plottable series")
    for name in ("Step", *series_names):
        if not all(math.isfinite(value) for value in table[name]):
            raise PlotDataError(f"thermo column {name} contains non-finite values")
    return {"Step": steps, **{name: table[name] for name in series_names}}


def _load_rdf_oo(
    store: RunArtifactStore, manifest: RunManifest
) -> tuple[list[float], list[float]]:
    candidates = [path for path in manifest.output_files if Path(path).name == "rdf_oo.dat"]
    fallback = f"{manifest.workspace_dir}/rdf_oo.dat"
    if not candidates and store.resolve_manifest_path(fallback).is_file():
        candidates.append(fallback)
    if len(candidates) != 1:
        raise PlotDataError("run must contain exactly one rdf_oo.dat output")
    return parse_rdf_oo(store.resolve_manifest_path(candidates[0]))


def _render_thermo(figure_factory: Callable[..., Any], table: dict[str, list[float]]) -> Any:
    figure, axes = figure_factory(figsize=(10, 6), dpi=100, constrained_layout=True)
    names = [name for name in table if name != "Step"]
    primary = axes
    secondary = axes.twinx() if len(names) > 1 else None
    for index, name in enumerate(names):
        axis = primary if index == 0 or secondary is None else secondary
        axis.plot(table["Step"], table[name], linewidth=1.5, label=name)
    primary.set_xlabel("Step")
    primary.set_ylabel(names[0])
    primary.grid(True, linewidth=0.5, alpha=0.35)
    if secondary is not None:
        secondary.set_ylabel(" / ".join(names[1:]))
    lines = primary.lines + ([] if secondary is None else secondary.lines)
    primary.legend(lines, [line.get_label() for line in lines], loc="best")
    return figure


def _render_rdf(
    figure_factory: Callable[..., Any], radius: list[float], rdf: list[float]
) -> Any:
    figure, axes = figure_factory(figsize=(10, 6), dpi=100, constrained_layout=True)
    axes.plot(radius, rdf, color="#1864ab", linewidth=1.8)
    axes.axhline(1.0, color="#555555", linewidth=0.8, linestyle="--")
    axes.set_xlabel("r (Angstrom)")
    axes.set_ylabel("g_OO(r)")
    axes.set_title("O-O radial distribution function")
    axes.grid(True, linewidth=0.5, alpha=0.35)
    return figure


def _save_atomic_png(
    destination: Path,
    render: Callable[[], Any],
    pyplot: Any,
    image_reader: Callable[[str], Any],
) -> tuple[int, int, int]:
    descriptor, name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".png"
    )
    os.close(descriptor)
    temporary = Path(name)
    figure = None
    try:
        figure = render()
        figure.savefig(temporary, format="png", dpi=100, metadata={"Software": "IamLAMMPS"})
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        width, height, size = _validate_png(temporary, image_reader)
        os.replace(temporary, destination)
        return width, height, size
    finally:
        if figure is not None:
            pyplot.close(figure)
        temporary.unlink(missing_ok=True)


def _validate_png(path: Path, image_reader: Callable[[str], Any]) -> tuple[int, int, int]:
    size = path.stat().st_size
    if size <= 100:
        raise PlotDataError("generated PNG is empty")
    try:
        pixels = image_reader(str(path))
        height, width = int(pixels.shape[0]), int(pixels.shape[1])
        minimum = float(pixels.min())
        maximum = float(pixels.max())
    except Exception as exc:
        raise PlotDataError(f"generated PNG cannot be read: {exc}") from exc
    if width <= 0 or height <= 0 or not maximum > minimum:
        raise PlotDataError("generated PNG has no non-uniform pixels")
    return width, height, size


def _is_integer(value: str) -> bool:
    try:
        int(value)
    except ValueError:
        return False
    return True
