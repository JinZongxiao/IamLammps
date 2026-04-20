"""MLIP trainer hook — DeePMD / MACE / ACE.

Intended use: train interatomic potentials from DFT data,
then register them with the :class:`PotentialRegistry` for use in
LAMMPS workflows.

All methods raise :class:`NotImplementedError` — this module defines
the **interface** for a future MLIP training integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainingConfig:
    """Configuration for MLIP training."""

    framework: str = "deepmd"           # "deepmd" | "mace" | "ace"
    epochs: int = 100
    learning_rate: float = 1e-3
    batch_size: int = 32
    model_dir: Path | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationMetrics:
    """Accuracy metrics for a trained MLIP."""

    energy_rmse_eV: float = 0.0
    force_rmse_eV_A: float = 0.0
    stress_rmse_GPa: float = 0.0
    n_test_structures: int = 0


class MLIPTrainer:
    """Placeholder for ML interatomic potential training.

    Supported frameworks (placeholders):

    * **DeePMD-kit**: produces ``frozen_model.pb``
    * **MACE**: produces ``mace_model.pt``
    * **ACE**: produces ``ace.yace``

    Parameters
    ----------
    framework : str
        ML framework to use.
    """

    SUPPORTED_FRAMEWORKS = ("deepmd", "mace", "ace")

    def __init__(self, framework: str = "deepmd") -> None:
        if framework not in self.SUPPORTED_FRAMEWORKS:
            raise ValueError(
                f"Unknown framework: {framework!r}. "
                f"Supported: {self.SUPPORTED_FRAMEWORKS}"
            )
        self.framework = framework

    def prepare_dataset(
        self,
        dft_results: list[Any],
        output_dir: Path | None = None,
    ) -> Path:
        """Convert DFT results to framework-specific training format.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            f"MLIPTrainer.prepare_dataset() for {self.framework} "
            f"is not yet implemented."
        )

    def train(self, config: TrainingConfig | None = None) -> Path:
        """Train an MLIP model and return the model file path.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            f"MLIPTrainer.train() for {self.framework} "
            f"is not yet implemented."
        )

    def validate(
        self,
        model_path: Path,
        test_data: Any,
    ) -> ValidationMetrics:
        """Validate a trained model against test data.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            f"MLIPTrainer.validate() for {self.framework} "
            f"is not yet implemented."
        )

    def register_potential(
        self,
        model_path: Path,
        elements: list[str],
        name: str | None = None,
    ) -> Any:
        """Register a trained model as a PotentialEntry in the registry.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            f"MLIPTrainer.register_potential() for {self.framework} "
            f"is not yet implemented."
        )
