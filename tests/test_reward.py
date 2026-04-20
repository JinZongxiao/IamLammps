"""Tests for the reward scoring system."""

from __future__ import annotations

from pathlib import Path

from src.lammps_agent.reward import compute_reward, RewardSignal


def test_successful_run_reward(successful_run_dir: Path):
    reward = compute_reward(
        return_code=0,
        run_dir=successful_run_dir,
        retry_count=0,
        task="LJ fluid NVT equilibration 1000 steps",
    )
    assert isinstance(reward, RewardSignal)
    assert reward.script_validity == 1.0
    assert reward.efficiency == 1.0
    assert reward.convergence > 0.0
    assert reward.total > 0.5


def test_failed_run_reward(failed_run_dir: Path):
    reward = compute_reward(
        return_code=1,
        run_dir=failed_run_dir,
        retry_count=0,
        task="Create atoms",
    )
    assert reward.script_validity == 0.0
    assert reward.convergence == 0.0
    assert reward.total < 0.2


def test_retry_penalty(successful_run_dir: Path):
    r0 = compute_reward(return_code=0, run_dir=successful_run_dir, retry_count=0)
    r2 = compute_reward(return_code=0, run_dir=successful_run_dir, retry_count=2)
    assert r0.efficiency > r2.efficiency
    assert r0.total >= r2.total


def test_zero_temp_plausibility(zero_temp_run_dir: Path):
    reward = compute_reward(
        return_code=0,
        run_dir=zero_temp_run_dir,
        retry_count=0,
        task="NVT dynamics at 300K",
    )
    # T=0 for dynamics should hurt plausibility.
    assert reward.physical_plausibility < 1.0
