"""Tests for the model-produced LAMMPS script safety boundary."""

from __future__ import annotations

import pytest

from lammps_agent.script_policy import (
    ScriptPolicy,
    ScriptPolicyViolation,
    validate_lammps_script,
)


@pytest.mark.parametrize("command", ["shell rm -rf output", "python callback here"])
def test_rejects_host_code_commands(command: str) -> None:
    with pytest.raises(ScriptPolicyViolation) as error:
        validate_lammps_script(f"units metal\n{command}\nrun 10")

    assert error.value.line_number == 2
    assert error.value.command == command.split()[0]
    assert "line 2" in str(error.value)


@pytest.mark.parametrize(
    "command",
    [
        "include ../unreviewed.lammps",
        "jump /tmp/unreviewed.lammps next",
        "plugin load /tmp/arbitrary.so",
    ],
)
def test_rejects_external_script_and_plugin_loading(command: str) -> None:
    with pytest.raises(ScriptPolicyViolation, match="unsafe LAMMPS script"):
        validate_lammps_script(command)


@pytest.mark.parametrize(
    "command",
    [
        "variable callback python evaluate",
        "fix callback all python/invoke 1 end_of_step evaluate",
        "pair_style hybrid eam python model",
    ],
)
def test_rejects_indirect_python_callbacks(command: str) -> None:
    with pytest.raises(ScriptPolicyViolation, match="Python"):
        validate_lammps_script(command)


@pytest.mark.parametrize(
    "script",
    [
        'if "${temp} > 0" then "shell touch escaped"',
        'if "${temp} > 0" then "print safe" else "include other.in"',
        'run 100 every 10 "python callback here"',
    ],
)
def test_rejects_dangerous_nested_commands(script: str) -> None:
    with pytest.raises(ScriptPolicyViolation):
        validate_lammps_script(script)


@pytest.mark.parametrize(
    "script",
    [
        'if "${n} == 0" then "print safe" elif "${n} == 1" "shell touch escaped" else "print safe"',
        'if "${n} == 0" then "print safe" elif "${n} == 1" '
        '"if \'1 == 1\' then \'python callback here\'"',
    ],
)
def test_rejects_dangerous_elif_and_nested_if_commands(script: str) -> None:
    with pytest.raises(ScriptPolicyViolation):
        validate_lammps_script(script)


@pytest.mark.parametrize(
    "script",
    [
        "variable token getenv OPENAI_API_KEY",
        "read_data ${unreviewed_path}",
        "pair_coeff * * ${potential_file} W",
        'dump d all custom 10 "|gzip > dump.gz" id type x y z',
        "write_dump all custom output.gz id type x y z",
    ],
)
def test_rejects_environment_dynamic_paths_pipes_and_compressors(script: str) -> None:
    with pytest.raises(ScriptPolicyViolation):
        validate_lammps_script(script)


def test_file_tokens_in_inert_strings_are_not_mistaken_for_external_io() -> None:
    assert validate_lammps_script('print "report.gz | not a pipe"') is None


@pytest.mark.parametrize(
    "script",
    [
        'if "${n} == 0" "print missing-then"',
        'if "${n} == 0" then "print safe" elif "${n} == 1"',
        'if "${n} == 0" then "print safe" else',
    ],
)
def test_rejects_malformed_control_flow(script: str) -> None:
    with pytest.raises(ScriptPolicyViolation, match="if|elif|else|branch"):
        validate_lammps_script(script)


def test_rejects_variable_expanded_command_name() -> None:
    with pytest.raises(ScriptPolicyViolation, match="cannot be validated statically"):
        validate_lammps_script("${command} arbitrary arguments")


@pytest.mark.parametrize(
    "script",
    [
        "read_data /tmp/stolen.data",
        "write_restart ../../escaped.restart",
        "pair_coeff * * ~/private.eam W",
    ],
)
def test_rejects_paths_outside_run_directory(script: str) -> None:
    with pytest.raises(ScriptPolicyViolation, match="paths"):
        validate_lammps_script(script)


def test_accepts_normal_scientific_script_and_quoted_words() -> None:
    script = '''
        # The word "shell" in a comment is inert.
        units metal
        atom_style atomic
        variable message string "python and shell are disabled"
        print "shell is not a command here # still quoted"
        pair_style eam/alloy
        pair_coeff * * W.eam.alloy W
        minimize 1e-10 1e-10 1000 10000
    '''

    assert validate_lammps_script(script) is None


def test_checks_continued_command_and_reports_starting_line() -> None:
    script = "units metal\nfix callback all &\n  python/invoke 1 end_of_step evaluate\n"

    with pytest.raises(ScriptPolicyViolation) as error:
        validate_lammps_script(script)

    assert error.value.line_number == 2


def test_rejects_malformed_quotes_and_unfinished_continuation() -> None:
    with pytest.raises(ScriptPolicyViolation, match="cannot be parsed safely"):
        validate_lammps_script('print "unterminated')

    with pytest.raises(ScriptPolicyViolation, match="unfinished"):
        validate_lammps_script("units metal &")


def test_policy_can_add_project_specific_blocked_commands() -> None:
    policy = ScriptPolicy(blocked_commands=frozenset({"quit"}))

    with pytest.raises(ScriptPolicyViolation) as error:
        policy.validate("quit 1")

    assert error.value.command == "quit"


def test_requires_text_input() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        validate_lammps_script(None)  # type: ignore[arg-type]
