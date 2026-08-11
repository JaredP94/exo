"""DSML parser tolerance and value recovery for the DeepSeek V4 0731 protocol.

Break each test catches:

* whitespace_before_the_closing_bracket - the invoke/parameter patterns require
  `">` exactly, so a model emitting `name="f" >` produces no match at all and
  the whole tool-call block leaks into visible assistant content as text.
* padding_newline_is_stripped_from_a_string_value - models pad values with a
  newline before the closing tag; keeping it corrupts every string argument.
* intentional_surrounding_spaces_are_preserved - over-stripping (e.g. `.strip()`
  instead of one newline) silently rewrites the caller's data.
* python_literals_are_recovered - `True` currently arrives as the string
  "True", so a tool declaring a boolean parameter receives a string.
* a_decodable_but_unserializable_value_does_not_raise - `ast.literal_eval`
  accepts set literals, which `json.dumps` then rejects with TypeError.
  `parse_dsml_output` is called with no try/except from
  `model_output_parsers._try_parse_tool_call`, itself inside the runner's
  generation loop, so that would tear down generation rather than degrading to
  visible text.
* the rest are contract guards: multiple invokes, JSON recovery, undecodable
  text preserved, duplicate-name resolution, and the documented `None` return.
"""

import json
from typing import cast

import pytest

from exo.worker.engines.mlx.vendor.dsml_encoding import (
    DSML_TOKEN,
    parse_dsml_output,
)

DSML = DSML_TOKEN


def _invoke(name: str, params: str, *, name_suffix: str = "") -> str:
    return f'<{DSML}invoke name="{name}"{name_suffix}>\n{params}\n</{DSML}invoke>'


def _param(key: str, value: str, *, is_string: bool, suffix: str = "") -> str:
    flag = "true" if is_string else "false"
    return (
        f'<{DSML}parameter name="{key}" string="{flag}"{suffix}>'
        f"{value}"
        f"</{DSML}parameter>"
    )


def _arguments(text: str) -> dict[str, object]:
    calls = parse_dsml_output(text)
    assert calls is not None, "expected the block to parse"
    assert len(calls) == 1
    return cast("dict[str, object]", json.loads(calls[0].arguments))


# --------------------------------------------------------------------------- #
# Tolerance
# --------------------------------------------------------------------------- #


def test_whitespace_before_the_closing_bracket_of_an_invoke_tag() -> None:
    text = _invoke("f", _param("a", "1", is_string=False), name_suffix=" ")

    assert _arguments(text) == {"a": 1}


def test_whitespace_before_the_closing_bracket_of_a_parameter_tag() -> None:
    text = _invoke("f", _param("a", "1", is_string=False, suffix=" "))

    assert _arguments(text) == {"a": 1}


def test_two_invokes_in_one_block_yield_two_calls() -> None:
    text = (
        _invoke("first", _param("a", "1", is_string=False))
        + "\n"
        + _invoke("second", _param("b", "hello", is_string=True))
    )

    calls = parse_dsml_output(text)

    assert calls is not None
    assert [call.name for call in calls] == ["first", "second"]
    assert cast("dict[str, object]", json.loads(calls[1].arguments)) == {"b": "hello"}


# --------------------------------------------------------------------------- #
# Value decoding
# --------------------------------------------------------------------------- #


def test_padding_newline_is_stripped_from_a_string_value() -> None:
    text = _invoke("f", _param("a", "\nhello\n", is_string=True))

    assert _arguments(text) == {"a": "hello"}


def test_only_one_padding_newline_is_stripped_from_each_end() -> None:
    text = _invoke("f", _param("a", "\n\nhello\n\n", is_string=True))

    assert _arguments(text) == {"a": "\nhello\n"}


def test_intentional_surrounding_spaces_are_preserved() -> None:
    text = _invoke("f", _param("a", "  padded  ", is_string=True))

    assert _arguments(text) == {"a": "  padded  "}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("42", 42),
        ("-1.5", -1.5),
        ("true", True),
        ("false", False),
        ("null", None),
        ("[1, 2, 3]", [1, 2, 3]),
        ('{"k": "v"}', {"k": "v"}),
    ],
)
def test_json_values_are_recovered(raw: str, expected: object) -> None:
    text = _invoke("f", _param("a", raw, is_string=False))

    assert _arguments(text) == {"a": expected}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("True", True),
        ("False", False),
        ("None", None),
        ("(1, 2)", [1, 2]),
        ("{'k': 'v'}", {"k": "v"}),
    ],
)
def test_python_literals_are_recovered(raw: str, expected: object) -> None:
    text = _invoke("f", _param("a", raw, is_string=False))

    assert _arguments(text) == {"a": expected}


def test_undecodable_non_string_text_is_preserved_as_a_string() -> None:
    text = _invoke("f", _param("a", "not json at all", is_string=False))

    assert _arguments(text) == {"a": "not json at all"}


def test_a_decodable_but_unserializable_value_does_not_raise() -> None:
    """`ast.literal_eval` accepts a set; `json.dumps` rejects it.

    The caller has no try/except, so this must degrade to text rather than
    escaping the generator.
    """
    text = _invoke("f", _param("a", "{1, 2}", is_string=False))

    assert _arguments(text) == {"a": "{1, 2}"}


# --------------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------------- #


def test_duplicate_parameter_names_resolve_to_the_last_occurrence() -> None:
    """Matches OMLX, whose `_parse_single_invoke` assigns into a dict in order."""
    text = _invoke(
        "f",
        _param("a", "1", is_string=False) + "\n" + _param("a", "2", is_string=False),
    )

    assert _arguments(text) == {"a": 2}


def test_text_without_an_invoke_returns_none() -> None:
    assert parse_dsml_output("just prose, no tool call") is None


def test_string_parameters_are_never_decoded() -> None:
    text = _invoke("f", _param("a", "42", is_string=True))

    assert _arguments(text) == {"a": "42"}


# --------------------------------------------------------------------------- #
# Review fix round.
#
# * non_finite_numbers_degrade_to_text - `json.loads` accepts JSON's non-standard
#   `Infinity`/`-Infinity`/`NaN` extensions, and an overflowing literal like
#   `1e400` becomes `inf`. `json.dumps` then emits those tokens back, so
#   `ToolCallItem.arguments` is a string a strict client `JSON.parse` rejects.
#   The first guard missed this because the `json.loads` branch returned early.
# * an_oversized_integer_degrades_to_text - pins the deliberate divergence from
#   the OMLX reference: a value above CPython's int_max_str_digits limit makes
#   `json.loads` raise a bare ValueError, not a JSONDecodeError. Catching only
#   JSONDecodeError "for fidelity" would let it escape the generator.
# * reversed_parameter_attribute_order_is_a_known_limitation - documents that
#   the parameter pattern is attribute-order sensitive, exactly as OMLX's is.
#   NOT fixed by returning None on zero parameters: a legitimate zero-argument
#   tool call produces byte-identical output, so that would break every no-arg
#   tool. See the ledger for the two viable fixes.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("raw", ["Infinity", "-Infinity", "NaN", "1e400", "-1e400"])
def test_non_finite_numbers_degrade_to_text(raw: str) -> None:
    text = _invoke("f", _param("a", raw, is_string=False))

    assert _arguments(text) == {"a": raw}
    assert "Infinity" not in json.dumps(_arguments(text)).replace(raw, "")


def test_an_oversized_integer_degrades_to_text() -> None:
    raw = "1" * 5000

    text = _invoke("f", _param("a", raw, is_string=False))

    assert _arguments(text) == {"a": raw}


@pytest.mark.parametrize("raw", ["b'bytes'", "3+4j", "{1: {2}}", "[{1, 2}]"])
def test_every_unserializable_literal_type_degrades_to_text(raw: str) -> None:
    text = _invoke("f", _param("a", raw, is_string=False))

    assert _arguments(text) == {"a": raw}


def test_a_zero_argument_invoke_is_a_valid_tool_call() -> None:
    """Guards the fix for the limitation below from breaking no-arg tools."""
    calls = parse_dsml_output(f'<{DSML}invoke name="get_time">\n</{DSML}invoke>')

    assert calls is not None
    assert len(calls) == 1
    assert cast("dict[str, object]", json.loads(calls[0].arguments)) == {}


def test_reversed_parameter_attribute_order_is_a_known_limitation() -> None:
    """Pins current behaviour; matches OMLX. See the module docstring.

    The parameter is silently lost rather than degrading to visible text,
    because one matched invoke is enough for the caller to take the
    tool-call branch. Deliberately not "fixed" here: the output is
    indistinguishable from a legitimate zero-argument call.
    """
    text = _invoke(
        "get_weather",
        f'<{DSML}parameter string="true" name="city">Tokyo</{DSML}parameter>',
    )

    assert _arguments(text) == {}
