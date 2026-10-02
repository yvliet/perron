"""
Unit tests for agent multi-turn history compaction and XML preservation.
"""

from perron.agent import PerronAgent


def test_history_compaction_preserves_xml_edits():
    raw_content = (
        "Here is my reasoning. I will now edit the target file to fix the bug.\n\n"
        '<edit file="src/math.py" start_line="10" end_line="15">\n'
        "<old>\ndef divide(a, b):\n    return a / b\n</old>\n"
        "<new>\ndef divide(a, b):\n    if b == 0:\n        raise ZeroDivisionError('Cannot divide by zero')\n    return a / b\n</new>\n"
        "</edit>\n\n"
        "Let me know if further tests are needed."
    )

    compacted = PerronAgent.compact_history_content(raw_content)

    # Conversational filler is discarded
    assert "Here is my reasoning" not in compacted
    assert "Let me know if further tests are needed" not in compacted

    # The edit block remains 100% intact
    assert '<edit file="src/math.py" start_line="10" end_line="15">' in compacted
    assert "<old>\ndef divide(a, b):\n    return a / b\n</old>" in compacted
    assert "</edit>" in compacted


def test_history_compaction_strips_thinking_tags():
    raw_content = (
        "<|think|>\nInvestigating edge cases in division.\nChecking if b is float or int.\n<|/think|>\n"
        '<edit file="core/calc.py">\n'
        "<old>val = 1</old>\n"
        "<new>val = 2</new>\n"
        "</edit>"
    )

    compacted = PerronAgent.compact_history_content(raw_content)

    assert "<|think|>" not in compacted
    assert "<|/think|>" not in compacted
    assert "Investigating edge cases" not in compacted
    assert '<edit file="core/calc.py">' in compacted
    assert "<new>val = 2</new>" in compacted


def test_history_compaction_fallback_without_edits():
    raw_content = "This is a long conversational response without any XML edit blocks. " * 30
    assert len(raw_content) > 800

    compacted = PerronAgent.compact_history_content(raw_content)

    assert len(compacted) <= 850
    assert "...[truncated historical edit]..." in compacted
    assert "<edit" not in compacted


def test_history_compaction_massive_edit_truncation():
    massive_code = "x = 1\n" * 600
    raw_content = (
        '<edit file="huge.py">\n'
        "<old>pass</old>\n"
        f"<new>\n{massive_code}</new>\n"
        "</edit>"
    )
    assert len(raw_content) > 2500

    compacted = PerronAgent.compact_history_content(raw_content)

    assert len(compacted) <= 2600
    assert "</new>\n</edit>\n...[truncated historical edit]..." in compacted
