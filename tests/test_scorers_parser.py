import pytest

from groundwork.agent.parser import ParseError, parse_step
from groundwork.data.scorers import extract_answer, is_abstention, score
from groundwork.data.tasks import Task


def test_extract_answer_takes_last_answer_line():
    assert extract_answer("Some reasoning.\nAnswer: 12\nmore\nAnswer: 42") == "42"
    assert extract_answer("just text") == "just text"


@pytest.mark.parametrize("pred,gold,kind,expected", [
    ("The total is $1,120.53", "1120.53", "numeric", 1.0),
    ("Answer: 3/4", "0.75", "numeric", 1.0),
    ("Answer: 41.9", "42", "numeric", 0.0),
    ("Answer: Paris.", "paris", "exact", 1.0),
    ("Answer: The Paris", "Paris", "exact", 1.0),
    ("It is Mount Kilimanjaro in Tanzania", "Kilimanjaro", "contains", 1.0),
    ("Answer: (B)", "B", "choice", 1.0),
    ("Answer: 1991", "1991", "exact", 1.0),
])
def test_score(pred, gold, kind, expected):
    assert score(Task(id="t", question="q", answer=gold, answer_type=kind), pred) == expected


def test_code_scoring():
    task = Task(id="c", question="q", answer_type="code", tests="assert add(2, 3) == 5")
    assert score(task, "```python\ndef add(a, b):\n    return a + b\n```") == 1.0
    assert score(task, "```python\ndef add(a, b):\n    return a - b\n```") == 0.0


def test_abstention():
    assert is_abstention("I'm not sure about that, and I'd rather not guess.")
    assert is_abstention("Answer: unknown")
    assert not is_abstention("The answer is 42.")


def test_parse_step_canonical():
    s = parse_step("Thought: add numbers\nCode:\n```py\nx = 1 + 1\nprint(x)\n```<end_code>")
    assert s.thought == "add numbers" and s.code == "x = 1 + 1\nprint(x)"
    assert s.render().endswith("```<end_code>")
    # stop sequence removed the closing fence
    s2 = parse_step("Thought: t\nCode:\n```python\nprint(2)\n")
    assert s2.code == "print(2)"


def test_parse_errors():
    with pytest.raises(ParseError, match="no code block"):
        parse_step("Thought: I will just answer 42")
    with pytest.raises(ParseError, match="SyntaxError"):
        parse_step("Thought: t\nCode:\n```py\nprint(\n```")
