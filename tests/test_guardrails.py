from src.guardrails import is_safe


class FakeCategory:
    def __init__(self, severity):
        self.severity = severity


class FakeResult:
    def __init__(self, severities):
        self.categories_analysis = [FakeCategory(s) for s in severities]


class FakeClient:
    def __init__(self, severities):
        self._severities = severities

    def analyze_text(self, options):
        return FakeResult(self._severities)


def test_all_low_severity_is_safe():
    assert is_safe(FakeClient([0, 0, 1, 0]), "hello") is True


def test_one_category_at_threshold_is_unsafe():
    assert is_safe(FakeClient([0, 2, 0, 0]), "bad text") is False


def test_high_severity_is_unsafe():
    assert is_safe(FakeClient([6, 0, 0, 0]), "very bad") is False
