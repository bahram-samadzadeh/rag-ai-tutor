from src.verify_answer import verify_answer, check_claim


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, content):
        self._content = content

    def create(self, **kwargs):
        return FakeResponse(self._content)


class FakeChat:
    def __init__(self, content):
        self.completions = FakeCompletions(content)


class FakeClient:
    """Stands in for AzureOpenAI: always returns the given content."""

    def __init__(self, content):
        self.chat = FakeChat(content)


CHUNKS = [{"content": "Fabric is an analytics platform."}]


def test_all_claims_supported():
    client = FakeClient('{"claims": [{"claim": "Fabric is a platform.", "supported": true}]}')
    result = verify_answer(client, "some answer", CHUNKS)
    assert result["all_supported"] is True
    assert result["unsupported"] == []


def test_unsupported_claim_is_flagged():
    client = FakeClient(
        '{"claims": [{"claim": "A", "supported": true},'
        ' {"claim": "B", "supported": false}]}'
    )
    result = verify_answer(client, "some answer", CHUNKS)
    assert result["all_supported"] is False
    assert result["unsupported"] == ["B"]


def test_garbage_output_fails_safe():
    client = FakeClient("I am not JSON at all")
    result = verify_answer(client, "some answer", CHUNKS)
    assert result["all_supported"] is False  # never wrongly declares grounded


def test_check_claim_true():
    assert check_claim(FakeClient('{"supported": true}'), "A", CHUNKS) is True


def test_check_claim_garbage_fails_safe():
    assert check_claim(FakeClient("nonsense"), "A", CHUNKS) is False
