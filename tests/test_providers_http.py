import httpx

from jobrec.providers.http_chat import HttpChatProvider


def _provider(handler) -> HttpChatProvider:
    p = HttpChatProvider("http://llm.test/v1", "demo-model", api_key="not-a-real-key")
    p._client = httpx.Client(transport=httpx.MockTransport(handler), timeout=5)
    return p


def test_http_provider_parses_chat_completion():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"choices": [{"message": {"content": "  Why? "}}]})

    out = _provider(handler).complete([{"role": "user", "content": "x"}])
    assert out == "Why?"
    assert seen["url"] == "http://llm.test/v1/chat/completions"
    assert seen["auth"] == "Bearer not-a-real-key"


def test_http_provider_fails_soft():
    def boom(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="down")

    assert _provider(boom).complete([{"role": "user", "content": "x"}]) is None

    def garbage(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    assert _provider(garbage).complete([{"role": "user", "content": "x"}]) is None
