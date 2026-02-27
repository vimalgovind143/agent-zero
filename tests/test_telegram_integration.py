import sys
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import telegram module for accessing internal state
from python.helpers import telegram as telegram_module


class _FakeReq:
    def __init__(self, json_payload: dict, headers: dict[str, str] | None = None):
        self._payload = json_payload
        self._headers = headers or {}
        self.is_json = True

    def get_json(self, force=False, silent=False):
        return self._payload

    @property
    def headers(self):
        return self._headers


def _install_test_stubs():
    flask_stub = SimpleNamespace()

    class _Response:
        def __init__(self, response=None, status=None, mimetype=None):
            self.response = response
            self.status = status
            self.mimetype = mimetype

    class _Request:
        pass

    class _Flask:
        def __init__(self, *args, **kwargs):
            pass

    flask_stub.Request = _Request
    flask_stub.Response = _Response
    flask_stub.jsonify = lambda *_, **__: {}
    flask_stub.Flask = _Flask
    flask_stub.session = {}
    flask_stub.request = SimpleNamespace()
    flask_stub.send_file = lambda *_, **__: None
    sys.modules.setdefault("flask", flask_stub)

    simpleeval_stub = SimpleNamespace(simple_eval=lambda *_args, **_kwargs: None)
    sys.modules.setdefault("simpleeval", simpleeval_stub)

    sys.modules.setdefault("webcolors", SimpleNamespace())
    sys.modules.setdefault("whisper", SimpleNamespace(load_model=lambda **_: object()))

    werkzeug_serving_stub = SimpleNamespace(make_server=lambda *_, **__: None)
    sys.modules.setdefault("werkzeug", SimpleNamespace(serving=werkzeug_serving_stub))
    sys.modules.setdefault("werkzeug.serving", werkzeug_serving_stub)

    agent_stub = SimpleNamespace()

    class _UserMessage:
        def __init__(self, content, attachments):
            self.content = content
            self.attachments = attachments

    class _AgentContextType:
        USER = "user"

    class _FakeAgentContext:
        @staticmethod
        def use(_cid):
            return None

        @staticmethod
        def get(_cid):
            return None

        @staticmethod
        def remove(_cid):
            pass

    agent_stub.UserMessage = _UserMessage
    agent_stub.AgentContextType = _AgentContextType
    agent_stub.AgentContext = _FakeAgentContext
    sys.modules.setdefault("agent", agent_stub)

    init_stub = SimpleNamespace(initialize_agent=lambda **_: object())
    sys.modules.setdefault("initialize", init_stub)

    models_stub = SimpleNamespace(get_api_key=lambda *_: "")
    sys.modules.setdefault("models", models_stub)


@pytest.mark.asyncio
async def test_telegram_webhook_auth_and_routing(monkeypatch):
    _install_test_stubs()
    from python.api.telegram_webhook import TelegramWebhook
    from python.helpers import settings as settings_helper

    base = settings_helper.get_settings()
    base.update(
        {
            "telegram_enabled": True,
            "telegram_mode": "webhook",
            "telegram_allowed_chat_ids": "12345",
        }
    )
    settings_helper.set_settings(base, apply=False)

    monkeypatch.setattr(
        "python.helpers.dotenv.get_dotenv_value",
        lambda k, d=None: {
            "TELEGRAM_BOT_TOKEN": "token",
            "TELEGRAM_WEBHOOK_SECRET": "secret",
        }.get(k, d),
    )

    class _FakeContext:
        id = "tg-12345"

        class _Log:
            def log(self, **_):
                pass

        log = _Log()

        def communicate(self, _msg):
            class _Task:
                async def result(self):
                    return "ok-response"

            return _Task()

    import agent as agent_module

    agent_module.AgentContext.use = lambda cid: _FakeContext()

    sent = {"calls": []}

    def fake_post(url, json=None, timeout=10):
        sent["calls"].append({"url": url, "payload": json})

        class _R:
            pass

        return _R()

    monkeypatch.setattr("python.api.telegram_webhook.requests.post", fake_post)

    handler = TelegramWebhook(SimpleNamespace(), SimpleNamespace())
    payload = {"message": {"chat": {"id": 12345}, "text": "hi"}}
    req = _FakeReq(payload, headers={"X-Telegram-Bot-Api-Secret-Token": "secret"})
    res = await handler.process({}, req)
    assert isinstance(res, dict) and res.get("ok") is True
    send_message_calls = [
        c
        for c in sent["calls"]
        if isinstance(c.get("payload"), dict)
        and c["payload"].get("chat_id") == 12345
        and c["payload"].get("text") == "ok-response"
    ]
    assert len(send_message_calls) == 1

    req_bad = _FakeReq(payload, headers={"X-Telegram-Bot-Api-Secret-Token": "bad"})
    resp = await handler.process({}, req_bad)
    assert getattr(resp, "status", None) == 401

    payload2 = {"message": {"chat": {"id": 999}, "text": "hi"}}
    req2 = _FakeReq(payload2, headers={"X-Telegram-Bot-Api-Secret-Token": "secret"})
    resp2 = await handler.process({}, req2)
    assert getattr(resp2, "status", None) == 403


@pytest.mark.asyncio
async def test_telegram_polling_handle_update_sends_reply(monkeypatch):
    _install_test_stubs()
    from python.helpers import settings as settings_helper

    cfg = settings_helper.get_settings()
    cfg.update(
        {
            "telegram_enabled": True,
            "telegram_mode": "polling",
            "telegram_allowed_chat_ids": "12345",
        }
    )

    monkeypatch.setattr(
        "python.helpers.dotenv.get_dotenv_value",
        lambda k, d=None: {
            "TELEGRAM_BOT_TOKEN": "token",
            "TELEGRAM_WEBHOOK_SECRET": "secret",
        }.get(k, d),
    )

    sent = {"calls": []}

    async def fake_send(token, chat_id, text, attachments=None):
        sent["calls"].append({"chat_id": chat_id, "text": text})

    monkeypatch.setattr("python.helpers.telegram._send_telegram_message", fake_send)

    class _FakeContext:
        id = "tg-12345"

        class _Log:
            def log(self, **_):
                pass

        log = _Log()

        def communicate(self, _msg):
            class _Task:
                async def result(self):
                    return "pong"

            return _Task()

    import agent as agent_module

    monkeypatch.setattr(agent_module.AgentContext, "use", lambda cid: _FakeContext())

    # Clear rate limits
    telegram_module._telegram_rate_limits.clear()

    update = {"update_id": 1, "message": {"chat": {"id": 12345}, "text": "hello"}}
    await telegram_module._handle_telegram_update(update, "token", cfg)
    
    # Wait for async queue processing - need longer wait for agent communication
    await asyncio.sleep(1.0)

    send_message_calls = [
        c for c in sent["calls"]
        if c.get("chat_id") == 12345 and c.get("text") == "pong"
    ]
    assert len(send_message_calls) >= 1


@pytest.mark.asyncio
async def test_telegram_start_command(monkeypatch):
    _install_test_stubs()
    import agent as agent_module

    class _FakeAgentContext:
        @staticmethod
        def use(_cid):
            return None

        @staticmethod
        def get(_cid):
            return None

        @staticmethod
        def remove(_cid):
            pass

    agent_module.AgentContext = _FakeAgentContext

    from python.helpers.telegram import _handle_telegram_command

    sent = {}

    async def fake_send(token, chat_id, text, attachments=None):
        sent["payload"] = {"chat_id": chat_id, "text": text}

    monkeypatch.setattr("python.helpers.telegram._send_telegram_message", fake_send)

    update = {"message": {"chat": {"id": 12345}, "text": "/start"}}
    await _handle_telegram_command(update, "token", 12345)

    assert "Welcome to Agent Zero!" in sent.get("payload", {}).get("text", "")


@pytest.mark.asyncio
async def test_telegram_help_command(monkeypatch):
    _install_test_stubs()
    import agent as agent_module

    class _FakeAgentContext:
        @staticmethod
        def use(_cid):
            return None

        @staticmethod
        def get(_cid):
            return None

        @staticmethod
        def remove(_cid):
            pass

    agent_module.AgentContext = _FakeAgentContext

    from python.helpers.telegram import _handle_telegram_command

    sent = {}

    async def fake_send(token, chat_id, text, attachments=None):
        sent["payload"] = {"chat_id": chat_id, "text": text}

    monkeypatch.setattr("python.helpers.telegram._send_telegram_message", fake_send)

    update = {"message": {"chat": {"id": 12345}, "text": "/help"}}
    await _handle_telegram_command(update, "token", 12345)

    assert "Agent Zero Commands" in sent.get("payload", {}).get("text", "")


@pytest.mark.asyncio
async def test_telegram_reset_command(monkeypatch):
    _install_test_stubs()
    import agent as agent_module

    class _FakeAgentContext:
        @staticmethod
        def use(_cid):
            return _FakeContext()

        @staticmethod
        def get(_cid):
            return None

        @staticmethod
        def remove(_cid):
            pass

    agent_module.AgentContext = _FakeAgentContext

    from python.helpers.telegram import _handle_telegram_command

    class _FakeContext:
        id = "tg-12345"

        def reset(self):
            pass

        class _Log:
            def log(self, **_):
                pass

        log = _Log()

    sent = {}

    async def fake_send(token, chat_id, text, attachments=None):
        sent["payload"] = {"chat_id": chat_id, "text": text}

    monkeypatch.setattr("python.helpers.telegram._send_telegram_message", fake_send)

    update = {"message": {"chat": {"id": 12345}, "text": "/reset"}}
    await _handle_telegram_command(update, "token", 12345)

    assert "Conversation reset" in sent.get("payload", {}).get("text", "")


@pytest.mark.asyncio
async def test_telegram_unknown_command(monkeypatch):
    _install_test_stubs()
    import agent as agent_module

    class _FakeAgentContext:
        @staticmethod
        def use(_cid):
            return None

        @staticmethod
        def get(_cid):
            return None

        @staticmethod
        def remove(_cid):
            pass

    agent_module.AgentContext = _FakeAgentContext

    from python.helpers.telegram import _handle_telegram_command

    sent = {}

    async def fake_send(token, chat_id, text, attachments=None):
        sent["payload"] = {"chat_id": chat_id, "text": text}

    monkeypatch.setattr("python.helpers.telegram._send_telegram_message", fake_send)

    update = {"message": {"chat": {"id": 12345}, "text": "/unknowncmd"}}
    await _handle_telegram_command(update, "token", 12345)

    assert "Unknown command" in sent.get("payload", {}).get("text", "")


@pytest.mark.asyncio
async def test_telegram_webhook_rate_limiting(monkeypatch):
    _install_test_stubs()
    from python.api.telegram_webhook import (
        TelegramWebhook,
        _webhook_rate_limits,
    )
    from python.helpers import settings as settings_helper

    _webhook_rate_limits.clear()

    base = settings_helper.get_settings()
    base.update(
        {
            "telegram_enabled": True,
            "telegram_mode": "webhook",
            "telegram_allowed_chat_ids": "12345",
        }
    )
    settings_helper.set_settings(base, apply=False)

    def mock_get_dotenv(k, d=None):
        if k == "TELEGRAM_BOT_TOKEN":
            return "token"
        if k == "TELEGRAM_WEBHOOK_SECRET":
            return "secret"
        return d

    monkeypatch.setattr("python.helpers.dotenv.get_dotenv_value", mock_get_dotenv)

    class _FakeContext:
        id = "tg-12345"

        class _Log:
            def log(self, **_):
                pass

        log = _Log()

        def communicate(self, _msg):
            class _Task:
                async def result(self):
                    return "ok"

            return _Task()

    import agent as agent_module

    original_use = agent_module.AgentContext.use
    agent_module.AgentContext.use = lambda cid: _FakeContext()

    sent = {}

    def fake_post(url, json=None, timeout=10):
        sent["payload"] = json

        class _R:
            pass

        return _R()

    monkeypatch.setattr("python.api.telegram_webhook.requests.post", fake_post)

    handler = TelegramWebhook(SimpleNamespace(), SimpleNamespace())
    payload = {"message": {"chat": {"id": 12345}, "text": "hi"}}
    req = _FakeReq(payload, headers={"X-Telegram-Bot-Api-Secret-Token": "secret"})

    await handler.process({}, req)
    resp = await handler.process({}, req)
    assert getattr(resp, "status", None) == 429


@pytest.mark.asyncio
async def test_telegram_polling_rate_limiting(monkeypatch):
    _install_test_stubs()
    from python.helpers import settings as settings_helper

    cfg = settings_helper.get_settings()
    cfg.update(
        {
            "telegram_enabled": True,
            "telegram_mode": "polling",
            "telegram_allowed_chat_ids": "12345",
        }
    )

    monkeypatch.setattr(
        "python.helpers.dotenv.get_dotenv_value",
        lambda k, d=None: {
            "TELEGRAM_BOT_TOKEN": "token",
        }.get(k, d),
    )

    sent = {}

    async def fake_send(token, chat_id, text, attachments=None):
        sent["payload"] = {"chat_id": chat_id, "text": text}

    monkeypatch.setattr("python.helpers.telegram._send_telegram_message", fake_send)

    class _FakeContext:
        id = "tg-12345"

        class _Log:
            def log(self, **_):
                pass

        log = _Log()

        def communicate(self, _msg):
            class _Task:
                async def result(self):
                    return "pong"

            return _Task()

    import agent as agent_module

    monkeypatch.setattr(agent_module.AgentContext, "use", lambda cid: _FakeContext())

    # Clear rate limits
    telegram_module._telegram_rate_limits.clear()

    update = {"update_id": 1, "message": {"chat": {"id": 12345}, "text": "hello"}}
    await telegram_module._handle_telegram_update(update, "token", cfg)
    
    # Wait for async queue processing - need longer wait for agent communication
    await asyncio.sleep(1.0)
    
    # Second call should be rate limited
    await telegram_module._handle_telegram_update(update, "token", cfg)

    assert sent.get("payload", {}).get("text") == "pong"


def test_telegram_chat_cleanup(monkeypatch):
    _install_test_stubs()
    import agent as agent_module
    from datetime import datetime, timedelta

    telegram_module._telegram_chat_lifetimes["tg-12345"] = datetime.now() - timedelta(
        hours=1
    )
    telegram_module._telegram_chat_lifetimes["tg-67890"] = datetime.now() + timedelta(
        hours=1
    )

    class _FakeContext:
        id = "tg-12345"

        def reset(self):
            pass

    monkeypatch.setattr(
        agent_module.AgentContext,
        "get",
        lambda cid: _FakeContext() if cid == "tg-12345" else None,
    )
    monkeypatch.setattr(agent_module.AgentContext, "remove", lambda _: None)

    telegram_module._telegram_cleanup_expired_chats()

    assert "tg-12345" not in telegram_module._telegram_chat_lifetimes
    assert "tg-67890" in telegram_module._telegram_chat_lifetimes


@pytest.mark.asyncio
async def test_telegram_polling_commands(monkeypatch):
    _install_test_stubs()
    from python.helpers import settings as settings_helper

    cfg = settings_helper.get_settings()
    cfg.update(
        {
            "telegram_enabled": True,
            "telegram_mode": "polling",
            "telegram_allowed_chat_ids": "12345",
        }
    )

    monkeypatch.setattr(
        "python.helpers.dotenv.get_dotenv_value",
        lambda k, d=None: {
            "TELEGRAM_BOT_TOKEN": "token",
        }.get(k, d),
    )

    sent = {}

    async def fake_send(token, chat_id, text, attachments=None):
        sent["payload"] = {"chat_id": chat_id, "text": text}

    monkeypatch.setattr("python.helpers.telegram._send_telegram_message", fake_send)

    # Clear rate limits
    telegram_module._telegram_rate_limits.clear()

    update = {"update_id": 1, "message": {"chat": {"id": 12345}, "text": "/start"}}
    await telegram_module._handle_telegram_update(update, "token", cfg)
    
    # Wait for async processing
    await asyncio.sleep(0.2)

    assert "Welcome to Agent Zero!" in sent.get("payload", {}).get("text", "")


@pytest.mark.asyncio
async def test_telegram_polling_reset_command(monkeypatch):
    _install_test_stubs()
    from python.helpers import settings as settings_helper
    import agent as agent_module

    cfg = settings_helper.get_settings()
    cfg.update(
        {
            "telegram_enabled": True,
            "telegram_mode": "polling",
            "telegram_allowed_chat_ids": "12345",
        }
    )

    monkeypatch.setattr(
        "python.helpers.dotenv.get_dotenv_value",
        lambda k, d=None: {
            "TELEGRAM_BOT_TOKEN": "token",
        }.get(k, d),
    )

    reset_called = {}

    class _FakeContext:
        id = "tg-12345"

        def reset(self):
            reset_called["reset"] = True

        class _Log:
            def log(self, **_):
                pass

        log = _Log()

    monkeypatch.setattr(agent_module.AgentContext, "use", lambda cid: _FakeContext())

    sent = {}

    async def fake_send(token, chat_id, text, attachments=None):
        sent["payload"] = {"chat_id": chat_id, "text": text}

    monkeypatch.setattr("python.helpers.telegram._send_telegram_message", fake_send)

    # Clear rate limits
    telegram_module._telegram_rate_limits.clear()

    update = {"update_id": 1, "message": {"chat": {"id": 12345}, "text": "/reset"}}
    await telegram_module._handle_telegram_update(update, "token", cfg)
    
    # Wait for async processing
    await asyncio.sleep(0.2)

    assert "Conversation reset" in sent.get("payload", {}).get("text", "")


def test_telegram_settings_telegram_chat_lifetime_updated(monkeypatch):
    _install_test_stubs()
    import agent as agent_module
    from datetime import datetime

    telegram_module._telegram_chat_lifetimes.clear()

    from python.helpers import settings as settings_helper
    cfg = settings_helper.get_settings()
    cfg.update(
        {
            "telegram_enabled": True,
            "telegram_mode": "polling",
            "telegram_allowed_chat_ids": "12345",
        }
    )

    monkeypatch.setattr(
        "python.helpers.dotenv.get_dotenv_value",
        lambda k, d=None: {
            "TELEGRAM_BOT_TOKEN": "token",
        }.get(k, d),
    )

    async def fake_send(token, chat_id, text, attachments=None):
        pass

    monkeypatch.setattr("python.helpers.telegram._send_telegram_message", fake_send)

    class _FakeContext:
        id = "tg-12345"

        class _Log:
            def log(self, **_):
                pass

        log = _Log()

        def communicate(self, _msg):
            class _Task:
                async def result(self):
                    return "ok"

            return _Task()

    monkeypatch.setattr(agent_module.AgentContext, "use", lambda cid: _FakeContext())

    import asyncio

    asyncio.run(
        telegram_module._process_telegram_message(
            {"message": {"text": "hello"}}, "token", cfg, 12345
        )
    )

    assert "tg-12345" in telegram_module._telegram_chat_lifetimes
    assert telegram_module._telegram_chat_lifetimes["tg-12345"] > datetime.now()


def test_telegram_disabled_returns_404(monkeypatch):
    _install_test_stubs()
    from python.api.telegram_webhook import TelegramWebhook
    from python.helpers import settings as settings_helper

    base = settings_helper.get_settings()
    base.update(
        {
            "telegram_enabled": False,
            "telegram_mode": "webhook",
        }
    )
    settings_helper.set_settings(base, apply=False)

    handler = TelegramWebhook(SimpleNamespace(), SimpleNamespace())
    payload = {"message": {"chat": {"id": 12345}, "text": "hi"}}
    req = _FakeReq(payload)

    import asyncio

    resp = asyncio.run(handler.process({}, req))

    assert getattr(resp, "status", None) == 404


def test_telegram_wrong_mode_returns_400(monkeypatch):
    _install_test_stubs()
    from python.api.telegram_webhook import TelegramWebhook
    from python.helpers import settings as settings_helper

    base = settings_helper.get_settings()
    base.update(
        {
            "telegram_enabled": True,
            "telegram_mode": "polling",
        }
    )
    settings_helper.set_settings(base, apply=False)

    handler = TelegramWebhook(SimpleNamespace(), SimpleNamespace())
    payload = {"message": {"chat": {"id": 12345}, "text": "hi"}}
    req = _FakeReq(payload)

    import asyncio

    resp = asyncio.run(handler.process({}, req))

    assert getattr(resp, "status", None) == 400


def test_telegram_no_token_returns_400(monkeypatch):
    _install_test_stubs()
    from python.api.telegram_webhook import TelegramWebhook
    from python.helpers import settings as settings_helper

    base = settings_helper.get_settings()
    base.update(
        {
            "telegram_enabled": True,
            "telegram_mode": "webhook",
        }
    )
    settings_helper.set_settings(base, apply=False)

    monkeypatch.setattr(
        "python.helpers.dotenv.get_dotenv_value", lambda k, d=None: None
    )

    handler = TelegramWebhook(SimpleNamespace(), SimpleNamespace())
    payload = {"message": {"chat": {"id": 12345}, "text": "hi"}}
    req = _FakeReq(payload)

    import asyncio

    resp = asyncio.run(handler.process({}, req))

    assert getattr(resp, "status", None) == 400


def test_telegram_unauthorized_missing_secret(monkeypatch):
    _install_test_stubs()
    from python.api.telegram_webhook import TelegramWebhook
    from python.helpers import settings as settings_helper

    base = settings_helper.get_settings()
    base.update(
        {
            "telegram_enabled": True,
            "telegram_mode": "webhook",
        }
    )
    settings_helper.set_settings(base, apply=False)

    monkeypatch.setattr(
        "python.helpers.dotenv.get_dotenv_value",
        lambda k, d=None: {
            "TELEGRAM_BOT_TOKEN": "token",
            "TELEGRAM_WEBHOOK_SECRET": "secret",
        }.get(k, d),
    )

    handler = TelegramWebhook(SimpleNamespace(), SimpleNamespace())
    payload = {"message": {"chat": {"id": 12345}, "text": "hi"}}
    req = _FakeReq(payload, headers={})

    import asyncio

    resp = asyncio.run(handler.process({}, req))

    assert getattr(resp, "status", None) == 401


def test_telegram_empty_message_ignored(monkeypatch):
    _install_test_stubs()
    from python.api.telegram_webhook import TelegramWebhook
    from python.helpers import settings as settings_helper

    base = settings_helper.get_settings()
    base.update(
        {
            "telegram_enabled": True,
            "telegram_mode": "webhook",
            "telegram_allowed_chat_ids": "12345",
        }
    )
    settings_helper.set_settings(base, apply=False)

    monkeypatch.setattr(
        "python.helpers.dotenv.get_dotenv_value",
        lambda k, d=None: {
            "TELEGRAM_BOT_TOKEN": "token",
        }.get(k, d),
    )

    handler = TelegramWebhook(SimpleNamespace(), SimpleNamespace())
    payload = {"message": {"chat": {"id": 12345}, "text": ""}}
    req = _FakeReq(payload)

    import asyncio

    resp = asyncio.run(handler.process({}, req))

    assert getattr(resp, "status", None) == 200


def test_telegram_caption_as_message(monkeypatch):
    _install_test_stubs()
    from python.api.telegram_webhook import TelegramWebhook
    from python.helpers import settings as settings_helper

    base = settings_helper.get_settings()
    base.update(
        {
            "telegram_enabled": True,
            "telegram_mode": "webhook",
            "telegram_allowed_chat_ids": "12345",
        }
    )
    settings_helper.set_settings(base, apply=False)

    monkeypatch.setattr(
        "python.helpers.dotenv.get_dotenv_value",
        lambda k, d=None: {
            "TELEGRAM_BOT_TOKEN": "token",
        }.get(k, d),
    )

    class _FakeContext:
        id = "tg-12345"

        class _Log:
            def log(self, **_):
                pass

        log = _Log()

        def communicate(self, _msg):
            class _Task:
                async def result(self):
                    return "ok"

            return _Task()

    import agent as agent_module

    monkeypatch.setattr(agent_module.AgentContext, "use", lambda cid: _FakeContext())

    sent = {}

    def fake_post(url, json=None, timeout=10):
        sent["payload"] = json

        class _R:
            pass

        return _R()

    monkeypatch.setattr("python.api.telegram_webhook.requests.post", fake_post)

    handler = TelegramWebhook(SimpleNamespace(), SimpleNamespace())
    payload = {"message": {"chat": {"id": 12345}, "caption": "image caption"}}
    req = _FakeReq(payload)

    import asyncio

    resp = asyncio.run(handler.process({}, req))

    assert isinstance(resp, dict) and resp.get("ok") is True


def test_telegram_edited_message_handled(monkeypatch):
    _install_test_stubs()
    from python.api.telegram_webhook import TelegramWebhook
    from python.helpers import settings as settings_helper

    base = settings_helper.get_settings()
    base.update(
        {
            "telegram_enabled": True,
            "telegram_mode": "webhook",
            "telegram_allowed_chat_ids": "12345",
        }
    )
    settings_helper.set_settings(base, apply=False)

    monkeypatch.setattr(
        "python.helpers.dotenv.get_dotenv_value",
        lambda k, d=None: {
            "TELEGRAM_BOT_TOKEN": "token",
        }.get(k, d),
    )

    class _FakeContext:
        id = "tg-12345"

        class _Log:
            def log(self, **_):
                pass

        log = _Log()

        def communicate(self, _msg):
            class _Task:
                async def result(self):
                    return "ok"

            return _Task()

    import agent as agent_module

    monkeypatch.setattr(agent_module.AgentContext, "use", lambda cid: _FakeContext())

    sent = {}

    def fake_post(url, json=None, timeout=10):
        sent["payload"] = json

        class _R:
            pass

        return _R()

    monkeypatch.setattr("python.api.telegram_webhook.requests.post", fake_post)

    handler = TelegramWebhook(SimpleNamespace(), SimpleNamespace())
    payload = {"edited_message": {"chat": {"id": 12345}, "text": "edited text"}}
    req = _FakeReq(payload)

    import asyncio

    resp = asyncio.run(handler.process({}, req))

    assert isinstance(resp, dict) and resp.get("ok") is True


@pytest.mark.asyncio
async def test_telegram_polling_message_queue(monkeypatch):
    _install_test_stubs()
    from python.helpers import settings as settings_helper

    settings_helper.TELEGRAM_MESSAGE_QUEUE.clear()

    cfg = settings_helper.get_settings()
    cfg.update(
        {
            "telegram_enabled": True,
            "telegram_mode": "polling",
            "telegram_allowed_chat_ids": "12345",
        }
    )

    monkeypatch.setattr(
        "python.helpers.dotenv.get_dotenv_value",
        lambda k, d=None: {
            "TELEGRAM_BOT_TOKEN": "token",
        }.get(k, d),
    )

    sent = {}

    async def fake_post(url, json=None, timeout=10):
        sent["payload"] = json

        class _R:
            pass

        return _R()

    monkeypatch.setattr("python.helpers.settings._send_telegram_message", fake_post)

    class _FakeContext:
        id = "tg-12345"

        class _Log:
            def log(self, **_):
                pass

        log = _Log()

        def communicate(self, _msg):
            class _Task:
                async def result(self):
                    return "pong"

            return _Task()

    import agent as agent_module

    monkeypatch.setattr(agent_module.AgentContext, "use", lambda cid: _FakeContext())

    update = {"update_id": 1, "message": {"chat": {"id": 12345}, "text": "hello"}}
    await settings_helper._handle_telegram_update(update, "token", cfg)

    assert "tg-12345" in settings_helper.TELEGRAM_MESSAGE_QUEUE


def test_telegram_not_in_allowed_list_ignored(monkeypatch):
    _install_test_stubs()
    from python.helpers import settings as settings_helper

    cfg = settings_helper.get_settings()
    cfg.update(
        {
            "telegram_enabled": True,
            "telegram_mode": "polling",
            "telegram_allowed_chat_ids": "12345",
        }
    )

    import agent as agent_module

    class _FakeContext:
        id = "tg-99999"

        class _Log:
            def log(self, **_):
                pass

        log = _Log()

    monkeypatch.setattr(
        agent_module.AgentContext,
        "use",
        lambda cid: _FakeContext() if cid == "tg-99999" else None,
    )

    update = {"update_id": 1, "message": {"chat": {"id": 99999}, "text": "hello"}}
    import asyncio

    asyncio.run(settings_helper._handle_telegram_update(update, "token", cfg))
