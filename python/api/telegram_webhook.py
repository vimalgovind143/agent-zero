from typing import Any
import asyncio
import hmac
import json
import logging
import time
import requests
import threading
from agent import AgentContext, UserMessage, AgentContextType
from python.helpers.api import ApiHandler, Request, Response
from python.helpers import settings as settings_helper, dotenv
from python.helpers.telegram import (
    _send_telegram_message,
    _process_telegram_attachments,
    _handle_telegram_command,
    _is_chat_allowed,
    _check_rate_limit,
    _sanitize_telegram_output,
    TELEGRAM_API_BASE,
    TELEGRAM_MAX_MESSAGE_LENGTH,
)
from initialize import initialize_agent

logger = logging.getLogger(__name__)

WEBHOOK_RATE_LIMIT_SECONDS = 1.0
_webhook_rate_limits: dict[str, float] = {}
_rate_limit_lock = threading.Lock()


async def _send_chat_action(token: str, chat_id: int | str, action: str = "typing") -> None:
    try:
        await asyncio.to_thread(
            requests.post,
            f"https://api.telegram.org/bot{token}/sendChatAction",
            json={"chat_id": chat_id, "action": action},
            timeout=10,
        )
    except Exception:
        return


async def _chat_action_loop(token: str, chat_id: int | str, action: str, stop: asyncio.Event) -> None:
    await _send_chat_action(token, chat_id, action)
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=4.5)
        except TimeoutError:
            await _send_chat_action(token, chat_id, action)


class TelegramWebhook(ApiHandler):
    @classmethod
    def requires_auth(cls) -> bool:
        return False

    @classmethod
    def requires_csrf(cls) -> bool:
        return False

    @classmethod
    def requires_api_key(cls) -> bool:
        return False

    async def process(
        self, input: dict[str, Any], request: Request
    ) -> dict[str, Any] | Response:
        cfg = settings_helper.get_settings()
        if not cfg.get("telegram_enabled"):
            return Response("disabled", 404)
        if cfg.get("telegram_mode") != "webhook":
            return Response("invalid_mode", 400)

        token = dotenv.get_dotenv_value(dotenv.KEY_TELEGRAM_BOT_TOKEN) or ""
        if not token:
            return Response("no_token", 400)

        expected_secret = (
            dotenv.get_dotenv_value(dotenv.KEY_TELEGRAM_WEBHOOK_SECRET) or ""
        )
        skip_rate_limit = False
        if not expected_secret:
            logger.warning(
                "Telegram webhook is enabled but TELEGRAM_WEBHOOK_SECRET is not configured. "
                "This is a security risk. Please set a secret token in your .env file."
            )
            skip_rate_limit = True
        else:
            got_secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token") or ""
            if not hmac.compare_digest(got_secret, expected_secret):
                logger.warning("Telegram webhook unauthorized: invalid or missing secret token")
                return Response("unauthorized", 401)

        try:
            update = request.get_json(force=True, silent=True) or {}
        except Exception:
            update = {}

        msg = update.get("message") or update.get("edited_message") or {}
        if not isinstance(msg, dict):
            return Response("ok", 200)

        chat = msg.get("chat") or {}
        chat_id = chat.get("id")
        if not chat_id:
            return Response("ok", 200)

        # Use shared chat validation
        if not _is_chat_allowed(cfg, chat_id):
            return Response("forbidden", 403)

        text = msg.get("text") or ""
        if not text and isinstance(msg.get("caption"), str):
            text = msg.get("caption") or ""
        if not isinstance(text, str):
            text = ""
        has_attachments = any(
            msg.get(key) for key in ["voice", "photo", "document", "audio"]
        )
        if not text and not has_attachments:
            return Response("ok", 200)

        # Handle commands using shared handler
        if text.startswith("/"):
            await _handle_telegram_command(update, token, chat_id)
            return {"ok": True}

        logger.debug(f"Processing message from chat {chat_id}")

        # Rate limiting (webhook-specific)
        if not skip_rate_limit:
            with _rate_limit_lock:
                last_time = _webhook_rate_limits.get(str(chat_id), 0)
                current_time = time.time()
                if current_time - last_time < WEBHOOK_RATE_LIMIT_SECONDS:
                    logger.debug(f"Rate limited chat {chat_id}")
                    return Response("rate_limited", 429)
                _webhook_rate_limits[str(chat_id)] = current_time

        # Process attachments (voice, photo, document, audio)
        attachments = await _process_telegram_attachments(msg, token)
        
        # Add voice transcription to text if available
        if attachments:
            voice_transcriptions = [
                att.get("transcription", "") for att in attachments 
                if att.get("type") == "voice" and att.get("transcription")
            ]
            if voice_transcriptions:
                if text:
                    text += f"\n\nVoice note transcription: {voice_transcriptions[0]}"
                else:
                    text = voice_transcriptions[0]
        
        # If still no text, create a default message
        if not text:
            if attachments:
                attachment_types = [att.get("type", "file") for att in attachments]
                text = f"Received {', '.join(attachment_types)}"
            else:
                return Response("ok", 200)  # No content to process

        ctxid = f"tg-{chat_id}"
        context = AgentContext.use(ctxid)
        if not context:
            config = initialize_agent()
            context = AgentContext(config=config, id=ctxid, type=AgentContextType.USER)
            AgentContext.use(ctxid)
        
        # Log message
        context.log.log(
            type="user",
            heading="",
            content=text,
            kvps={"source": "telegram", "chat_id": str(chat_id)},
        )

        # Create UserMessage with attachments
        user_attachments = []
        temp_files_to_cleanup = []
        
        try:
            import tempfile
            import os
            
            for att in attachments:
                if att.get("type") in ["photo", "document", "audio"]:
                    # Save attachment data to temporary file
                    suffix = ".jpg" if att.get("type") == "photo" else ".bin"
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
                        temp_file.write(att.get("data", b""))
                        temp_file_path = temp_file.name

                    temp_files_to_cleanup.append(temp_file_path)
                    user_attachments.append({
                        "file_path": temp_file_path,
                        "filename": att.get("file_name", f"{att.get('type', 'file')}{suffix}"),
                        "mime_type": att.get("mime_type", "application/octet-stream")
                    })

            task = context.communicate(UserMessage(text, user_attachments))
            typing_stop = asyncio.Event()
            typing_task = asyncio.create_task(_chat_action_loop(token, chat_id, "typing", typing_stop))
            try:
                result = await task.result()
            finally:
                typing_stop.set()
                try:
                    await asyncio.wait_for(typing_task, timeout=1)
                except Exception:
                    typing_task.cancel()
            
            # Process response
            try:
                if not isinstance(result, str):
                    result_text = json.dumps(result, ensure_ascii=False)
                else:
                    result_text = result
            except Exception:
                result_text = str(result)

            # Sanitize output before sending to Telegram
            result_text = _sanitize_telegram_output(result_text)

            # Send response using shared function
            await _send_telegram_message(token, chat_id, result_text)
            return {"ok": True}
        finally:
            # Clean up temporary files
            for file_path in temp_files_to_cleanup:
                try:
                    if file_path and os.path.exists(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temporary file {file_path}: {e}")
