"""Telegram integration module for Agent Zero.

This module handles all Telegram bot functionality including:
- Message sending and receiving
- File uploads and downloads
- Command handling
- Message queuing
- Polling and webhook modes
"""

import asyncio
import logging
import os
import re
import threading
import time
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import requests

from python.helpers import defer, dotenv
from python.helpers import runtime as runtime_helpers

if TYPE_CHECKING:
    from python.helpers.settings import Settings

# Constants
TELEGRAM_CHAT_LIFETIME_HOURS = 24
TELEGRAM_RATE_LIMIT_SECONDS = 1.0
TELEGRAM_MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024
TELEGRAM_MAX_QUEUE_SIZE = 2
TELEGRAM_CHAT_ACTION_INTERVAL_SECONDS = 4.5

# API URL templates
TELEGRAM_API_BASE = "https://api.telegram.org/bot{token}"
TELEGRAM_FILE_BASE = "https://api.telegram.org/file/bot{token}"

# Message limits
TELEGRAM_MAX_MESSAGE_LENGTH = 4096
TELEGRAM_MAX_CAPTION_LENGTH = 1024

# Global state
_telegram_poll_task: defer.DeferredTask | None = None
_telegram_poll_generation = 0
_telegram_chat_lifetimes: dict[str, datetime] = {}
_telegram_rate_limits: dict[str, float] = {}
_telegram_cleanup_lock = threading.Lock()
_telegram_rate_limit_lock = threading.Lock()

# Message queue state with proper async locks
TELEGRAM_MESSAGE_QUEUE: dict[str, list[dict]] = {}
_telegram_processing_locks: dict[str, asyncio.Lock] = {}
_telegram_locks_lock = threading.Lock()  # Protects _telegram_processing_locks

# Logger
_logger = logging.getLogger(__name__)


def _get_processing_lock(ctxid: str) -> asyncio.Lock:
    """Get or create an asyncio.Lock for a given context ID."""
    with _telegram_locks_lock:
        if ctxid not in _telegram_processing_locks:
            _telegram_processing_locks[ctxid] = asyncio.Lock()
        return _telegram_processing_locks[ctxid]


def _cleanup_processing_lock(ctxid: str) -> None:
    """Remove processing lock for a context (called during cleanup)."""
    with _telegram_locks_lock:
        _telegram_processing_locks.pop(ctxid, None)


def _reset_telegram_polling(start: bool) -> None:
    """Reset and optionally start Telegram polling."""
    global _telegram_poll_task, _telegram_poll_generation
    _telegram_poll_generation += 1
    if _telegram_poll_task:
        _telegram_poll_task.kill()
        _telegram_poll_task = None
    if start:
        generation = _telegram_poll_generation
        _telegram_poll_task = defer.DeferredTask("TelegramPolling").start_task(
            _telegram_poll_loop, generation
        )
        _logger.info("Telegram polling started")


async def _telegram_poll_loop(generation: int) -> None:
    """Main polling loop for Telegram updates."""
    from python.helpers.settings import get_settings

    offset = 0
    _logger.info("Telegram polling loop started")

    # Register commands on startup
    token = dotenv.get_dotenv_value(dotenv.KEY_TELEGRAM_BOT_TOKEN) or ""
    if token:
        await _register_telegram_commands(token)

    while True:
        if generation != _telegram_poll_generation:
            _logger.info("Telegram polling loop stopped (generation mismatch)")
            return
        cfg = get_settings()
        if not cfg.get("telegram_enabled") or cfg.get("telegram_mode") != "polling":
            _logger.info("Telegram polling stopped (disabled)")
            return
        token = dotenv.get_dotenv_value(dotenv.KEY_TELEGRAM_BOT_TOKEN) or ""
        if not token:
            _logger.debug("No Telegram bot token configured")
            await asyncio.sleep(5)
            continue
        try:
            _logger.debug(f"Polling Telegram API with offset {offset}")
            api_url = TELEGRAM_API_BASE.format(token=token)
            resp = await asyncio.to_thread(
                requests.get,
                f"{api_url}/getUpdates",
                params={"timeout": 15, "offset": offset},
                timeout=20,
            )
            data = resp.json() if resp is not None else {}
        except Exception as e:
            _logger.error(f"Telegram polling error: {e}")
            await asyncio.sleep(2)
            continue
        if not isinstance(data, dict) or not data.get("ok"):
            _logger.warning(f"Telegram API error: {data}")
            await asyncio.sleep(2)
            continue
        updates = data.get("result") or []
        if isinstance(updates, list) and updates:
            _logger.info(f"Received {len(updates)} Telegram updates")
            for update in updates:
                if not isinstance(update, dict):
                    continue
                update_id = update.get("update_id")
                if isinstance(update_id, int) and update_id >= offset:
                    offset = update_id + 1
                _logger.debug(f"Processing update {update_id}")
                asyncio.create_task(_handle_telegram_update(update, token, cfg))
        else:
            _logger.debug("No new updates, continuing to poll")


async def _send_telegram_message(
    token: str, chat_id: int | str, text: str, attachments: list | None = None
) -> bool:
    """Send message to Telegram with support for local file uploads."""
    try:
        attachments = attachments or []
        text_sent = False
        api_url = TELEGRAM_API_BASE.format(token=token)

        _logger.debug(f"Sending message to chat {chat_id} with {len(attachments)} attachments")

        if attachments:
            # Send with attachments
            for idx, attachment in enumerate(attachments):
                # Check if attachment has local file_path (from Agent Zero)
                file_path = attachment.get("file_path")
                file_id = attachment.get("file_id")

                # Determine caption: use text for first attachment only
                caption = text[:TELEGRAM_MAX_CAPTION_LENGTH] if text and idx == 0 else None

                if file_path and os.path.exists(file_path):
                    # Local file - upload via multipart
                    success = await _upload_local_file_to_telegram(
                        token, chat_id, file_path, attachment, caption
                    )
                    _logger.debug(f"Upload result: {success}")
                    text_sent = caption is not None
                elif file_id:
                    # Telegram file_id - reuse existing file on Telegram servers
                    success = await _send_by_file_id(token, chat_id, attachment, caption)
                    _logger.debug(f"Send by file_id result: {success}")
                    text_sent = caption is not None
                else:
                    _logger.warning(f"Attachment has neither file_path nor file_id")

            # Send remaining text if not already sent as caption or text is too long
            if text:
                if not text_sent or len(text) > TELEGRAM_MAX_CAPTION_LENGTH:
                    remaining_text = text if not text_sent else text[TELEGRAM_MAX_CAPTION_LENGTH:]
                    if remaining_text:
                        await asyncio.to_thread(
                            requests.post,
                            f"{api_url}/sendMessage",
                            json={"chat_id": chat_id, "text": remaining_text[:TELEGRAM_MAX_MESSAGE_LENGTH]},
                            timeout=10,
                        )
        else:
            # Send text only
            await asyncio.to_thread(
                requests.post,
                f"{api_url}/sendMessage",
                json={"chat_id": chat_id, "text": text[:TELEGRAM_MAX_MESSAGE_LENGTH]},
                timeout=10,
            )
        return True
    except Exception as e:
        _logger.error(f"Failed to send Telegram message: {e}")
        return False


async def _send_telegram_chat_action(
    token: str, chat_id: int | str, action: str = "typing"
) -> bool:
    try:
        api_url = TELEGRAM_API_BASE.format(token=token)
        await asyncio.to_thread(
            requests.post,
            f"{api_url}/sendChatAction",
            json={"chat_id": chat_id, "action": action},
            timeout=10,
        )
        return True
    except Exception:
        return False


async def _telegram_chat_action_loop(
    token: str, chat_id: int | str, action: str, stop: asyncio.Event
) -> None:
    await _send_telegram_chat_action(token, chat_id, action)
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=TELEGRAM_CHAT_ACTION_INTERVAL_SECONDS)
        except TimeoutError:
            await _send_telegram_chat_action(token, chat_id, action)


async def _upload_local_file_to_telegram(
    token: str, chat_id: int | str, file_path: str, attachment: dict, caption: str | None = None
) -> bool:
    """Upload a local file to Telegram using multipart/form-data."""
    try:
        mime_type = attachment.get("mime_type", "application/octet-stream")
        filename = attachment.get("filename", os.path.basename(file_path))
        api_url = TELEGRAM_API_BASE.format(token=token)

        # Determine send method based on mime type
        if mime_type.startswith("image/"):
            method = "sendPhoto"
            file_field = "photo"
        elif mime_type.startswith("audio/"):
            method = "sendAudio"
            file_field = "audio"
        elif mime_type.startswith("video/"):
            method = "sendVideo"
            file_field = "video"
        else:
            # Documents for everything else
            method = "sendDocument"
            file_field = "document"

        url = f"{api_url}/{method}"

        def _upload():
            with open(file_path, "rb") as f:
                files = {file_field: (filename, f, mime_type)}
                data = {"chat_id": chat_id}
                if caption:
                    data["caption"] = caption

                response = requests.post(url, files=files, data=data, timeout=60)
                return response

        response = await asyncio.to_thread(_upload)

        if response.status_code == 200:
            result = response.json()
            if result.get("ok"):
                _logger.debug(f"Successfully uploaded {filename} to Telegram")
                return True
            else:
                _logger.warning(f"Telegram API returned error: {result.get('description')}")
                return False
        else:
            _logger.warning(f"Failed to upload {filename}: HTTP {response.status_code}")
            return False

    except Exception as e:
        _logger.error(f"Failed to upload local file {file_path}: {e}")
        return False


async def _send_by_file_id(
    token: str, chat_id: int | str, attachment: dict, caption: str | None = None
) -> bool:
    """Send a file using existing Telegram file_id."""
    try:
        file_id = attachment.get("file_id")
        attachment_type = attachment.get("type", "document")
        api_url = TELEGRAM_API_BASE.format(token=token)

        # Map attachment type to Telegram method
        method_map = {
            "photo": "sendPhoto",
            "document": "sendDocument",
            "audio": "sendAudio",
            "voice": "sendVoice",
            "video": "sendVideo",
        }
        method = method_map.get(attachment_type, "sendDocument")

        # Map to correct field name
        field_map = {
            "sendPhoto": "photo",
            "sendDocument": "document",
            "sendAudio": "audio",
            "sendVoice": "voice",
            "sendVideo": "video",
        }
        field = field_map.get(method, "document")

        url = f"{api_url}/{method}"
        payload = {
            "chat_id": chat_id,
            field: file_id,
        }
        if caption:
            payload["caption"] = caption

        response = await asyncio.to_thread(
            requests.post, url, json=payload, timeout=30
        )

        if response.status_code == 200:
            return True
        else:
            _logger.warning(f"Failed to send by file_id: HTTP {response.status_code}")
            return False

    except Exception as e:
        _logger.error(f"Failed to send by file_id: {e}")
        return False


async def _download_telegram_file(token: str, file_id: str) -> bytes | None:
    """Download file from Telegram servers."""
    try:
        api_url = TELEGRAM_API_BASE.format(token=token)
        file_url = TELEGRAM_FILE_BASE.format(token=token)
        
        # Get file info
        response = await asyncio.to_thread(
            requests.get,
            f"{api_url}/getFile",
            params={"file_id": file_id},
            timeout=10,
        )
        file_info = response.json() if response else {}

        if not file_info.get("ok"):
            return None

        file_path = file_info.get("result", {}).get("file_path")
        if not file_path:
            return None
        file_size = file_info.get("result", {}).get("file_size", 0)
        if isinstance(file_size, int) and file_size > TELEGRAM_MAX_ATTACHMENT_BYTES:
            _logger.warning(
                "Skipping Telegram file %s: size %s exceeds max allowed %s bytes",
                file_id,
                file_size,
                TELEGRAM_MAX_ATTACHMENT_BYTES,
            )
            return None

        # Download file
        download_url = f"{file_url}/{file_path}"
        response = await asyncio.to_thread(
            requests.get,
            download_url,
            timeout=30,
        )

        return response.content if response else None

    except Exception as e:
        _logger.error(f"Failed to download Telegram file {file_id}: {e}")
        return None


async def _process_telegram_attachments(msg: dict, token: str) -> list:
    """Process attachments from Telegram message."""
    from python.helpers import whisper

    attachments = []

    # Handle voice notes
    if msg.get("voice"):
        _logger.debug("Processing voice note from message")
        voice = msg["voice"]
        file_id = voice.get("file_id")
        if file_id:
            file_data = await _download_telegram_file(token, file_id)
            if file_data:
                _logger.debug(f"Voice file downloaded, size: {len(file_data)} bytes")

                try:
                    import base64
                    audio_b64 = base64.b64encode(file_data).decode("utf-8")
                    transcription_result = await whisper.transcribe("base", audio_b64)

                    if isinstance(transcription_result, dict):
                        transcription = str(transcription_result.get("text", "")).strip()
                    elif isinstance(transcription_result, str):
                        transcription = transcription_result.strip()
                    else:
                        transcription = str(transcription_result).strip()

                    if transcription:
                        _logger.debug(f"Voice transcription completed: '{transcription[:100]}...'" )
                        attachments.append({
                            "type": "voice",
                            "file_id": file_id,
                            "transcription": transcription,
                            "duration": voice.get("duration", 0),
                            "mime_type": voice.get("mime_type", "audio/ogg")
                        })
                    else:
                        _logger.warning("Voice transcription returned empty text")
                except Exception as e:
                    _logger.error(f"Voice transcription failed: {e}")
            else:
                _logger.warning(f"Failed to download voice file {file_id}")

    # Handle photos
    elif msg.get("photo"):
        _logger.debug("Processing photo from message")
        photos = msg["photo"]
        if photos and isinstance(photos, list):
            # Get highest resolution photo
            photo = max(photos, key=lambda p: p.get("width", 0) * p.get("height", 0))
            file_id = photo.get("file_id")
            if file_id:
                file_data = await _download_telegram_file(token, file_id)
                if file_data:
                    attachments.append({
                        "type": "photo",
                        "file_id": file_id,
                        "width": photo.get("width", 0),
                        "height": photo.get("height", 0),
                        "file_size": photo.get("file_size", 0),
                        "data": file_data
                    })
                    _logger.debug(f"Photo processed: {photo.get('width', 0)}x{photo.get('height', 0)}")

    # Handle documents
    elif msg.get("document"):
        _logger.debug("Processing document from message")
        document = msg["document"]
        file_id = document.get("file_id")
        if file_id:
            file_data = await _download_telegram_file(token, file_id)
            if file_data:
                attachments.append({
                    "type": "document",
                    "file_id": file_id,
                    "file_name": document.get("file_name", "document"),
                    "mime_type": document.get("mime_type", "application/octet-stream"),
                    "file_size": document.get("file_size", 0),
                    "data": file_data
                })
                _logger.debug(f"Document processed: {document.get('file_name', 'unknown')}")

    # Handle audio files
    elif msg.get("audio"):
        _logger.debug("Processing audio file from message")
        audio = msg["audio"]
        file_id = audio.get("file_id")
        if file_id:
            file_data = await _download_telegram_file(token, file_id)
            if file_data:
                attachments.append({
                    "type": "audio",
                    "file_id": file_id,
                    "duration": audio.get("duration", 0),
                    "title": audio.get("title", "Audio"),
                    "mime_type": audio.get("mime_type", "audio/mpeg"),
                    "file_size": audio.get("file_size", 0),
                    "data": file_data
                })
                _logger.debug(f"Audio file processed: {audio.get('title', 'unknown')}")

    if attachments:
        _logger.debug(f"Total attachments processed: {len(attachments)}")

    return attachments


async def _process_telegram_message(
    update: dict, token: str, cfg: "Settings", chat_id: int | str
) -> None:
    """Process a Telegram message and send response."""
    from agent import AgentContext, UserMessage, AgentContextType
    from initialize import initialize_agent

    msg = update.get("message") or update.get("edited_message") or {}

    # Process attachments
    attachments = await _process_telegram_attachments(msg, token)

    # Get text content
    text = msg.get("text") or ""
    if not text and isinstance(msg.get("caption"), str):
        text = msg.get("caption") or ""

    # Add voice transcription to text if available
    if attachments:
        voice_transcriptions = [att.get("transcription", "") for att in attachments if att.get("type") == "voice" and att.get("transcription")]
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
            return  # No content to process

    ctxid = f"tg-{chat_id}"
    context = AgentContext.use(ctxid)
    if not context:
        config = initialize_agent()
        context = AgentContext(config=config, id=ctxid, type=AgentContextType.USER)
        AgentContext.use(ctxid)

    # Log message with attachment info
    log_content = text
    if attachments:
        attachment_info = []
        for att in attachments:
            if att.get("type") == "voice":
                attachment_info.append(f"Voice note ({att.get('duration', 0)}s)")
            elif att.get("type") == "photo":
                attachment_info.append(f"Photo ({att.get('width', 0)}x{att.get('height', 0)})")
            elif att.get("type") == "document":
                attachment_info.append(f"Document: {att.get('file_name', 'unknown')}")
            elif att.get("type") == "audio":
                attachment_info.append(f"Audio: {att.get('title', 'unknown')}")

        if attachment_info:
            log_content += f"\nAttachments: {', '.join(attachment_info)}"

    context.log.log(
        type="user",
        heading="",
        content=log_content,
        kvps={"source": "telegram", "chat_id": str(chat_id)},
    )

    # Create UserMessage with attachments
    user_attachments = []
    temp_files_to_cleanup = []

    try:
        for att in attachments:
            if att.get("type") in ["photo", "document", "audio"]:
                # Create attachment for Agent Zero using file path
                import tempfile

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
        typing_task = asyncio.create_task(
            _telegram_chat_action_loop(token, chat_id, "typing", typing_stop)
        )
        try:
            result = await task.result()
        finally:
            typing_stop.set()
            try:
                await asyncio.wait_for(typing_task, timeout=1)
            except Exception:
                typing_task.cancel()

        # Process response for attachments
        response_text = ""
        response_attachments = []

        _logger.debug(f"Processing agent response: type={type(result).__name__}")

        try:
            if isinstance(result, dict):
                # Check if this is a tool response with tool_args
                if "tool_args" in result and isinstance(result["tool_args"], dict):
                    tool_args = result["tool_args"]
                    response_text = tool_args.get("text", str(result))
                    response_attachments = tool_args.get("attachments", [])
                else:
                    # Direct response format
                    response_text = result.get("text", str(result))
                    response_attachments = result.get("attachments", [])
            elif isinstance(result, str):
                response_text = result
            else:
                response_text = str(result)
        except Exception as e:
            response_text = str(result)
            _logger.error(f"Error processing agent response: {e}")

        # Send response
        await _send_telegram_message(token, chat_id, response_text, response_attachments)
    finally:
        # Clean up temporary files - always run even if processing fails
        for file_path in temp_files_to_cleanup:
            try:
                if file_path and os.path.exists(file_path):
                    os.unlink(file_path)
            except Exception as e:
                _logger.warning(f"Failed to cleanup temporary file {file_path}: {e}")

        with _telegram_cleanup_lock:
            _telegram_chat_lifetimes[ctxid] = datetime.now() + timedelta(
                hours=TELEGRAM_CHAT_LIFETIME_HOURS
            )


def _is_chat_allowed(cfg: "Settings", chat_id: int | str) -> bool:
    """Check if a chat ID is in the allowed list."""
    allowed = (cfg.get("telegram_allowed_chat_ids") or "").strip()
    if not allowed:
        return True  # No allowlist configured, allow all
    allowed_set = {a.strip() for a in allowed.split(",") if a.strip()}
    return str(chat_id) in allowed_set


def _check_rate_limit(chat_id: int | str) -> bool:
    """Check and update rate limit for a chat. Returns True if allowed, False if rate limited."""
    with _telegram_rate_limit_lock:
        last_message_time = _telegram_rate_limits.get(str(chat_id), 0)
        current_time = time.time()
        if current_time - last_message_time < TELEGRAM_RATE_LIMIT_SECONDS:
            return False
        _telegram_rate_limits[str(chat_id)] = current_time
        return True


def _sanitize_telegram_output(text: str) -> str:
    """Sanitize text before sending to Telegram.
    
    - Limits message length to Telegram's 4096 character limit
    - Removes null bytes and control characters
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Remove null bytes and problematic control characters
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', text)
    
    # Telegram message limit is 4096 characters
    if len(text) > TELEGRAM_MAX_MESSAGE_LENGTH:
        text = text[:TELEGRAM_MAX_MESSAGE_LENGTH - 3] + "..."
        _logger.debug(f"Telegram message truncated to {TELEGRAM_MAX_MESSAGE_LENGTH} characters")
    
    return text


async def _handle_telegram_update(update: dict, token: str, cfg: "Settings") -> None:
    """Handle incoming Telegram update."""
    msg = update.get("message") or update.get("edited_message") or {}
    if not isinstance(msg, dict):
        return
    chat = msg.get("chat") or {}
    chat_id = chat.get("id")
    if not chat_id:
        return
    
    # Validate chat ID is allowed
    if not _is_chat_allowed(cfg, chat_id):
        _logger.debug(f"Chat {chat_id} not in allowed list")
        return

    text = msg.get("text") or ""
    if not text and isinstance(msg.get("caption"), str):
        text = msg.get("caption") or ""

    # Check if this message has attachments (voice, photo, document, audio)
    has_attachments = any(msg.get(key) for key in ["voice", "photo", "document", "audio"])

    # Allow processing if there's text OR attachments
    if not (isinstance(text, str) and text) and not has_attachments:
        _logger.debug("Message has no text or attachments, ignoring")
        return

    if not _check_rate_limit(chat_id):
        _logger.debug(f"Rate limited chat {chat_id}")
        return

    _telegram_cleanup_expired_chats()

    if text.startswith("/"):
        await _handle_telegram_command(update, token, chat_id)
        return

    ctxid = f"tg-{chat_id}"

    # Check queue size
    current_queue = TELEGRAM_MESSAGE_QUEUE.get(ctxid, [])
    if len(current_queue) >= TELEGRAM_MAX_QUEUE_SIZE:
        # Queue is full (2 messages already), reject with error
        await _send_telegram_message(
            token, chat_id,
            "⚠️ Queue full!\n\n"
            "You have 2 messages waiting.\n"
            "Please wait for me to finish before sending more.\n\n"
            "Use /cancel to stop current tasks."
        )
        _logger.warning(f"Queue full for chat {chat_id}, rejected message")
        return

    # Add message to queue
    message_data = {
        "update": update,
        "token": token,
        "cfg": cfg,
        "chat_id": chat_id,
        "ctxid": ctxid,
    }

    if ctxid not in TELEGRAM_MESSAGE_QUEUE:
        TELEGRAM_MESSAGE_QUEUE[ctxid] = []
    TELEGRAM_MESSAGE_QUEUE[ctxid].append(message_data)

    queue_position = len(TELEGRAM_MESSAGE_QUEUE[ctxid])
    _logger.debug(f"Message added to queue for chat {chat_id}, position {queue_position}/{TELEGRAM_MAX_QUEUE_SIZE}")

    # Notify user if queued
    if queue_position > 1:
        await _send_telegram_message(
            token, chat_id,
            f"📥 Queued ({queue_position}/{TELEGRAM_MAX_QUEUE_SIZE})\n\n"
            f"Processing your previous message first..."
        )

    # Start processing with proper async lock
    asyncio.create_task(_process_queued_messages(ctxid))


async def _process_queued_messages(ctxid: str) -> None:
    """Process messages from the queue sequentially with proper async locking."""
    lock = _get_processing_lock(ctxid)
    
    # Try to acquire the lock, if already processing, return
    if lock.locked():
        return
    
    async with lock:
        while ctxid in TELEGRAM_MESSAGE_QUEUE and TELEGRAM_MESSAGE_QUEUE[ctxid]:
            # Get the next message from the queue
            message_data = TELEGRAM_MESSAGE_QUEUE[ctxid][0]

            update = message_data["update"]
            token = message_data["token"]
            cfg = message_data["cfg"]
            chat_id = message_data["chat_id"]

            try:
                # Process the message
                await _process_telegram_message(update, token, cfg, chat_id)
            except Exception as e:
                _logger.error(f"Error processing queued message for chat {chat_id}: {e}")
                try:
                    await _send_telegram_message(
                        token, chat_id,
                        "❌ Error processing your message. Please try again."
                    )
                except Exception:
                    pass
            finally:
                # Remove the processed message from queue
                if ctxid in TELEGRAM_MESSAGE_QUEUE and TELEGRAM_MESSAGE_QUEUE[ctxid]:
                    TELEGRAM_MESSAGE_QUEUE[ctxid].pop(0)

                # Clean up empty queue
                if ctxid in TELEGRAM_MESSAGE_QUEUE and not TELEGRAM_MESSAGE_QUEUE[ctxid]:
                    del TELEGRAM_MESSAGE_QUEUE[ctxid]
                    _cleanup_processing_lock(ctxid)


async def _handle_telegram_command(
    update: dict, token: str, chat_id: int | str
) -> None:
    """Handle Telegram bot commands."""
    from agent import AgentContext

    msg = update.get("message") or {}
    text = msg.get("text") or ""
    if not isinstance(text, str):
        return

    command = text.split()[0].lower()

    if command == "/start":
        welcome_text = (
            "Welcome to Agent Zero! 🤖\n\n"
            "I'm an AI assistant that can help you with various tasks.\n\n"
            "Just send me a message and I'll respond.\n\n"
            "Available commands:\n"
            "/start - Show this welcome message\n"
            "/help - Show help information\n"
            "/newchat - Start a fresh conversation\n"
            "/reset - Reset conversation & clear memory\n"
            "/clear - Clear chat display\n"
            "/status - Show bot and session status\n"
            "/cancel - Stop current task"
        )
        await _send_telegram_message(token, chat_id, welcome_text)

    elif command == "/help":
        help_text = (
            "🤖 Agent Zero Commands\n\n"
            "Chat Management:\n"
            "• /start - Welcome message\n"
            "• /newchat - Start fresh conversation\n"
            "• /reset - Reset conversation & clear memory\n"
            "• /clear - Clear chat display\n\n"
            "Information:\n"
            "• /help - This help message\n"
            "• /status - Show bot status, model, and token usage\n\n"
            "Control:\n"
            "• /cancel - Stop current running task\n\n"
            "I can help with:\n"
            "• Writing and debugging code\n"
            "• Generating images and files\n"
            "• Answering questions\n"
            "• File operations\n"
            "• And much more!"
        )
        await _send_telegram_message(token, chat_id, help_text)

    elif command == "/newchat":
        ctxid = f"tg-{chat_id}"
        context = AgentContext.use(ctxid)
        if context:
            context.reset()
            AgentContext.remove(ctxid)
        with _telegram_cleanup_lock:
            _telegram_chat_lifetimes.pop(ctxid, None)
        await _send_telegram_message(
            token, chat_id,
            "✨ New conversation started!\n\n"
            "I've reset our chat. Previous context is cleared.\n"
            "How can I help you today?"
        )
        _logger.debug(f"Started new chat for {chat_id}")

    elif command == "/reset":
        ctxid = f"tg-{chat_id}"
        context = AgentContext.use(ctxid)
        if context:
            context.reset()
            AgentContext.remove(ctxid)
        with _telegram_cleanup_lock:
            _telegram_chat_lifetimes.pop(ctxid, None)
        await _send_telegram_message(
            token, chat_id,
            "🔄 Conversation reset!\n\n"
            "Memory cleared and context reset.\n"
            "How can I help you?"
        )
        _logger.debug(f"Reset conversation for chat {chat_id}")

    elif command == "/clear":
        # Visual chat clearing with newlines + optional delete message
        clear_message = (
            "🧹 Chat display cleared!\n\n"
            "(Previous messages remain above in Telegram history)"
        )
        await _send_telegram_message(token, chat_id, clear_message)

    elif command == "/status":
        from python.helpers.settings import get_settings
        ctxid = f"tg-{chat_id}"
        context = AgentContext.use(ctxid)
        cfg = get_settings()

        # Build status info
        status_lines = ["📊 Agent Zero Status", ""]

        # Bot status
        status_lines.append("🟢 Status: Online")
        status_lines.append(f"🤖 Model: {cfg.get('model_name', 'Unknown')}")
        status_lines.append(f"🏢 Provider: {cfg.get('model_provider', 'Unknown')}")
        status_lines.append("")

        # Session info
        if context:
            status_lines.append("📁 Session Info:")
            status_lines.append(f"• Chat ID: {chat_id}")
            status_lines.append(f"• Context ID: {ctxid}")

            # Try to get token usage if available
            try:
                agent = context.agent0
                if agent and hasattr(agent, 'history') and agent.history:
                    total_tokens = 0
                    # history is a list, iterate safely
                    for msg in agent.history:
                        if hasattr(msg, 'usage') and msg.usage:
                            total_tokens += msg.usage
                    if total_tokens > 0:
                        status_lines.append(f"• Tokens this session: {total_tokens:,}")

                # Message count from context log
                if hasattr(context, 'log') and context.log:
                    log = context.log
                    # Try to get messages count safely
                    msg_count = 0
                    if hasattr(log, 'messages') and isinstance(log.messages, (list, tuple)):
                        msg_count = len(log.messages)
                    elif hasattr(log, 'get_messages') and callable(log.get_messages):
                        try:
                            msgs = log.get_messages()
                            msg_count = len(msgs) if msgs else 0
                        except Exception:
                            pass
                    if msg_count > 0:
                        status_lines.append(f"• Messages: {msg_count}")
            except Exception:
                pass

            status_lines.append("")
            status_lines.append("💡 Tip: Use /reset to clear memory and start fresh")
        else:
            status_lines.append("📁 Session: No active conversation")
            status_lines.append("")
            status_lines.append("💡 Send a message to start chatting!")

        status_text = "\n".join(status_lines)
        await _send_telegram_message(token, chat_id, status_text)

    elif command == "/cancel":
        ctxid = f"tg-{chat_id}"
        queue_cleared = False
        processing_stopped = False

        # Clear the message queue
        if ctxid in TELEGRAM_MESSAGE_QUEUE and TELEGRAM_MESSAGE_QUEUE[ctxid]:
            queue_size = len(TELEGRAM_MESSAGE_QUEUE[ctxid])
            TELEGRAM_MESSAGE_QUEUE[ctxid] = []
            del TELEGRAM_MESSAGE_QUEUE[ctxid]
            queue_cleared = True
            _logger.debug(f"Cleared {queue_size} messages from queue for chat {chat_id}")

        # Check if currently processing (using async lock)
        lock = _get_processing_lock(ctxid)
        if lock.locked():
            processing_stopped = True
            _logger.debug(f"Processing in progress for chat {chat_id}")

        if queue_cleared or processing_stopped:
            await _send_telegram_message(
                token, chat_id,
                "⏹️ Cancelled!\n\n"
                f"{'• Cleared message queue' if queue_cleared else ''}\n"
                f"{'• Stopped current task' if processing_stopped else ''}\n\n"
                "Ready for new messages."
            )
        else:
            await _send_telegram_message(
                token, chat_id,
                "✅ Nothing to cancel.\n\n"
                "I'm ready for your next message!"
            )

    else:
        await _send_telegram_message(
            token,
            chat_id,
            f"❓ Unknown command: {command}\n\n"
            f"Send /help for available commands.",
        )


async def _register_telegram_commands(token: str) -> bool:
    """Register bot commands with Telegram for the command menu."""
    try:
        api_url = TELEGRAM_API_BASE.format(token=token)
        commands = [
            {"command": "start", "description": "Show welcome message"},
            {"command": "help", "description": "Show available commands"},
            {"command": "newchat", "description": "Start a fresh conversation"},
            {"command": "reset", "description": "Reset conversation & clear memory"},
            {"command": "clear", "description": "Clear chat display"},
            {"command": "status", "description": "Show bot and session status"},
            {"command": "cancel", "description": "Stop current running task"},
        ]

        response = await asyncio.to_thread(
            requests.post,
            f"{api_url}/setMyCommands",
            json={"commands": commands},
            timeout=10,
        )

        if response.status_code == 200:
            result = response.json()
            if result.get("ok"):
                _logger.info("Telegram commands registered successfully")
                return True
            else:
                _logger.warning(f"Failed to register commands: {result.get('description')}")
                return False
        else:
            _logger.warning(f"Failed to register commands: HTTP {response.status_code}")
            return False

    except Exception as e:
        _logger.error(f"Error registering Telegram commands: {e}")
        return False


def _telegram_cleanup_expired_chats() -> None:
    """Clean up expired Telegram chat sessions and stale rate limits."""
    from agent import AgentContext

    now = datetime.now()
    
    # Clean up expired chat sessions
    with _telegram_cleanup_lock:
        expired = [
            cid for cid, expiry in _telegram_chat_lifetimes.items() if now > expiry
        ]
        for ctxid in expired:
            try:
                context = AgentContext.get(ctxid)
                if context:
                    context.reset()
                    AgentContext.remove(ctxid)
                del _telegram_chat_lifetimes[ctxid]
                _cleanup_processing_lock(ctxid)
                _logger.debug(f"Cleaned up expired Telegram chat: {ctxid}")
            except Exception as e:
                _logger.error(f"Failed to cleanup chat {ctxid}: {e}")

    # Clean up stale rate limit entries (older than 1 hour)
    with _telegram_rate_limit_lock:
        stale_time = time.time() - 3600  # 1 hour ago
        stale_chats = [
            chat_id for chat_id, last_time in _telegram_rate_limits.items()
            if last_time < stale_time
        ]
        for chat_id in stale_chats:
            del _telegram_rate_limits[chat_id]


def start_telegram_polling_if_enabled() -> None:
    """Start Telegram polling if enabled in settings."""
    from python.helpers.settings import get_settings

    cfg = get_settings()
    _reset_telegram_polling(
        bool(cfg.get("telegram_enabled")) and cfg.get("telegram_mode") == "polling"
    )


# Export public interface
__all__ = [
    "start_telegram_polling_if_enabled",
    "_send_telegram_message",
    "_handle_telegram_update",
    "_process_telegram_attachments",
    "_handle_telegram_command",
    "TELEGRAM_MAX_ATTACHMENT_BYTES",
    "TELEGRAM_MESSAGE_QUEUE",
    "TELEGRAM_MAX_QUEUE_SIZE",
    "TELEGRAM_API_BASE",
    "TELEGRAM_FILE_BASE",
    "TELEGRAM_MAX_MESSAGE_LENGTH",
    "TELEGRAM_MAX_CAPTION_LENGTH",
    "_telegram_cleanup_expired_chats",
    "_reset_telegram_polling",
    "_is_chat_allowed",
    "_check_rate_limit",
    "_sanitize_telegram_output",
]
