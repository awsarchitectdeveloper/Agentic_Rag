import asyncio
import inspect
import json
import websockets
from openai import AsyncAzureOpenAI
from datetime import datetime
from collections import defaultdict
import base64
from io import BytesIO
from chainlit import logger
from azure.identity import DefaultAzureCredential, get_bearer_token_provider


class EventHandler:
    def __init__(self):
        self.event_handlers = defaultdict(list)

    def on(self, event_name, handler):
        self.event_handlers[event_name].append(handler)

    def clear_event_handlers(self):
        self.event_handlers = defaultdict(list)

    def dispatch(self, event_name, event):
        for handler in self.event_handlers[event_name]:
            if inspect.iscoroutinefunction(handler):
                asyncio.create_task(handler(event))
            else:
                handler(event)

    async def wait_for_next(self, event_name):
        future = asyncio.Future()

        def handler(event):
            if not future.done():
                future.set_result(event)

        self.on(event_name, handler)
        return await future


class TTS_model(EventHandler):
    def __init__(self, settings):
        super().__init__()
        self.endpoint = settings.tts_endpoint
        self.credentials = DefaultAzureCredential()
        self.acquire_token = get_bearer_token_provider(self.credentials, settings.bearer_token_scope)
        self.api_version = settings.tts_api_version
        self.azure_deployment = settings.tts_deployment_name
        self.session = None

    def is_connected(self):
        return self.session is not None

    async def connect(self):
        self.session = AsyncAzureOpenAI(
            azure_endpoint=self.endpoint,
            azure_deployment=self.azure_deployment,
            api_version=self.api_version,
            azure_ad_token_provider=self.acquire_token,
        )
        return self.session

    async def create_audio(self, response):
        buffer = BytesIO()
        async with self.session.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="shimmer",
            input=response,
            response_format="pcm",
            instructions="Voice: High Energy, Enthusiastic but clear. You are happy to give insightful information in either Dutch or English.",
        ) as response:
            async for chunk in response.iter_bytes():
                buffer.write(chunk)

        return buffer.getvalue()


class RealtimeSTT(EventHandler):
    def __init__(self, settings):
        super().__init__()
        self.url = settings.stt_endpoint
        self.credentials = DefaultAzureCredential()
        self.acquire_token = get_bearer_token_provider(self.credentials, settings.bearer_token_scope)
        self.api_version = settings.stt_api_version
        self.azure_deployment = settings.stt_deployment_name
        self.ws = None

    def _log_event(self, event):
        realtime_event = {
            "time": datetime.utcnow().isoformat(),
            "source": "client" if event["type"].startswith("client.") else "server",
            "event": event,
        }
        self.dispatch("realtime.event", realtime_event)

    def is_connected(self):
        return self.ws is not None

    def log(self, *args):
        logger.debug(f"[Websocket/{datetime.utcnow().isoformat()}]", *args)

    async def connect(self):
        if self.is_connected():
            raise Exception("Already connected")
        self.ws = await websockets.connect(
            f"{self.url}openai/realtime?api-version={self.api_version}&deployment={self.azure_deployment}",
            additional_headers={"Authorization": f"Bearer {self.acquire_token()}"},
        )
        self.log(f"Connected to {self.url}")
        await self.send(
            "session.update",
            {
                "session": {
                    "input_audio_format": "pcm16",
                    "input_audio_transcription": {
                        "model": "tts",
                        "prompt": "The input will be questions in either Dutch or English.",
                    },
                    "turn_detection": {
                        "type": "server_vad",
                        # These values can be tweaked
                        "threshold": 0.75,  # adjust based on expected amount of background noise
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 300,  # ms of silence before the model decides your turn is over
                    },
                    "input_audio_noise_reduction": {"type": "near_field"},
                }
            },
        )
        asyncio.create_task(self._receive_messages())

    async def _receive_messages(self):
        async for message in self.ws:
            event = json.loads(message)
            if event["type"] == "error":
                logger.error("ERROR", message)
            self.log("received:", event)
            self.dispatch(f"server.{event['type']}", event)
            self.dispatch("server.*", event)

    async def send(self, event_name, data=None):
        if not self.is_connected():
            raise Exception("RealtimeAPI is not connected")
        data = data or {}
        if not isinstance(data, dict):
            raise Exception("data must be a dictionary")
        event = {"event_id": self._generate_id("evt_"), "type": event_name, **data}
        self.dispatch(f"client.{event_name}", event)
        self.dispatch("client.*", event)
        self.log("sent:", event)
        await self.ws.send(json.dumps(event))

    def _generate_id(self, prefix):
        return f"{prefix}{int(datetime.utcnow().timestamp() * 1000)}"

    async def disconnect(self):
        if self.ws:
            await self.ws.close()
            self.ws = None
            self.log(f"Disconnected from {self.url}")

    async def append_audio_chunk(self, audio_bytes: bytes):
        encoded = base64.b64encode(audio_bytes).decode("utf-8")
        await self.send("input_audio_buffer.append", {"audio": encoded})
        return True
