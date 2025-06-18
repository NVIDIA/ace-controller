# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""FastAPI WebSocket transport implementation for ACE Controller.

This module provides WebSocket transport functionality for the ACE Controller,
supporting both input (RTSP streams, WebSocket messages) and output (audio, frames)
capabilities. It includes classes for handling WebSocket connections, RTSP streams,
and audio processing with proper synchronization and error handling.
"""

import asyncio
import io
import time
import typing
import wave
from enum import Enum

import av
import av.logging
from av.audio.resampler import AudioResampler
from loguru import logger
from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    FatalErrorFrame,
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
    StartInterruptionFrame,
    TransportMessageFrame,
    TransportMessageUrgentFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.serializers.base_serializer import FrameSerializer, FrameSerializerType
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketCallbacks,
)

try:
    from fastapi import WebSocket
    from starlette.websockets import WebSocketState
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use FastAPI websockets, you need to `pip install pipecat-ai[websocket]`.")
    raise Exception(f"Missing module: {e}") from None


class AudioEncoding(Enum):
    """Audio encoding formats.

    Attributes:
        PCM: Raw PCM audio format
        WAV: WAV container format
    """

    PCM = "pcm"
    WAV = "wav"


class ACETransportParams(TransportParams):
    """Parameters for the ACETransport.

    Attributes:
        rtsp_url: RTSP input URL.
        rtsp_transport: Transport protocol (tcp/udp).
        rtsp_max_delay: Buffer delay in microseconds.
        audio_in_enabled: Enable audio input.
        camera_in_enabled: Enable camera input.
        audio_out_enabled: Enable audio output.
        audio_out_sample_rate: Output sample rate in Hz.
        audio_in_encoding: Input audio encoding format.
        add_wav_header: Add WAV header to output.
        audio_out_bitrate: Output bitrate in bits/second.
        serializer: Frame serializer instance.
        session_timeout: WebSocket timeout in seconds.
        burst_mode_duration: Initial duration for which the output audio data will be sent in burst mode in seconds.
    """

    rtsp_url: str = ""  # RTSP Input URL
    rtsp_transport: str = "tcp"  # Use TCP instead of UDP for better reliability.
    rtsp_max_delay: str = "500000"  # Increase buffer delay (in microseconds).
    audio_in_enabled: bool = True  # Enable/Disable audio input
    camera_in_enabled: bool = False  # Enable/Disable camera input
    audio_out_enabled: bool = True  #  Enable/Disable audio output
    audio_out_sample_rate: int = 16000  # Sample rate of the audio output
    audio_in_encoding: AudioEncoding = AudioEncoding.PCM  # Encoding of the audio input
    add_wav_header: bool = True  # Add a WAV header to the audio output
    audio_out_bitrate: int = 32000  # Bitrate of the audio output
    serializer: FrameSerializer = ProtobufFrameSerializer()  # Serializer for the audio output
    session_timeout: int | None = None  # Timeout for the websocket connection
    burst_mode_duration: float = 0.4  # Initial duration for which the audio data will be sent in burst mode


class FastAPIWebsocketClient:
    """FastAPI WebSocket client implementation.

    This class provides a client implementation for FastAPI WebSocket connections,
    handling setup, message sending, and disconnection.
    """

    def __init__(self, websocket: WebSocket, is_binary: bool, callbacks: FastAPIWebsocketCallbacks):
        """Initialize the FastAPIWebsocketClient.

        Args:
            websocket: The FastAPI WebSocket connection.
            is_binary: Whether the Serializer is binary.
            callbacks: The callbacks for the WebSocket client.
        """
        self._websocket = websocket
        self._closing = False
        self._is_binary = is_binary
        self._callbacks = callbacks
        self._leave_counter = 0

    async def setup(self, _: StartFrame):
        """Setup the WebSocket client.

        Args:
            _: The StartFrame.
        """
        self._leave_counter += 1

    def receive(self) -> typing.AsyncIterator[bytes | str]:
        """Receive messages from the WebSocket connection.

        Returns:
            typing.AsyncIterator[bytes | str]: An asynchronous iterator over the received messages.
        """
        return self._websocket.iter_bytes() if self._is_binary else self._websocket.iter_text()

    async def send(self, data: str | bytes):
        """Send a message through the WebSocket connection.

        Args:
            data: The message to send.
        """
        if self._can_send():
            if self._is_binary:
                await self._websocket.send_bytes(data)
            else:
                await self._websocket.send_text(data)

    async def disconnect(self):
        """Disconnect from the WebSocket connection.

        This method will disconnect from the WebSocket connection and trigger the client disconnected event.
        """
        self._leave_counter -= 1
        if self._leave_counter > 0:
            return

        if self.is_connected and not self.is_closing:
            self._closing = True
            await self._websocket.close()
            await self.trigger_client_disconnected()

    async def trigger_client_disconnected(self):
        """Trigger the client disconnected event.

        This method will trigger the client disconnected event.
        """
        await self._callbacks.on_client_disconnected(self._websocket)

    async def trigger_client_connected(self):
        """Trigger the client connected event.

        This method will trigger the client connected event.
        """
        await self._callbacks.on_client_connected(self._websocket)

    async def trigger_client_timout(self):
        """Trigger the client timeout event.

        This method will trigger the client timeout event.
        """
        await self._callbacks.on_session_timeout(self._websocket)

    def _can_send(self):
        """Check if the client can send messages.

        Returns:
            bool: True if the client can send messages, False otherwise.
        """
        return self.is_connected and not self.is_closing

    @property
    def is_connected(self) -> bool:
        """Check if the client is connected.

        Returns:
            bool: True if the client is connected, False otherwise.
        """
        return self._websocket.client_state == WebSocketState.CONNECTED

    @property
    def is_closing(self) -> bool:
        """Check if the client is closing.

        Returns:
            bool: True if the client is closing, False otherwise.
        """
        return self._closing


class ACEInputTransport(BaseInputTransport):
    """Input transport handling RTSP and WebSocket connections.

    Manages input streams from:
        - RTSP audio/video feeds
        - WebSocket client messages
        - Audio frame processing
        - Session monitoring
    """

    def __init__(
        self,
        transport: BaseTransport,
        client: FastAPIWebsocketClient,
        params: ACETransportParams,
        **kwargs,
    ):
        """Initializes the input transport.

        Args:
            params: Transport configuration parameters.
            transport: Transport instance.
            client: WebSocket client instance.
            **kwargs: Additional arguments for BaseInputTransport.
        """
        super().__init__(params, **kwargs)
        self._transport = transport
        self._client = client
        self._params = params
        self._receive_task = None
        self._monitor_websocket_task = None
        self._start_frame = None
        self._read_rtsp_task = None
        self._process_rtsp_task = None

    async def start(self, frame: StartFrame) -> None:
        """Starts input transport processing.

        Args:
            frame: Start frame triggering initialization.
        """
        await super().start(frame)
        await self._params.serializer.setup(frame)
        self.vad_chunk_size = int(self._sample_rate * self._params.audio_in_channels * 2 * 0.032)  # 32 ms
        self._start_frame = frame
        if self._client:
            await self._start_websocket()
        if self._params.rtsp_url != "":
            self._resampler = AudioResampler(
                layout="stereo" if self._params.audio_in_channels == 2 else "mono",
                rate=self._sample_rate,
                format="s16",
            )
            await self._start_rtsp()
        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame):
        """Stop the input transport and cleanup resources.

        Args:
            frame: The EndFrame that triggered the stop.
        """
        await super().stop(frame)
        await self._stop_websocket_tasks()
        await self._stop_rtsp_tasks()

    async def cancel(self, frame: CancelFrame):
        """Cancel the input transport and cleanup resources.

        Args:
            frame: The CancelFrame that triggered the cancellation.
        """
        await super().cancel(frame)
        await self._stop_websocket_tasks()
        await self._stop_rtsp_tasks()

    async def cleanup(self):
        """Cleanup the input transport and resources.

        This method will cleanup the input transport and resources.
        """
        await super().cleanup()
        await self._transport.cleanup()

    ## Functions for WebSocket

    async def set_websocket(self, client: FastAPIWebsocketClient):
        """Set/Update the WebSocket connection.

        This method will start background tasks to read from the new WebSocket connection.
        """
        await self._stop_websocket_tasks()
        self._client = client
        await self._start_websocket()

    async def _start_websocket(self):
        """Start the WebSocket connection.

        This method will start background tasks to read from the WebSocket connection.
        """
        await self._client.setup(self._start_frame)
        if not self._monitor_websocket_task and self._params.session_timeout:
            self._monitor_websocket_task = self.create_task(self._monitor_websocket())
        await self._client.trigger_client_connected()
        if not self._receive_task:
            self._receive_task = self.create_task(self._receive_websocket_messages())

    async def _stop_websocket_tasks(self):
        if self._monitor_websocket_task:
            await self.cancel_task(self._monitor_websocket_task)
            self._monitor_websocket_task = None
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        if self._client:
            await self._client.disconnect()

    async def _receive_websocket_messages(self):
        """Receive messages from the WebSocket connection.

        This method will receive messages from the WebSocket connection and push the frames to the pipeline.
        """
        try:
            async for message in self._client.receive():
                frame = await self._params.serializer.deserialize(message)

                if not frame:
                    continue

                if isinstance(frame, InputAudioRawFrame):
                    if self._params.audio_in_encoding == AudioEncoding.WAV:
                        with io.BytesIO(frame.audio) as buffer, wave.open(buffer, "rb") as wf:
                            frame.audio = wf.readframes(wf.getnframes())
                            frame.sample_rate = wf.getframerate()
                            frame.num_channels = wf.getnchannels()

                    # Sileo VAD expects 32 ms input audio chunk
                    # TODO: This is a temporary fix, we need to update the VAD to handle variable length audio chunks
                    if self._params.vad_enabled and len(frame.audio) > self.vad_chunk_size:
                        logger.error("VAD is enabled, but audio chunk larger than 32 ms received from WebSocket")
                        await self.push_frame(
                            FatalErrorFrame(
                                error="VAD is enabled, but audio chunk larger than 32 ms received from WebSocket"
                            )
                        )
                    await self.push_audio_frame(frame)
                else:
                    await self.push_frame(frame)
        except Exception as e:
            logger.error(f"{self} exception receiving data: {e.__class__.__name__} ({e})")

        await self._client.trigger_client_disconnected()

    async def _monitor_websocket(self):
        """Wait for self._params.session_timeout seconds, if the websocket is still open, trigger timeout event."""
        await asyncio.sleep(self._params.session_timeout)
        await self._client.trigger_client_timout()

    ## Functions for RTSP

    async def _start_rtsp(self):
        """Start the RTSP connection.

        This method will start background tasks to read from the RTSP stream and push the audio frames to the pipeline.
        """
        self._read_rtsp_task = self.create_task(self._thread_task_handler())
        self._rtsp_queue = asyncio.Queue()
        self._process_rtsp_task = self.create_task(self._process_rtsp_task_handler())

    async def _stop_rtsp_tasks(self):
        if self._read_rtsp_task:
            await self.cancel_task(self._read_rtsp_task)
            self._read_rtsp_task = None
        if self._process_rtsp_task:
            await self.cancel_task(self._process_rtsp_task)
            self._process_rtsp_task = None

    async def _thread_task_handler(self):
        """This method will start background task to read from the RTSP stream and push to rtsp queue."""
        try:
            self._thread_running = True
            await asyncio.to_thread(self._receive_rtsp_messages)
        except asyncio.CancelledError:
            self._thread_running = False
            raise

    async def _process_rtsp_task_handler(self):
        """This method will read from the rtsp queue and push frames to the pipeline."""
        while True:
            frame = await self._rtsp_queue.get()
            if isinstance(frame, InputAudioRawFrame):
                await self.push_audio_frame(frame)
            else:
                await self.push_frame(frame)

    def _receive_rtsp_messages(self):
        """This method will read from the RTSP stream and push to rtsp queue."""
        av.logging.set_level(av.logging.DEBUG)

        options = {
            "rtsp_transport": self._params.rtsp_transport,
            "max_delay": self._params.rtsp_max_delay,
        }

        logger.info(f"Connecting to RTSP stream: {self._params.rtsp_url}")
        self._container = av.open(self._params.rtsp_url, options=options)

        streams = []
        for stream in self._container.streams:
            if stream.type == "video":
                streams.append(stream)
                logger.debug(f"Checking RTSP camera stream: {self._params.rtsp_url}")
            elif stream.type == "audio":
                streams.append(stream)
                logger.debug(f"Checking RTSP audio stream: {self._params.rtsp_url}")

        logger.debug("Reading from RTSP stream")
        buffer = b""
        for packet in self._container.demux(streams):
            try:
                if packet.stream.type == "video":
                    # Decode video frames
                    pass
                    # TODO send video frames downstream
                elif packet.stream.type == "audio" and packet.size != 0:
                    for frame in packet.decode():
                        resampled_frame = self._resampler.resample(frame)
                        for el in resampled_frame:
                            buffer += el.to_ndarray().tobytes()
                            while len(buffer) >= self.vad_chunk_size:
                                pipecat_frame = InputAudioRawFrame(
                                    audio=buffer[: self.vad_chunk_size],
                                    sample_rate=self._sample_rate,
                                    num_channels=self._params.audio_in_channels,
                                )
                                asyncio.run_coroutine_threadsafe(
                                    self._rtsp_queue.put(pipecat_frame), self.get_event_loop()
                                )
                                buffer = buffer[self.vad_chunk_size :]
            except asyncio.CancelledError:
                logger.debug("RTSP stream read cancelled")
            except Exception as e:
                logger.error(f"Exception in RTSP stream read: {e}")
                asyncio.run_coroutine_threadsafe(
                    self._rtsp_queue.put(FatalErrorFrame(error="Error in reading RTSP Stream")), self.get_event_loop()
                )
        if len(buffer) > 0:
            pipecat_frame = InputAudioRawFrame(
                audio=buffer,
                sample_rate=self._sample_rate,
                num_channels=self._params.audio_in_channels,
            )
            asyncio.run_coroutine_threadsafe(self._rtsp_queue.put(pipecat_frame), self.get_event_loop())

        logger.info(f"RTSP stream {self._params.rtsp_url} closed")


class ACEOutputTransport(BaseOutputTransport):
    """Output transport for WebSocket connections.

    Handles:
        - Audio frame transmission
        - Message frame delivery
        - Playback timing simulation
        - Connection state management
    """

    def __init__(
        self,
        transport: BaseTransport,
        client: FastAPIWebsocketClient,
        params: ACETransportParams,
        **kwargs,
    ):
        """Initialize the ACEOutputTransport.

        Args:
            params: Transport configuration parameters.
            transport: Transport instance.
            client: WebSocket client instance.
            **kwargs: Additional arguments passed to BaseOutputTransport.
        """
        super().__init__(params, **kwargs)

        self._transport = transport
        self._client = client
        self._params = params

        # write_raw_audio_frames() is called quickly, as soon as we get audio
        # (e.g. from the TTS), and since this is just a network connection we
        # would be sending it to quickly. Instead, we want to block to emulate
        # an audio device, this is what the send interval is. It will be
        # computed on StartFrame.
        self._send_interval = 0
        self._next_send_time = 0
        self._total_sent_interval = 0

    async def start(self, frame: StartFrame):
        """Start the output transport.

        Args:
            frame: The StartFrame that triggered the start.
        """
        await super().start(frame)
        self._start_frame = frame
        if self._client:
            await self._client.setup(frame)
        await self._params.serializer.setup(frame)
        self._send_interval = (self.audio_chunk_size / self.sample_rate) / 2
        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame):
        """Stop the output transport and cleanup resources.

        Args:
            frame: The EndFrame that triggered the stop.
        """
        await super().stop(frame)
        if self._client:
            await self._client.disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the output transport and cleanup resources.

        Args:
            frame: The CancelFrame that triggered the cancellation.
        """
        await super().cancel(frame)
        if self._client:
            await self._client.disconnect()

    async def cleanup(self):
        """Cleanup the output transport and resources.

        This method will cleanup the output transport and resources.
        """
        await super().cleanup()
        await self._transport.cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process the frame.

        This method will process the frame and write to the WebSocket connection.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, StartInterruptionFrame):
            self._next_send_time = 0
            self._total_sent_interval = 0
        if not isinstance(frame, AudioRawFrame | TransportMessageFrame | TransportMessageUrgentFrame):
            await self._write_frame(frame)

    async def send_message(self, frame: TransportMessageFrame | TransportMessageUrgentFrame):
        """Send a transport message frame through the WebSocket.

        Args:
            frame: The message frame to send, either regular or urgent.
        """
        await self._write_frame(frame)

    async def set_websocket(self, client: FastAPIWebsocketClient):
        """Set/Update the WebSocket connection."""
        if self._client:
            await self._client.disconnect()
        self._client = client
        await self._client.setup(self._start_frame)

    async def write_raw_audio_frames(self, frames: bytes, destination: str | None = None):
        """Writes audio frames to WebSocket with timing control.

        Args:
            frames: Raw audio data to transmit.
            destination: The destination to write the audio frames to.

        Note:
            Simulates audio device timing to prevent flooding
            the network connection with audio data.
        """
        if not self._client or self._client.is_closing:
            return

        if not self._client.is_connected:
            # Simulate audio playback with a sleep.
            await self._write_audio_sleep()
            return

        frame = OutputAudioRawFrame(
            audio=frames,
            sample_rate=self.sample_rate,
            num_channels=self._params.audio_out_channels,
        )

        if self._params.add_wav_header:
            with io.BytesIO() as buffer:
                with wave.open(buffer, "wb") as wf:
                    wf.setsampwidth(2)
                    wf.setnchannels(frame.num_channels)
                    wf.setframerate(frame.sample_rate)
                    wf.writeframes(frame.audio)
                wav_frame = OutputAudioRawFrame(
                    buffer.getvalue(),
                    sample_rate=frame.sample_rate,
                    num_channels=frame.num_channels,
                )
                frame = wav_frame

        await self._write_frame(frame)

        # To avoid the audio breaks/glitches during playback, we want to send
        # the audio data in burst mode initially for some duration
        # and then sleep to simulate the audio playback. This will ensure
        # that there is some audio data always in the buffer and it is never empty.
        if self._total_sent_interval >= self._params.burst_mode_duration:
            # Simulate audio playback with a sleep.
            await self._write_audio_sleep()
        self._total_sent_interval += self._send_interval

    async def _write_frame(self, frame: Frame):
        try:
            payload = await self._params.serializer.serialize(frame)
            if payload:
                await self._client.send(payload)
        except Exception as e:
            logger.error(f"{self} exception sending data: {e.__class__.__name__} ({e})")

    async def _write_audio_sleep(self):
        # Simulate a clock.
        current_time = time.monotonic()
        sleep_duration = max(0, self._next_send_time - current_time)
        await asyncio.sleep(sleep_duration)
        if sleep_duration == 0:
            self._next_send_time = time.monotonic() + self._send_interval
        else:
            self._next_send_time += self._send_interval


class ACETransport(BaseTransport):
    """Transport for the ACETransport.

    Manages:
        - RTSP input streams
        - WebSocket connections
        - Audio input/output
        - Client event handling
    """

    def __init__(
        self,
        websocket: WebSocket,
        params: ACETransportParams,
        input_name: str | None = None,
        output_name: str | None = None,
    ):
        """Initializes the transport.

        Args:
            websocket: WebSocket connection instance.
            params: Transport configuration parameters.
            input_name: Optional name for input transport.
            output_name: Optional name for output transport.
        """
        super().__init__(input_name=input_name, output_name=output_name)

        self._params = params

        self._callbacks = FastAPIWebsocketCallbacks(
            on_client_connected=self._on_client_connected,
            on_client_disconnected=self._on_client_disconnected,
            on_session_timeout=self._on_session_timeout,
        )

        self._client = self.create_websocket_client(websocket)
        self._input = ACEInputTransport(self, self._client, self._params, name=self._input_name)
        self._output = ACEOutputTransport(self, self._client, self._params, name=self._output_name)

        # Register supported handlers. The user will only be able to register
        # these handlers.
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")
        self._register_event_handler("on_session_timeout")

    def create_websocket_client(self, websocket: WebSocket):
        """Create a new WebSocket client.

        Args:
            websocket: The WebSocket connection.

        Returns:
            FastAPIWebsocketClient: The new WebSocket client.
        """
        if not websocket:
            return None
        else:
            is_binary = self._params.serializer.type == FrameSerializerType.BINARY
            return FastAPIWebsocketClient(websocket, is_binary, self._callbacks)

    async def update_websocket(self, websocket: WebSocket):
        """Update the WebSocket connection.

        Args:
            websocket: The new WebSocket connection.
        """
        self._client = self.create_websocket_client(websocket)
        await self._input.set_websocket(self._client)
        await self._output.set_websocket(self._client)

    def input(self) -> ACEInputTransport:
        """Get the input transport instance.

        Returns:
            ACEInputTransport: The input transport for handling incoming data.
        """
        return self._input

    def output(self) -> ACEOutputTransport:
        """Get the output transport instance.

        Returns:
            ACEOutputTransport: The output transport for handling outgoing data.
        """
        return self._output

    async def _on_client_connected(self, websocket):
        await self._call_event_handler("on_client_connected", websocket)

    async def _on_client_disconnected(self, websocket):
        await self._call_event_handler("on_client_disconnected", websocket)

    async def _on_session_timeout(self, websocket):
        await self._call_event_handler("on_session_timeout", websocket)
