import os
import time
from typing import Any, Dict

import bittensor as bt
import requests

from .base_service import BaseGenerationService
from ..task_manager import GenerationTask


class GeminiVideoService(BaseGenerationService):
    """Video generation via Google's Gemini API (Veo models).

    This service uses the REST `predictLongRunning` endpoint for Veo 3.x models
    and returns raw MP4 bytes, matching the shape expected by the miner:

        {
            "data": <bytes>,
            "metadata": { ... }
        }

    Requirements:
        - GEMINI_API_KEY: Gemini API key with Veo video access

    Default model: `veo-3.1-generate-preview`, overridable per-task via
    `task.parameters["model"]` or globally via GEMINI_VIDEO_MODEL.
    """

    def __init__(self, config: Any = None):
        super().__init__(config)

        self.api_key = os.getenv("GEMINI_API_KEY")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.default_model = os.getenv(
            "GEMINI_VIDEO_MODEL", "veo-3.1-generate-preview"
        )

        if not self.api_key:
            bt.logging.warning("GEMINI_API_KEY not set; Gemini video service unavailable")
        else:
            bt.logging.info("GeminiVideoService initialized with GEMINI_API_KEY")

    # ------------------------------------------------------------------
    # BaseGenerationService interface
    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        """Service is available if an API key is configured."""

        return bool(self.api_key and self.api_key.strip())

    def supports_modality(self, modality: str) -> bool:
        """This service only supports video generation."""

        return modality == "video"

    def get_supported_tasks(self) -> Dict[str, list]:
        """Return supported tasks by modality."""

        return {"video": ["video_generation"]}

    def get_api_key_requirements(self) -> Dict[str, str]:
        """Return required environment variables for this service."""

        return {
            "GEMINI_API_KEY": "Gemini API key for Veo video generation via Gemini API",
            "GEMINI_VIDEO_MODEL": "Optional override for Veo video model (default veo-3.1-generate-preview)",
        }

    def process(self, task: GenerationTask) -> Dict[str, Any]:
        """Process a video generation task using Veo.

        The task is expected to have `modality == "video"` and may include
        parameters such as:

            - model: Veo model name (default veo-3.1-generate-preview)
            - seconds / duration / durationSeconds: requested duration
            - aspectRatio / aspect_ratio: e.g. "16:9" or "9:16"
            - resolution: e.g. "720p" or "1080p"
            - negativePrompt / negative_prompt: text to avoid
            - poll_interval: seconds between status polls (default 10.0)
            - timeout: overall timeout in seconds (default 600.0)
        """

        if task.modality != "video":
            raise ValueError(
                f"GeminiVideoService only supports 'video' modality, got {task.modality}"
            )

        if not self.is_available():
            raise RuntimeError(
                "GeminiVideoService is not available (missing GEMINI_API_KEY)"
            )

        return self._generate_video(task)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _generate_video(self, task: GenerationTask) -> Dict[str, Any]:
        params: Dict[str, Any] = task.parameters or {}

        model: str = params.get("model", self.default_model)

        duration_value = params.get(
            "durationSeconds", params.get("seconds", params.get("duration", 4))
        )
        duration_seconds = str(duration_value)

        aspect_ratio = params.get("aspectRatio", params.get("aspect_ratio", "16:9"))
        resolution = params.get("resolution", "720p")
        negative_prompt = params.get(
            "negativePrompt", params.get("negative_prompt")
        )

        poll_interval = float(params.get("poll_interval", 10.0))
        timeout = float(params.get("timeout", 600.0))

        bt.logging.info(
            "Gemini/Veo video generation requested: "
            f"model={model}, durationSeconds={duration_seconds}, "
            f"aspectRatio={aspect_ratio}, resolution={resolution}"
        )

        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        body: Dict[str, Any] = {
            "instances": [
                {
                    "prompt": task.prompt,
                }
            ],
            "parameters": {
                "durationSeconds": duration_seconds,
                "aspectRatio": aspect_ratio,
                "resolution": resolution,
            },
        }

        if negative_prompt:
            body["parameters"]["negativePrompt"] = negative_prompt

        url = f"{self.base_url}/models/{model}:predictLongRunning"
        bt.logging.info(f"Calling Gemini/Veo predictLongRunning at {url}")

        response = requests.post(url, headers=headers, json=body, timeout=60)
        if response.status_code != 200:
            bt.logging.error(
                f"Gemini video request failed: {response.status_code} {response.text}"
            )
            raise RuntimeError(
                f"Gemini video request failed with status {response.status_code}"
            )

        op = response.json()
        op_name = op.get("name")
        if not op_name:
            bt.logging.error(f"Gemini video operation response missing 'name': {op}")
            raise RuntimeError("Gemini video operation response missing 'name'")

        bt.logging.info(f"Gemini/Veo operation started: name={op_name}")

        start_time = time.monotonic()
        status_url = f"{self.base_url}/{op_name}"

        while True:
            elapsed = time.monotonic() - start_time
            if elapsed > timeout:
                raise TimeoutError(
                    f"Gemini/Veo video generation timed out after {timeout}s "
                    f"(operation={op_name})"
                )

            status_resp = requests.get(
                status_url,
                headers={"x-goog-api-key": self.api_key},
                timeout=60,
            )
            if status_resp.status_code != 200:
                bt.logging.error(
                    "Gemini operation status failed: "
                    f"{status_resp.status_code} {status_resp.text}"
                )
                raise RuntimeError(
                    f"Gemini operation status failed with status {status_resp.status_code}"
                )

            status_data = status_resp.json()
            done = status_data.get("done", False)
            bt.logging.debug(
                f"Gemini/Veo operation status: done={done}, elapsed={elapsed:.1f}s"
            )

            if done:
                if "error" in status_data:
                    raise RuntimeError(
                        f"Gemini/Veo video generation error: {status_data['error']}"
                    )

                response_obj = status_data.get("response") or {}
                video_uri = self._extract_video_uri(response_obj)
                if not video_uri:
                    bt.logging.error(
                        f"Could not find video uri in Gemini/Veo response: {status_data}"
                    )
                    raise RuntimeError("Gemini/Veo response missing video URI")

                bt.logging.info(
                    f"Gemini/Veo video ready, downloading from {video_uri}"
                )

                video_resp = requests.get(
                    video_uri,
                    headers={"x-goog-api-key": self.api_key},
                    timeout=600,
                    allow_redirects=True,
                )
                video_resp.raise_for_status()
                video_bytes = video_resp.content

                bt.logging.success(
                    f"Downloaded {len(video_bytes)} bytes from Gemini/Veo"
                )

                return {
                    "data": video_bytes,
                    "metadata": {
                        "model": model,
                        "provider": "google",
                        "mime_type": "video/mp4",
                        "durationSeconds": duration_seconds,
                        "aspectRatio": aspect_ratio,
                        "resolution": resolution,
                        "operation_name": op_name,
                    },
                }

            time.sleep(poll_interval)

    def _extract_video_uri(self, response_obj: Dict[str, Any]) -> str:
        """Extract the video URI from the Gemini long-running operation response.

        According to the Gemini API Veo docs, the REST shape is:

            response.generateVideoResponse.generatedSamples[0].video.uri
        """

        generate_resp = response_obj.get("generateVideoResponse")
        if isinstance(generate_resp, dict):
            samples = generate_resp.get("generatedSamples") or []
            if samples and isinstance(samples[0], dict):
                video_obj = samples[0].get("video") or {}
                uri = video_obj.get("uri")
                if uri:
                    return uri

        return ""
