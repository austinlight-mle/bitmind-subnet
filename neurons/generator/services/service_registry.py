import os
from typing import Dict, List, Optional, Any

import bittensor as bt

from .base_service import BaseGenerationService
from .openai_service import OpenAIService
from .openrouter_service import OpenRouterService
from .stabilityai_service import StabilityAIService
from .local_service import LocalService
from .gemini_service import GeminiVideoService


SERVICE_MAP: Dict[str, Any] = {
    "openai": OpenAIService,
    "openrouter": OpenRouterService,
    "local": LocalService,
    "stabilityai": StabilityAIService,
    "gemini": GeminiVideoService,
}


class ServiceRegistry:
    """Registry for managing generation services.

    Set per-modality service via env vars:
      IMAGE_SERVICE=openai|openrouter|local|stabilityai|none
      VIDEO_SERVICE=openai|local|gemini|none

    Services:
      - openai: DALL-E 3 + Sora video (requires OPENAI_API_KEY)
      - openrouter: Google Gemini image via OpenRouter (requires OPEN_ROUTER_API_KEY)
      - local: Local Stable Diffusion / AnimateDiff models
      - stabilityai: Stability AI image generation (requires STABILITY_API_KEY)
      - gemini: Veo video via Gemini API (requires GEMINI_API_KEY)
      - none: Disable this modality (no service loaded)

    If not set, falls back to loading all available services.
    """

    def __init__(self, config: Any = None) -> None:
        self.config = config
        # Explicit per-modality services, e.g. {"image": OpenAIService(...), "video": GeminiVideoService(...)}
        self.services: Dict[str, BaseGenerationService] = {}
        # Fallback list when no explicit IMAGE_SERVICE/VIDEO_SERVICE is configured
        self._all_services: List[BaseGenerationService] = []
        self._initialize_services()

    def _initialize_services(self) -> None:
        """Initialize services based on IMAGE_SERVICE and VIDEO_SERVICE env vars."""
        image_service = os.getenv("IMAGE_SERVICE", "").lower().strip()
        video_service = os.getenv("VIDEO_SERVICE", "").lower().strip()

        if image_service or video_service:
            self._init_modality_services(image_service, video_service)
        else:
            self._init_all_services()

    def _init_modality_services(self, image_service: str, video_service: str) -> None:
        """Initialize specific services for each modality based on env vars."""
        initialized: set[str] = set()

        # Image service
        if image_service == "none":
            bt.logging.info("IMAGE_SERVICE=none, no image service will be loaded")
        elif image_service:
            service = self._create_service(image_service, "image")
            if service:
                self.services["image"] = service
                initialized.add(image_service)

        # Video service
        if video_service == "none":
            bt.logging.info("VIDEO_SERVICE=none, no video service will be loaded")
        elif video_service:
            # Optionally reuse a non-local service for both modalities
            can_reuse = video_service in initialized and video_service != "local"
            if can_reuse:
                self.services["video"] = (
                    self.services.get("image")
                    or self._create_service(video_service, "video")
                )
            else:
                service = self._create_service(video_service, "video")
                if service:
                    self.services["video"] = service

    def _create_service(self, service_name: str, modality: str) -> Optional[BaseGenerationService]:
        """Create and validate a single service instance."""

        if service_name not in SERVICE_MAP:
            bt.logging.error(
                f"Unknown service: {service_name}. Valid options: {list(SERVICE_MAP.keys())}"
            )
            return None

        bt.logging.info(f"Initializing {service_name} for {modality}")
        service_class = SERVICE_MAP[service_name]

        try:
            if service_name == "local":
                # Local service may want to know which modality it's targeting
                service = service_class(self.config, target_modality=modality)
            else:
                service = service_class(self.config)

            if service.is_available():
                bt.logging.success(f"\x1b[92m\x1b[1m{service.name} ready for {modality}\x1b[0m")
                return service

            bt.logging.error(
                f"\x1b[91m\x1b[1m{service.name} configured for {modality} but not available (check API keys)\x1b[0m"
            )
        except Exception as e:  # noqa: BLE001
            bt.logging.error(f"Failed to initialize {service_name}: {e}")

        return None

    def _init_all_services(self) -> None:
        """Initialize all available services (fallback behavior)."""

        bt.logging.info(
            "No IMAGE_SERVICE/VIDEO_SERVICE set, initializing all available services..."
        )

        for name, service_class in SERVICE_MAP.items():
            try:
                service = service_class(self.config)
                if service.is_available():
                    self._all_services.append(service)
                    bt.logging.info(f"\x1b[92m\x1b[1m{service.name} is available\x1b[0m")
                else:
                    bt.logging.info(f"\x1b[91m\x1b[1m{service.name} is not available\x1b[0m")
            except Exception as e:  # noqa: BLE001
                bt.logging.warning(f"Failed to initialize {name}: {e}")

        bt.logging.info(f"Initialized {len(self._all_services)} generation services")

    def get_service(self, modality: str) -> Optional[BaseGenerationService]:
        """Get the service for a modality (image/video)."""

        # Check explicit modality mapping first
        if modality in self.services:
            service = self.services[modality]
            bt.logging.debug(f"Using {service.name} for {modality}")
            return service

        # Fallback to scanning all services
        for service in self._all_services:
            if service.supports_modality(modality):
                bt.logging.debug(f"Using {service.name} for {modality}")
                return service

        bt.logging.warning(f"No service available for modality={modality}")
        return None

    def get_available_services(self) -> List[Dict[str, Any]]:
        """Get information about all available services."""

        seen: set[str] = set()
        result: List[Dict[str, Any]] = []

        for service in list(self.services.values()) + self._all_services:
            if service.name not in seen:
                seen.add(service.name)
                result.append(service.get_info())

        return result

    def get_all_api_key_requirements(self) -> Dict[str, str]:
        """Get API key requirements from all services (configured or not)."""

        all_requirements: Dict[str, str] = {
            "IMAGE_SERVICE": "Service for images: openai, openrouter, local, stabilityai, or none",
            "VIDEO_SERVICE": "Service for videos: openai, local, gemini, or none",
        }

        for name, service_class in SERVICE_MAP.items():
            try:
                temp_service = service_class(self.config)
                all_requirements.update(temp_service.get_api_key_requirements())
            except Exception as e:  # noqa: BLE001
                bt.logging.warning(
                    f"Failed to get API key requirements from {name}: {e}"
                )

        return all_requirements

    def reload_services(self) -> None:
        """Reload all services (useful for configuration changes)."""

        self.services.clear()
        self._all_services.clear()
        self._initialize_services()

