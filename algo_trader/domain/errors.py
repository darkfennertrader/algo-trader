from __future__ import annotations

from typing import Mapping

ErrorContext = Mapping[str, str]


class AlgoTraderError(Exception):
    def __init__(
        self, message: str, *, context: ErrorContext | None = None
    ) -> None:
        self.context = dict(context) if context else {}
        super().__init__(message)


class ConfigError(AlgoTraderError):
    pass


class EnvVarError(AlgoTraderError):
    pass


class ProviderError(AlgoTraderError):
    pass


class ProviderConnectionError(ProviderError):
    pass


class ExportError(AlgoTraderError):
    pass


class DataSourceError(AlgoTraderError):
    pass


class DataProcessingError(AlgoTraderError):
    pass


class InferenceError(AlgoTraderError):
    pass


class ModelSelectionError(AlgoTraderError):
    pass
