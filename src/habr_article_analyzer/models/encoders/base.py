from abc import ABC, abstractmethod
from typing import Any


class TextEncoder(ABC):
    """Interface for text encoders."""

    @abstractmethod
    def encode(self, text: str) -> Any:
        """
        Converts input text into a vector representation.
        """
        raise NotImplementedError
