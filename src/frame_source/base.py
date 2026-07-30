from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class FrameSource(ABC):

    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        pass

    def release(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        frame = self.read()
        if frame is None:
            raise StopIteration
        return frame

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()
