import dataclasses
from typing import Optional


@dataclasses.dataclass
class Pixel:
    min_intensity: int = 255
    max_intensity: int = 0
    last_diff_intensity: Optional[float] = None

    def get_intensity_avg(self):
        return (self.min_intensity + self.max_intensity) / 2

