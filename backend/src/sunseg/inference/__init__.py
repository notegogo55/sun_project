"""บริการ inference ที่ webapp เรียกใช้"""

from .forecast import ForecastService
from .segment import SegmentationService

__all__ = ["ForecastService", "SegmentationService"]
