from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ZipformerEncoderOutput:
    encoder_out: Optional[Any] = None
    encoder_out_lens: Optional[Any] = None
    encoder_out_full: Optional[Any] = None
