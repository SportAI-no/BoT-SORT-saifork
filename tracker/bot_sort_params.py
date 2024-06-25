from dataclasses import dataclass


@dataclass
class BoTSORTParams:
    name: str = ""
    with_reid: bool = False
    ablation: str = ""
    proximity_thresh: float = 0.5  # Min IOU for matching
    appearance_thresh: float = 0.25
    track_low_thresh: float = 0.1  # Confidence threshold
    track_high_thresh: float = 0.3
    new_track_thresh: float = 0.6  # track_high_thresh + 0.1
    track_buffer: int = 15
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 1.6
    fuse_score: bool = False
    cmc_method: str = "None"
    mot20: bool = False
