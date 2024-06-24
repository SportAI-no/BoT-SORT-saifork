from dataclasses import dataclass

@dataclass
class BoTSORTParams:
    name = ""
    with_reid = False
    ablation = ""
    proximity_thresh = 0.5  # Min IOU for matching
    appearance_thresh = 0.25

    track_low_thresh = 0.1  # Confidence threshold
    track_high_thresh = 0.3
    new_track_thresh = 0.6 #track_high_thresh + 0.1

    track_buffer = 15
    match_thresh = 0.8
    aspect_ratio_thresh = 1.6
    fuse_score = False

    # cmc_method = "orb"
    cmc_method = "None"
    mot20 = False

