import numpy as np

from anno import VideoEntry, AnnotationFrame, AnnotationInstance

from tracker import BoTSORT, BoTSORTParams



def track_entry(
    video_entry: VideoEntry,
) -> VideoEntry:
    args = BoTSORTParams()
    tracker = BoTSORT(args, frame_rate=30)
    new_entry = VideoEntry(
        video_entry.get_source_path(),
        annotation_format=video_entry.get_annotation_format(),
    )
    for frame in video_entry:
        frame: AnnotationFrame = frame

        detections = []
        for instance in frame:
            instance: AnnotationInstance = instance
            xmin, ymin, xmax, ymax = instance.get_bbox()
            x, y = (xmin + xmax) / 2, (ymin + ymax) / 2
            w, h = xmax - xmin, ymax - ymin
            conf = instance.get_box_confidence()
            detections.append([xmin, ymin, xmax, ymax, conf])
            
        detections = np.array(detections)
        new_frame = AnnotationFrame(frame.frame_nr, timestamp=frame.timestamp)
        tracks = tracker.update(detections, None)
        for track in tracks:
            xmin, ymin, xmax, ymax = track.tlbr
            new_frame.add_instance(
                AnnotationInstance(
                    np.array(
                        [
                            [xmin, ymin],
                            [xmax, ymin],
                            [xmax, ymax],
                            [xmin, ymax],
                        ]
                    ),
                    bbox=np.array([xmin, ymin, xmax, ymax]),
                    # box_confidence=track.score,
                    tracking_id=track.track_id,
                    annotation_format=video_entry.get_annotation_format(),
                )
            )
        new_entry.add_frame(new_frame)
    return new_entry