import cv2
import numpy as np
import pyrealsense2 as rs
import supervision as sv
from ultralytics import YOLO

if __name__ == "__main__":
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Start streaming
    pipeline.start(config)
    model = YOLO(
        "/home/wjc/Storage/humanoid_head/open_source/gaze_point_select_ws/src/STAD/scripts/mmaction2/weights/yolo11m.pt"
    )
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    tracker = sv.ByteTrack()
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        # Convert images to numpy arrays

        image = np.asanyarray(color_frame.get_data())
        result = model(image, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)
        labels = [f"{tracker_id}" for tracker_id in detections.tracker_id]
        annotated_image = image.copy()
        annotated_image = box_annotator.annotate(annotated_image, detections=detections)
        annotated_image = label_annotator.annotate(
            annotated_image, detections=detections, labels=labels
        )

        cv2.imshow("a", annotated_image)
        cv2.waitKey(1)
