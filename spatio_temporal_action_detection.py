import os
import sys

path = os.path.dirname(__file__)
sys.path.insert(0, path)

import copy
import threading
import time
from collections import deque
from types import SimpleNamespace

import cv2
import mmcv
import mmengine
import numpy as np
import rospy
import torch
from cv_bridge import CvBridge
from mmaction.registry import MODELS
from mmaction.structures import ActionDataSample
from mmengine.runner import load_checkpoint
from mmengine.structures import InstanceData
from sensor_msgs.msg import Image
from ultralytics import YOLO

# record video
enable_record_video = False
if enable_record_video:
    fourcc = cv2.VideoWriter.fourcc(*"XVID")
    fps = 30
    output_video_path = os.path.join(
        os.path.dirname(__file__), "output_video_color.avi"
    )
    img_width = 640
    img_height = 480
    video_writer = cv2.VideoWriter(
        output_video_path, fourcc, fps, (img_width, img_height)
    )

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.5
FONTCOLOR = (255, 255, 255)  # BGR, white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1


def hex2color(h):
    """Convert the 6-digit hex string to tuple of 3 int value (RGB)"""
    return (int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16))


plate_blue = "03045e-023e8a-0077b6-0096c7-00b4d8-48cae4"
plate_blue = plate_blue.split("-")
plate_blue = [hex2color(h) for h in plate_blue]
plate_green = "004b23-006400-007200-008000-38b000-70e000"
plate_green = plate_green.split("-")
plate_green = [hex2color(h) for h in plate_green]


def load_label_map(file_path):
    """Load Label Map.

    Args:
        file_path (str): The file path of label map.
    Returns:
        dict: The label map (int -> label name).
    """
    lines = open(file_path).readlines()
    lines = [x.strip().split(": ") for x in lines]
    return {int(x[0]): x[1] for x in lines}


def abbrev(name):
    """Get the abbreviation of label name:

    'take (an object) from (a person)' -> 'take ... from ...'
    """
    while name.find("(") != -1:
        st, ed = name.find("("), name.find(")")
        name = name[:st] + "..." + name[ed + 1 :]
    return name


def visualize(frames, annotations, plate=plate_blue, max_num=5):
    """Visualize frames with predicted annotations.

    Args:
        frames (list[np.ndarray]): Frames for visualization, note that
            len(frames) % len(annotations) should be 0.
        annotations (list[list[tuple]]): The predicted results.
        plate (str): The plate used for visualization. Default: plate_blue.
        max_num (int): Max number of labels to visualize for a person box.
            Default: 5.
    Returns:
        list[np.ndarray]: Visualized frames.
    """
    frames = [frames]
    annotations = [annotations]
    assert max_num + 1 <= len(plate)
    plate = [x[::-1] for x in plate]
    frames_out = copy.deepcopy(frames)
    nf, na = len(frames), len(annotations)
    assert nf % na == 0
    nfpa = len(frames) // len(annotations)
    anno = None
    h, w, _ = frames[0].shape
    scale_ratio = np.array([w, h, w, h])
    for i in range(na):
        anno = annotations[i]
        if anno is None:
            continue
        for j in range(nfpa):
            ind = i * nfpa + j
            frame = frames_out[ind]
            for ann in anno:
                box = ann[0]
                label = ann[1]
                if not len(label):
                    continue
                score = ann[2]
                box = (box * scale_ratio).astype(np.int64)
                st, ed = tuple(box[:2]), tuple(box[2:])
                cv2.rectangle(frame, st, ed, plate[0], 2)
                for k, lb in enumerate(label):
                    if k >= max_num:
                        break
                    text = abbrev(lb)
                    text = ": ".join([text, f"{score[k]:>.2f}"])
                    location = (0 + st[0], 18 + k * 18 + st[1])
                    textsize = cv2.getTextSize(text, FONTFACE, FONTSCALE, THICKNESS)[0]
                    textwidth = textsize[0]
                    diag0 = (location[0] + textwidth, location[1] - 14)
                    diag1 = (location[0], location[1] + 2)
                    cv2.rectangle(frame, diag0, diag1, plate[k + 1], -1)
                    cv2.putText(
                        frame,
                        text,
                        location,
                        FONTFACE,
                        FONTSCALE,
                        FONTCOLOR,
                        THICKNESS,
                        LINETYPE,
                    )

    frames_out = frames_out[0]
    return frames_out


def pack_result(human_detection, result, img_h, img_w):
    """Short summary.

    Args:
        human_detection (np.ndarray): Human detection result.
        result (type): The predicted label of each human proposal.
        img_h (int): The image height.
        img_w (int): The image width.
    Returns:
        tuple: Tuple of human proposal, label name and label score.
    """
    human_detection[:, 0::2] /= img_w
    human_detection[:, 1::2] /= img_h
    results = []
    if result is None:
        return None
    for prop, res in zip(human_detection, result):
        res.sort(key=lambda x: -x[1])
        results.append(
            (prop.data.cpu().numpy(), [x[0] for x in res], [x[1] for x in res])
        )
    return results


class VideoMAE:
    def __init__(self):
        # algs' parameters
        self.algs_args = SimpleNamespace(
            clip_len=16,
            frame_interval=4,
            config=os.path.join(
                os.path.dirname(__file__),
                "configs/detection/videomae/vit-base-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb.py",
            ),
            checkpoint=os.path.join(
                os.path.dirname(__file__),
                "weights/vit-base-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb_20230314-3dafab75.pth",
            ),
            action_score_thr=0.3,
            label_map=os.path.join(
                os.path.dirname(__file__), "tools/data/ava/label_map.txt"
            ),
            device="cuda:0",
            short_side=256,
            yolo_checkpoint=os.path.join(
                os.path.dirname(__file__), "weights/yolo11m.pt"
            ),
            human_detect_conf=0.7,
        )
        # model init
        self.config = mmengine.Config.fromfile(self.algs_args.config)
        self.config.model.backbone.pretrained = None
        self.model = MODELS.build(self.config.model)
        load_checkpoint(self.model, self.algs_args.checkpoint, map_location="cpu")
        self.model.to(self.algs_args.device)
        self.model.eval()
        # human detector init
        self.human_detector = YOLO(self.algs_args.yolo_checkpoint, verbose=False)
        # Load label_map
        self.label_map = load_label_map(self.algs_args.label_map)
        # ROS init
        self.bridge = CvBridge()
        rospy.Subscriber(
            "/camera/color/image_raw",
            Image,
            self._collect_image,
            queue_size=1,
        )
        # others
        self.max_buffer_length = (
            self.algs_args.clip_len * self.algs_args.frame_interval
            - (self.algs_args.frame_interval - 1)
        )
        self.image_buffer = deque(maxlen=self.max_buffer_length)
        self.img_norm_cfg = dict(
            mean=np.array(self.config.model.data_preprocessor.mean),
            std=np.array(self.config.model.data_preprocessor.std),
            to_rgb=False,
        )
        # loop
        self.thread_running = True
        self.loop_thread = threading.Thread(target=self._pipeline)
        self.loop_thread.start()

    def thread_shutdown(self):
        self.thread_running = False
        self.loop_thread.join()

    def _collect_image(self, image_color):
        image_color = self.bridge.imgmsg_to_cv2(image_color, desired_encoding="bgr8")
        self.image_buffer.append(image_color)

    def _pipeline(self):
        while self.thread_running:
            if len(self.image_buffer) < self.max_buffer_length:
                # 从python GIL里抢占用
                time.sleep(0.1)
                continue

            current_image_buffer = copy.deepcopy(list(self.image_buffer))
            h, w, _ = current_image_buffer[0].shape

            # Get Human detection results
            humans_detect_result_xyxy = self._detect_human(current_image_buffer[-1])
            if len(humans_detect_result_xyxy) == 0:
                print("No human in camera.")
                continue
            humans_detect_result_xyxy = np.array(
                humans_detect_result_xyxy, dtype=np.float32
            )

            # resize frames to shortside
            new_w, new_h = mmcv.rescale_size(
                (w, h), (self.algs_args.short_side, np.Inf)
            )
            frames = [
                mmcv.imresize(img, (new_w, new_h)).astype(np.float32)
                for img in current_image_buffer
            ]
            w_ratio, h_ratio = new_w / w, new_h / h

            # process human detection results
            humans_detect_result_xyxy[:, 0:4:2] *= w_ratio
            humans_detect_result_xyxy[:, 1:4:2] *= h_ratio
            humans_detect_result_xyxy = torch.from_numpy(
                humans_detect_result_xyxy[:, :4]
            ).to(self.algs_args.device)

            images_detect = frames[::4]
            _ = [
                mmcv.imnormalize_(image_detect, **self.img_norm_cfg)
                for image_detect in images_detect
            ]
            # THWC -> CTHW -> 1CTHW
            input_array = np.stack(images_detect).transpose((3, 0, 1, 2))[np.newaxis]
            input_tensor = torch.from_numpy(input_array).to(self.algs_args.device)

            datasample = ActionDataSample()
            datasample.proposals = InstanceData(bboxes=humans_detect_result_xyxy)
            datasample.set_metainfo(dict(img_shape=(new_h, new_w)))
            with torch.no_grad():
                result = self.model(input_tensor, [datasample], mode="predict")
                scores = result[0].pred_instances.scores
                prediction = []
                # N proposals
                for i in range(humans_detect_result_xyxy.shape[0]):
                    prediction.append([])
                # Perform action score thr
                for i in range(scores.shape[1]):
                    if i not in self.label_map:
                        continue
                    for j in range(humans_detect_result_xyxy.shape[0]):
                        if scores[j, i] > self.algs_args.action_score_thr:
                            prediction[j].append(
                                (self.label_map[i], scores[j, i].item())
                            )
            stad_result = pack_result(
                humans_detect_result_xyxy, prediction, new_h, new_w
            )
            image_viz = current_image_buffer[-1]
            image_viz = visualize(image_viz, stad_result)
            cv2.imshow("image_viz", image_viz)
            cv2.waitKey(1)
            if enable_record_video:
                video_writer.write(image_viz)

    def _detect_human(self, image_detect):
        humans_detect_result = self.human_detector(
            image_detect, conf=self.algs_args.human_detect_conf, verbose=False
        )
        humans_detect_result_xyxy = []
        for i, cls in enumerate(humans_detect_result[0].boxes.cls):
            if cls == 0:
                humans_detect_result_xyxy.append(
                    humans_detect_result[0].boxes.xyxy[i].detach().cpu().numpy()
                )
        return humans_detect_result_xyxy


def main():
    rospy.init_node("spatio_temporal_action_detection")
    video_mae = VideoMAE()
    try:
        rospy.spin()
    finally:
        if enable_record_video:
            video_writer.release()
        video_mae.thread_shutdown()


if __name__ == "__main__":
    main()
