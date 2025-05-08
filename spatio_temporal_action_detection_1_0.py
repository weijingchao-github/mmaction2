"""
只对画面中最后一帧有的人做STAD，最后一帧中没有这个人，这个人可能已经离开摄像头视野了，
不会再考虑看他。
后面也可以尝试像目标跟踪中的处理一样，如果某个人恰好在最后一帧附近被遮住了，后面又出现了的
情况，最后一帧虽然没有这个人，但是保留这个人动作的历史，没有这个人输出时不输出这个人的信息，
但是更新这个人动作的时间信息，直到消亡期这个人的动作历史才消除。
目前简单处理，只保留最后一帧有的人的动作历史，最后一帧没有的人动作历史会被从记录中清除。
"""

"""
通过读MMAction2的官方代码，输入神经网络的是当前帧前后16张图片在当前帧bbox位置的特征序列，利用
特征序列来判断当前帧bbox位置的这个人是什么动作。
这里我想的是假设原模型是用这特征序列（16帧）判断这个人的动作，利用的是一整个特征序列提供的信息，
也可能通过训练更加关注于中间特征序列那个人的动作，来解决一个序列后面人动作和中间不一样的情况，那
如果只有序列后半部分有人的bbox，也可以来判断这个人的动作。
"""

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
from face_and_person.msg import PersonBboxPerImage
from mmengine.runner import load_checkpoint
from mmengine.structures import InstanceData
from STAD.msg import CurrentPersonActionPair, PersonActionPair, STADResult

from mmaction.registry import MODELS
from mmaction.structures import ActionDataSample

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


class ActionRecorder:
    action_info = {}
    # record_counts = 0


class VideoMAE:
    def __init__(self):
        # others
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
            label_map=os.path.join(
                os.path.dirname(__file__),
                # "tools/data/ava/label_map_Chinese_80.txt",
                "tools/data/ava/label_map_Chinese_part.txt",
                # os.path.dirname(__file__),
                # "tools/data/ava/label_map.txt",
            ),
            device="cuda:0",
            short_side=256,
            action_score_thr=0.3,
        )
        recording_seconds = rospy.get_param("/history_record_seconds")
        self.duration = rospy.get_param("/llm_inferecnce_duration")
        self.recording_counts = int(recording_seconds / self.duration)
        audio_image_pub_frequency = rospy.get_param("/pub_frequency")
        self.per_seq_recv_times = int(self.duration * audio_image_pub_frequency)
        self.recv_msg_buffer = deque(maxlen=self.per_seq_recv_times)
        # self.central_index = self.per_seq_recv_times // 2
        self.recv_counter = 0
        self.recv_msg_buffer_copy = None
        self.do_STAD_flag = False
        self.seq_id = -1
        self.viz_flag = False
        # currently recorder don't need to init
        # model init
        self.config = mmengine.Config.fromfile(self.algs_args.config)
        self.config.model.backbone.pretrained = None
        self.model = MODELS.build(self.config.model)
        load_checkpoint(self.model, self.algs_args.checkpoint, map_location="cpu")
        self.model.to(self.algs_args.device)
        self.model.eval()
        self.img_norm_cfg = dict(
            mean=np.array(self.config.model.data_preprocessor.mean),
            std=np.array(self.config.model.data_preprocessor.std),
            to_rgb=False,
        )
        # Load label_map
        self.label_map = load_label_map(self.algs_args.label_map)
        # loop
        self.thread_running = True
        self.loop_thread = threading.Thread(target=self._pipeline)
        self.loop_thread.start()
        # ROS init
        self.bridge = CvBridge()
        rospy.Subscriber(
            "/face_and_person/person_detect_result",
            PersonBboxPerImage,
            self._process_recv_msg,
            queue_size=10,
        )
        self.pub_SATD_result = rospy.Publisher("STAD_result", STADResult, queue_size=10)

    def thread_shutdown(self):
        self.thread_running = False
        self.loop_thread.join()

    def _process_recv_msg(self, person_detect_result):
        person_detect_result_processed = {}
        person_detect_result_processed["seq_id"] = person_detect_result.seq_id
        person_detect_result_processed["bboxes_xyxy_and_ids"] = (
            person_detect_result.bboxes_xyxy_and_ids
        )
        person_detect_result_processed["color_image"] = self.bridge.imgmsg_to_cv2(
            person_detect_result.color_image, desired_encoding="bgr8"
        )
        self.recv_msg_buffer.append(person_detect_result_processed)
        self.recv_counter += 1
        if self.recv_counter == self.per_seq_recv_times:
            self.recv_msg_buffer_copy = copy.deepcopy(self.recv_msg_buffer)
            self.seq_id = self.recv_msg_buffer_copy[0]["seq_id"]
            self.do_STAD_flag = True
            self.recv_counter = 0

    def _pipeline(self):
        while self.thread_running:
            if not self.do_STAD_flag:
                time.sleep(0.001)
                continue
            self.do_STAD_flag = False

            STAD_result = STADResult()
            STAD_result.seq_id = self.seq_id
            STAD_result.current_person_and_action = []
            STAD_result.person_action_pair = []

            # Get current person track results
            current_person_bboxes_xyxy_and_ids = self.recv_msg_buffer_copy[-1][
                "bboxes_xyxy_and_ids"
            ]
            # print(current_person_bboxes_xyxy_and_ids)
            if len(current_person_bboxes_xyxy_and_ids) == 0:
                ActionRecorder.action_info = {}
                # ActionRecorder.record_counts += 1
                self.pub_SATD_result.publish(STAD_result)
                continue

            humans_detect_result_xyxy = []
            for bbox_xyxy_and_id in current_person_bboxes_xyxy_and_ids:
                humans_detect_result_xyxy.append(bbox_xyxy_and_id.bbox_xyxy)

            humans_detect_result_xyxy = np.array(
                humans_detect_result_xyxy, dtype=np.float32
            )

            # resize frames to shortside
            h, w, _ = (self.recv_msg_buffer_copy[-1]["color_image"]).shape
            new_w, new_h = mmcv.rescale_size(
                (w, h), (self.algs_args.short_side, np.Inf)
            )
            frames = [
                mmcv.imresize(recv_msg["color_image"], (new_w, new_h)).astype(
                    np.float32
                )
                for recv_msg in self.recv_msg_buffer_copy
            ]
            w_ratio, h_ratio = new_w / w, new_h / h

            # process human detection results
            humans_detect_result_xyxy[:, 0:4:2] *= w_ratio
            humans_detect_result_xyxy[:, 1:4:2] *= h_ratio
            humans_detect_result_xyxy = torch.from_numpy(
                humans_detect_result_xyxy[:, :4]
            ).to(self.algs_args.device)

            # 针对目前48张的情况，提取16张
            images_detect = frames[2::3]
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
                scores = result[0].pred_instances.scores  # shape[N, 81]
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
                for prediction_one_person in prediction:
                    sit_score = -1
                    stand_score = -1
                    for label in prediction_one_person:
                        if label[0] == "坐":
                            sit_score = label[1]
                        if label[0] == "站立":
                            stand_score = label[1]
                    if sit_score == -1 or stand_score == -1:
                        continue
                    else:
                        if sit_score > stand_score:
                            prediction_one_person.remove(("站立", stand_score))
                        else:
                            prediction_one_person.remove(("坐", sit_score))
            # manage recorder
            current_track_id_set = []
            for index, bbox_xyxy_and_id in enumerate(
                current_person_bboxes_xyxy_and_ids
            ):
                track_id = bbox_xyxy_and_id.track_id
                current_track_id_set.append(track_id)
                action_label = ""
                for pred_result in prediction[index]:
                    action_label += pred_result[0]
                    action_label += ","
                if track_id not in ActionRecorder.action_info.keys():
                    if action_label == "":  # STAD识别不出这个人的动作，那就不记录这个人
                        continue
                    else:
                        action_label = action_label[:-1]  # 不要最后的逗号
                        ActionRecorder.action_info[track_id] = {
                            "time_index": [],
                            "action": [],
                        }
                        (ActionRecorder.action_info[track_id]["time_index"]).append(0)
                        (ActionRecorder.action_info[track_id]["action"]).append(
                            action_label
                        )
                else:
                    if (
                        action_label == ""
                    ):  # 如果STAD识别不出这个人的动作，那就把这个人从记录中删除
                        ActionRecorder.action_info.pop(track_id)
                    else:
                        action_label = action_label[:-1]  # 不要最后的逗号
                        if (
                            len(ActionRecorder.action_info[track_id]["time_index"])
                            == self.recording_counts
                        ):
                            ActionRecorder.action_info[track_id]["time_index"] = (
                                ActionRecorder.action_info[track_id]["time_index"][1:]
                            )
                            ActionRecorder.action_info[track_id]["action"] = (
                                ActionRecorder.action_info[track_id]["action"][1:]
                            )
                        ActionRecorder.action_info[track_id]["time_index"] = [
                            i + 1
                            for i in ActionRecorder.action_info[track_id]["time_index"]
                        ]
                        (ActionRecorder.action_info[track_id]["time_index"]).append(0)
                        (ActionRecorder.action_info[track_id]["action"]).append(
                            action_label
                        )
            # 在最后一张图像中没有的人，清空recoder关于这个人的cache
            for track_id in list(ActionRecorder.action_info.keys()):
                if track_id not in current_track_id_set:
                    ActionRecorder.action_info.pop(track_id)
            # ActionRecorder.record_counts += 1

            # pub msg
            for track_id in ActionRecorder.action_info.keys():
                current_person_action_pair = CurrentPersonActionPair()
                current_person_action_pair.track_id = track_id
                current_person_action_pair.action_type = ActionRecorder.action_info[
                    track_id
                ]["action"][-1]
                STAD_result.current_person_and_action.append(current_person_action_pair)
                person_action_pair = PersonActionPair()
                person_action_pair.track_id = track_id
                person_action_pair.start_time = []
                person_action_pair.end_time = []
                person_action_pair.action_type = []
                for time_index, action in zip(
                    ActionRecorder.action_info[track_id]["time_index"],
                    ActionRecorder.action_info[track_id]["action"],
                ):
                    start_time = time_index * self.duration
                    end_time = (time_index + 1) * self.duration
                    person_action_pair.start_time.append(start_time)
                    person_action_pair.end_time.append(end_time)
                    person_action_pair.action_type.append(action)
                STAD_result.person_action_pair.append(person_action_pair)
            self.pub_SATD_result.publish(STAD_result)

            if self.viz_flag:
                stad_result = pack_result(
                    humans_detect_result_xyxy, prediction, new_h, new_w
                )
                image_viz = self.recv_msg_buffer_copy[-1]["color_image"]
                image_viz = visualize(image_viz, stad_result)
                cv2.imshow("image_viz", image_viz)
                cv2.waitKey(1)
                current_person_str = "Current person: "
                current_person_and_action = STAD_result.current_person_and_action
                for person_action in current_person_and_action:
                    current_person_str += (
                        f"{person_action.track_id}-{person_action.action_type}, "
                    )
                print(current_person_str)
                person_action_history = "History:\n"
                person_action_pair = STAD_result.person_action_pair
                for person_action in person_action_pair:
                    person_action_history += f"{person_action.track_id}: "
                    for start_time, end_time, action_type in zip(
                        person_action.start_time,
                        person_action.end_time,
                        person_action.action_type,
                    ):
                        person_action_history += (
                            f"{action_type} at {start_time}s-{end_time}s, "
                        )
                    person_action_history += "\n"
                print(person_action_history)
                if enable_record_video:
                    video_writer.write(image_viz)

    # def _detect_human(self, image_detect):
    #     humans_detect_result = self.human_detector(
    #         image_detect, conf=self.algs_args.human_detect_conf, verbose=False
    #     )
    #     humans_detect_result_xyxy = []
    #     for i, cls in enumerate(humans_detect_result[0].boxes.cls):
    #         if cls == 0:
    #             humans_detect_result_xyxy.append(
    #                 humans_detect_result[0].boxes.xyxy[i].detach().cpu().numpy()
    #             )
    #     return humans_detect_result_xyxy


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
