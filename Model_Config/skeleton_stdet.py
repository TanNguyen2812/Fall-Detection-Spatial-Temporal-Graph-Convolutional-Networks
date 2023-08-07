import argparse
import copy as cp
import os
import os.path as osp
import shutil
import warnings

import cv2
import mmcv
import numpy as np
import torch
from mmcv import DictAction
from mmcv.runner import load_checkpoint
from torch.nn.functional import softmax
from mmaction.apis import inference_recognizer
from mmaction.datasets.pipelines import Compose
from mmaction.models import build_detector, build_model, build_recognizer

try:
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    warnings.warn('Failed to import `inference_detector` and `init_detector` '
                  'form `mmdet.apis`. These apis are required in '
                  'skeleton-based applications! ')

try:
    from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                             vis_pose_result)
except (ImportError, ModuleNotFoundError):
    warnings.warn('Failed to import `inference_top_down_pose_model`, '
                  '`init_pose_model`, and `vis_pose_result` form '
                  '`mmpose.apis`. These apis are required in skeleton-based '
                  'applications! ')

try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.5
FONTCOLOR = (255, 255, 255)  # BGR, white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1


def hex2color(h):
    """Convert the 6-digit hex string to tuple of 3 int value (RGB)"""
    return (int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16))


PLATEBLUE = '03045e-023e8a-0077b6-0096c7-00b4d8-48cae4'
PLATEBLUE = PLATEBLUE.split('-')
PLATEBLUE = [hex2color(h) for h in PLATEBLUE]
PLATEGREEN = '004b23-006400-007200-008000-38b000-70e000'
PLATEGREEN = PLATEGREEN.split('-')
PLATEGREEN = [hex2color(h) for h in PLATEGREEN]


def visualize(frames,
              annotations,
              pose_results,
              action_result,
              pose_model,
              plate=PLATEBLUE,
              max_num=5):
    """Visualize frames with predicted annotations.

    Args:
        frames (list[np.ndarray]): Frames for visualization, note that
            len(frames) % len(annotations) should be 0.
        annotations (list[list[tuple]]): The predicted spatio-temporal
            detection results.
        pose_results (list[list[tuple]): The pose results.
        action_result (str): The predicted action recognition results.
        pose_model (nn.Module): The constructed pose model.
        plate (str): The plate used for visualization. Default: PLATEBLUE.
        max_num (int): Max number of labels to visualize for a person box.
            Default: 5.

    Returns:
        list[np.ndarray]: Visualized frames.
    """

    assert max_num + 1 <= len(plate)
    plate = [x[::-1] for x in plate]
    frames_ = cp.deepcopy(frames)
    nf, na = len(frames), len(annotations)
    assert nf % na == 0
    nfpa = len(frames) // len(annotations)
    anno = None
    h, w, _ = frames[0].shape
    scale_ratio = np.array([w, h, w, h])

    # add pose results
    if pose_results:
        for i in range(nf):
            frames_[i] = vis_pose_result(pose_model, frames_[i],
                                         pose_results[i])

    for i in range(na):
        anno = annotations[i]
        if anno is None:
            continue
        for j in range(nfpa):
            ind = i * nfpa + j
            frame = frames_[ind]

            # add action result for whole video
            cv2.putText(frame, action_result, (10, 30), FONTFACE, FONTSCALE,
                        FONTCOLOR, THICKNESS, LINETYPE)

            # add spatio-temporal action detection results
            for ann in anno:
                box = ann[0]
                label = ann[1]
                if not len(label):
                    continue
                score = ann[2]
                box = (box * scale_ratio).astype(np.int64)
                st, ed = tuple(box[:2]), tuple(box[2:])
                if not pose_results:
                    cv2.rectangle(frame, st, ed, plate[0], 2)

                for k, lb in enumerate(label):
                    if k >= max_num:
                        break
                    text = abbrev(lb)
                    text = ': '.join([text, str(score[k])])
                    location = (0 + st[0], 18 + k * 18 + st[1])
                    textsize = cv2.getTextSize(text, FONTFACE, FONTSCALE,
                                               THICKNESS)[0]
                    textwidth = textsize[0]
                    diag0 = (location[0] + textwidth, location[1] - 14)
                    diag1 = (location[0], location[1] + 2)
                    cv2.rectangle(frame, diag0, diag1, plate[k + 1], -1)
                    cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                                FONTCOLOR, THICKNESS, LINETYPE)

    return frames_

def frame_extraction(video_path):
    """Extract frames given video_path.

    Args:
        video_path (str): The video_path.
    """
    # Load the video, extract frames into ./tmp/video_name
    target_dir = osp.join('./tmp', osp.basename(osp.splitext(video_path)[0]))
    # target_dir = osp.join('./tmp','spatial_skeleton_dir')
    os.makedirs(target_dir, exist_ok=True)
    # Should be able to handle videos up to several hours
    frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')
    vid = cv2.VideoCapture(video_path)
    frames = []
    frame_paths = []
    flag, frame = vid.read()
    cnt = 0
    while flag:
        frames.append(frame)
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)
        cv2.imwrite(frame_path, frame)
        cnt += 1
        flag, frame = vid.read()
    return frame_paths, frames


def detection_inference(det_config, det_checkpoint ,frame_paths, det_score_thr=0.5,device='cuda' ):
    """Detect human boxes given frame paths.

    Args:
        args (argparse.Namespace): The arguments.
        frame_paths (list[str]): The paths of frames to do detection inference.

    Returns:
        list[np.ndarray]: The human detection results.
    """
    model = init_detector(det_config, det_checkpoint, device)
    assert model.CLASSES[0] == 'person', ('We require you to use a detector '
                                          'trained on COCO')
    results = []
    print('Performing Human Detection for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for frame_path in frame_paths:
        result = inference_detector(model, frame_path)
        # We only keep human detections with score larger than det_score_thr
        result = result[0][result[0][:, 4] >= det_score_thr]
        results.append(result)
        prog_bar.update()

    return results


def pose_inference(pose_config,pose_checkpoint, frame_paths, det_results, device='cuda'):
    model = init_pose_model(pose_config, pose_checkpoint, device)
    ret = []
    print('Performing Human Pose Estimation for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for f, d in zip(frame_paths, det_results):
        # Align input format
        d = [dict(bbox=x) for x in list(d)]

        pose = inference_top_down_pose_model(model, f, d, format='xyxy')[0]
        ret.append(pose)
        prog_bar.update()
    return ret


def load_label_map(file_path):
    """Load Label Map.

    Args:
        file_path (str): The file path of label map.

    Returns:
        dict: The label map (int -> label name).
    """
    lines = open(file_path).readlines()
    lines = [x.strip().split(': ') for x in lines]
    return {int(x[0]): x[1] for x in lines}


def abbrev(name):
    """Get the abbreviation of label name:

    'take (an object) from (a person)' -> 'take ... from ...'
    """
    while name.find('(') != -1:
        st, ed = name.find('('), name.find(')')
        name = name[:st] + '...' + name[ed + 1:]
    return name


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
            (prop.data.cpu().numpy(), [x[0] for x in res], [x[1]
                                                            for x in res]))
    return results


def expand_bbox(bbox, h, w, ratio=1.25):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    width = x2 - x1
    height = y2 - y1

    square_l = max(width, height)
    new_width = new_height = square_l * ratio

    new_x1 = max(0, int(center_x - new_width / 2))
    new_x2 = min(int(center_x + new_width / 2), w)
    new_y1 = max(0, int(center_y - new_height / 2))
    new_y2 = min(int(center_y + new_height / 2), h)
    return (new_x1, new_y1, new_x2, new_y2)


def cal_iou(box1, box2):
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    intersect = w * h
    union = s1 + s2 - intersect
    iou = intersect / union

    return iou


def skeleton_based_stdet(skeleton_config, skeleton_stdet_checkpoint, label_map, human_detections, pose_results,
                         num_frame, clip_len, frame_interval, h, w):
    predict_stepsize = 8
    action_score_thr = 0.5
    window_size = clip_len * frame_interval
    assert clip_len % 2 == 0, 'We would like to have an even clip_len'
    timestamps = np.arange(window_size // 2, num_frame + 1 - window_size // 2, predict_stepsize)

    skeleton_config = mmcv.Config.fromfile(skeleton_config)
    num_class = 2  # for AVA dataset (81)
    skeleton_config.model.cls_head.num_classes = num_class
    skeleton_pipeline = Compose(skeleton_config.test_pipeline)
    skeleton_stdet_model = build_model(skeleton_config.model)
    load_checkpoint(
        skeleton_stdet_model,
        skeleton_stdet_checkpoint,
        map_location='cpu')
    skeleton_stdet_model.to('cuda')
    skeleton_stdet_model.eval()

    skeleton_predictions = []

    print('Performing SpatioTemporal Action Detection for each clip')
    prog_bar = mmcv.ProgressBar(len(timestamps))
    for timestamp in timestamps:
        proposal = human_detections[timestamp - 1]
        if proposal.shape[0] == 0:  # no people detected
            skeleton_predictions.append(None)
            continue

        start_frame = timestamp - (clip_len // 2 - 1) * frame_interval
        frame_inds = start_frame + np.arange(0, window_size, frame_interval)
        frame_inds = list(frame_inds - 1)
        num_frame = len(frame_inds)  # 30

        pose_result = [pose_results[ind] for ind in frame_inds]

        skeleton_prediction = []
        for i in range(proposal.shape[0]):  # num_person
            skeleton_prediction.append([])

            fake_anno = dict(
                frame_dict='',
                label=-1,
                img_shape=(h, w),
                origin_shape=(h, w),
                start_index=0,
                modality='Pose',
                total_frames=num_frame)
            num_person = 1

            num_keypoint = 17
            keypoint = np.zeros(
                (num_person, num_frame, num_keypoint, 2))  # M T V 2
            keypoint_score = np.zeros(
                (num_person, num_frame, num_keypoint))  # M T V

            # pose matching
            person_bbox = proposal[i][:4]
            area = expand_bbox(person_bbox, h, w)

            for j, poses in enumerate(pose_result):  # num_frame
                max_iou = float('-inf')
                index = -1
                if len(poses) == 0:
                    continue
                for k, per_pose in enumerate(poses):
                    iou = cal_iou(per_pose['bbox'][:4], area)
                    if max_iou < iou:
                        index = k
                        max_iou = iou
                keypoint[0, j] = poses[index]['keypoints'][:, :2]
                keypoint_score[0, j] = poses[index]['keypoints'][:, 2]

            fake_anno['keypoint'] = keypoint
            fake_anno['keypoint_score'] = keypoint_score

            skeleton_imgs = skeleton_pipeline(fake_anno)['keypoint'][None]
            skeleton_imgs = skeleton_imgs.to('cuda')

            with torch.no_grad():
                output = skeleton_stdet_model(
                    return_loss=False, keypoint=skeleton_imgs)
                output = output[0]

                output = softmax(torch.Tensor(output),dim=0)
                output = np.array(output)
                for k in range(len(output)):  # 81
                    if k not in label_map:
                        continue
                    if output[k] > action_score_thr:
                        skeleton_prediction[i].append(
                            (label_map[k], output[k]))

        skeleton_predictions.append(skeleton_prediction)
        prog_bar.update()

    return timestamps, skeleton_predictions


if __name__ == '__main__': 
    mmdet_root = 'C:/Users/Admin/Desktop/MMOPEN/mmdetection'
    mmpose_root = 'C:/Users/Admin/Desktop/MMOPEN/mmpose'
    video = '2PERSON.mp4'
    det_config = f'{mmdet_root}/configs/yolox/yolox_s_8x8_300e_coco.py'
    det_checkpoint = 'yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'
    pose_config = f'{mmpose_root}/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py'
    pose_checkpoint = 'hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
    skeleton_config = 'ST_GCN.py'
    skeleton_stdet_checkpoint = 'checkpoint_94_test_100val.pth'
    label_map = 'label_stdet.txt'
    predict_stepsize = 8
    output_stepsize = 1
    frame_paths, original_frames = frame_extraction(video)
    num_frame = len(frame_paths)
    h, w, _ = original_frames[0].shape

    # Get Human detection results and pose results
    human_detections = detection_inference(det_config, det_checkpoint, frame_paths)

    pose_results = pose_inference(pose_config,pose_checkpoint, frame_paths, human_detections)

    # resize frames to shortside 256
    new_w, new_h = mmcv.rescale_size((w, h), (256, np.Inf))
    frames = [mmcv.imresize(img, (new_w, new_h)) for img in original_frames]
    w_ratio, h_ratio = new_w / w, new_h / h

    # Load spatio-temporal detection label_map
    stdet_label_map  = {0:'Not Fall', 1:'Fall'}

    clip_len, frame_interval = 48, 1

    timestamps, stdet_preds = skeleton_based_stdet(skeleton_config, skeleton_stdet_checkpoint, stdet_label_map,
                                                   human_detections,
                                                   pose_results, num_frame,
                                                   clip_len,
                                                   frame_interval, h, w)
    for i in range(len(human_detections)):
        det = human_detections[i]
        det[:, 0:4:2] *= w_ratio
        det[:, 1:4:2] *= h_ratio
        human_detections[i] = torch.from_numpy(det[:, :4]).to('cuda')

    stdet_results = []
    for timestamp, prediction in zip(timestamps, stdet_preds):
        human_detection = human_detections[timestamp - 1]
        stdet_results.append(
            pack_result(human_detection, prediction, new_h, new_w))

    def dense_timestamps(timestamps, n):
        """Make it nx frames."""
        old_frame_interval = (timestamps[1] - timestamps[0])
        start = timestamps[0] - old_frame_interval / n * (n - 1) / 2
        new_frame_inds = np.arange(
            len(timestamps) * n) * old_frame_interval / n + start
        return new_frame_inds.astype(np.int)

    dense_n = int(predict_stepsize / output_stepsize)
    output_timestamps = dense_timestamps(timestamps, dense_n)
    frames = [
        cv2.imread(frame_paths[timestamp - 1])
        for timestamp in output_timestamps
    ]

    print('Performing visualization')
    pose_model = init_pose_model(pose_config, pose_checkpoint, 'cuda')

    pose_results = [
            pose_results[timestamp - 1] for timestamp in output_timestamps
        ]

    vis_frames = visualize(frames, stdet_results, pose_results, None,
                           pose_model)
    vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_frames],
                                fps=30)
    vid.write_videofile('video_stdet_out.mp4')

    tmp_frame_dir = osp.dirname(frame_paths[0])
    shutil.rmtree(tmp_frame_dir)