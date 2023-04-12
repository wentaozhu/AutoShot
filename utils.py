import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os
import ffmpeg

def get_frames(fn, width=48, height=27):
    video_stream, err = (
        ffmpeg
        .input(fn)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
        .run(capture_stdout=True, capture_stderr=True)
    )
    video = np.frombuffer(video_stream, np.uint8).reshape([-1, height, width, 3])
    return video

def visualize_predictions(frames, predictions=None, predictions_2=None, predictions_3=None, show_frame_num=False):
    from PIL import Image, ImageDraw, ImageFont

    if isinstance(predictions, np.ndarray):
        predictions = [predictions]
    if isinstance(predictions_2, np.ndarray):
        predictions_2 = [predictions_2]
    if isinstance(predictions_3, np.ndarray):
        predictions_3 = [predictions_3]

    ih, iw, ic = frames.shape[1:]
    width = 25

    # pad frames so that length of the video is divisible by width
    # pad frames also by len(predictions) pixels in width in order to show predictions
    pad_with = width - len(frames) % width if len(frames) % width != 0 else 0
    frames = np.pad(frames, [(0, pad_with), (0, 1), (0, len(predictions)), (0, 0)])

    predictions = [np.pad(x, (0, pad_with)) for x in predictions]
    height = len(frames) // width

    img = frames.reshape([height, width, ih + 1, iw + len(predictions), ic])
    img_tmp = np.concatenate(np.split(
        np.concatenate(np.split(img, height), axis=2)[0], width
    ), axis=2)[0, :-1]
#     (1231, 1225, 3) 44 25 27 48
#     print(img_tmp.shape, height, width, ih, iw)

    img = Image.fromarray(img_tmp)
    draw = ImageDraw.Draw(img)
    
    if show_frame_num:
        font = ImageFont.truetype("/share/ai_platform/zhuwentao/times-ro.ttf", 12)
        # draw.text((x, y),"Sample Text",(r,g,b))
        for h in range(height):
            for w in range(width):
                avg_c = img_tmp[h * (ih + 1) + 3 : h * (ih + 1) + 9, w * (iw + 1) : w * (iw + 1)+12, :]
                avg_c = avg_c.sum()
                avg_c /= (3 * 6 * 12)
                n = h * width + w
                draw.text(
                    (
                        w * (iw + 1),
                        h * (ih + 1)+3
                    ),
                    str(n),
                    fill=(
                        255, # - img_tmp[h * (ih + 1) + 3, w * (iw + 1), 0],
                        255, # - img_tmp[h * (ih + 1) + 3, w * (iw + 1), 1],
                        255) if avg_c < 128 else (0, 0, 0), # - img_tmp[h * (ih + 1) + 3, w * (iw + 1), 2]),
                    font=font)
    
    if predictions is None:
        return img

    # iterate over all frames
    for i, pred in enumerate(zip(*predictions)):
#         print(i, pred)
        x, y = i % width, i // width
        x, y = x * (iw + len(predictions)) + iw, y * (ih + 1) + ih - 1

        # we can visualize multiple predictions per single frame
        for j, p in enumerate(pred):
            color = [0, 0, 0]
#             color[(j + 1) % 3] = 255
            color[0] = 255

            value = round(p * (ih - 1))
            if value != 0:
                draw.line((x + j, y, x + j, y - value), fill=tuple(color), width=5)
    if predictions_2 is None:
        return img
    
    # iterate over all frames
    for i, pred in enumerate(zip(*predictions_2)):
#         print(i, pred)
        x, y = i % width, i // width
        x, y = x * (iw + len(predictions)) + iw, y * (ih + 1) + ih - 1

        # we can visualize multiple predictions per single frame
        for j, p in enumerate(pred):
            color = [0, 0, 0]
#             color[(j + 1) % 3] = 255
            color[1] = 255
            if predictions[0][i] == 1:
                color[0] = 255

            value = round(p * (ih - 1))
            if value != 0:
                draw.line((x + j, y, x + j, y - value), fill=tuple(color), width=5)
    if predictions_3 is None:
        return img
    
    # iterate over all frames
    for i, pred in enumerate(zip(*predictions_3)):
        x, y = i % width, i // width
        x, y = x * (iw + len(predictions)) + iw, y * (ih + 1) + ih - 1

        # we can visualize multiple predictions per single frame
        for j, p in enumerate(pred):
            color = [0, 0, 0]
#             color[(j + 1) % 3] = 255
            color[2] = 255
            if predictions[0][i] == 1:
                color[0] = 255
            if predictions_2[0][i] == 1:
                color[1] = 255

            value = round(p[0] * (ih - 1))
            if value != 0:
                draw.line((x + j, y, x + j, y - value), fill=tuple(color), width=8)
    return img

def get_batches(frames):
    reminder = 50 - len(frames) % 50
    if reminder == 50:
        reminder = 0
    frames = np.concatenate([frames[:1]] * 25 + [frames] + [frames[-1:]] * (reminder + 25), 0)

    def func():
        for i in range(0, len(frames) - 50, 50):
            yield frames[i:i + 100]

    return func()

def scenes2zero_one_representation(scenes, n_frames):
    prev_end = 0
    one_hot = np.zeros([n_frames], np.uint64)
    many_hot = np.zeros([n_frames], np.uint64)

    for start, end in scenes:
        # number of frames in transition: start - prev_end - 1 (hardcut has 0)

        # values of many_hot_index
        # frame with index (0..n-1) is from a scene, frame [x] is a transition frame
        # [0][1] -> 0
        # [0][x][2] -> 0, 1
        # [0][x][x][3] -> 0, 1, 2
        # [0][x][x][x][4] -> 0, 1, 2, 3
        # [0][x][x][x][x][5] -> 0, 1, 2, 3, 4
        for i in range(prev_end, start):
            many_hot[i] = 1

        # values of one_hot_index
        # frame with index (0..n-1) is from a scene, frame [x] is a transition frame
        # [0]|[1] -> 0
        # [0][x]|[2] -> 1
        # [0][x]|[x][3] -> 1
        # [0][x][x]|[x][4] -> 2
        # [0][x][x]|[x][x][5] -> 2
        # ...
        if not (prev_end == 0 and start == 0):
            one_hot_index = prev_end + (start - prev_end) // 2
            one_hot[one_hot_index] = 1

        prev_end = end

    # if scene ends with transition
    if prev_end + 1 != n_frames:
        for i in range(prev_end, n_frames):
            many_hot[i] = 1

        one_hot_index = prev_end + (n_frames - prev_end) // 2
        one_hot[one_hot_index] = 1

    return one_hot, many_hot

def predictions_to_scenes(predictions):
    scenes = []
    t, t_prev, start = -1, 0, 0
    for i, t in enumerate(predictions):
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append([start, i])
        t_prev = t
    if t == 0:
        scenes.append([start, i])

    # just fix if all predictions are 1
    if len(scenes) == 0:
        return np.array([[0, len(predictions) - 1]], dtype=np.int32)

    return np.array(scenes, dtype=np.int32)


def evaluate_scenes(gt_scenes, pred_scenes, return_mistakes=False, n_frames_miss_tolerance=2):
    """
    Adapted from: https://github.com/gyglim/shot-detection-evaluation
    The original based on: http://imagelab.ing.unimore.it/imagelab/researchActivity.asp?idActivity=19

    n_frames_miss_tolerance:
        Number of frames it is possible to miss ground truth by, and still being counted as a correct detection.

    Examples of computation with different tolerance margin:
    n_frames_miss_tolerance = 0
      pred_scenes: [[0, 5], [6, 9]] -> pred_trans: [[5.5, 5.5]]
      gt_scenes:   [[0, 5], [6, 9]] -> gt_trans:   [[5.5, 5.5]] -> HIT
      gt_scenes:   [[0, 4], [5, 9]] -> gt_trans:   [[4.5, 4.5]] -> MISS
    n_frames_miss_tolerance = 1
      pred_scenes: [[0, 5], [6, 9]] -> pred_trans: [[5.0, 6.0]]
      gt_scenes:   [[0, 5], [6, 9]] -> gt_trans:   [[5.0, 6.0]] -> HIT
      gt_scenes:   [[0, 4], [5, 9]] -> gt_trans:   [[4.0, 5.0]] -> HIT
      gt_scenes:   [[0, 3], [4, 9]] -> gt_trans:   [[3.0, 4.0]] -> MISS
    n_frames_miss_tolerance = 2
      pred_scenes: [[0, 5], [6, 9]] -> pred_trans: [[4.5, 6.5]]
      gt_scenes:   [[0, 5], [6, 9]] -> gt_trans:   [[4.5, 6.5]] -> HIT
      gt_scenes:   [[0, 4], [5, 9]] -> gt_trans:   [[3.5, 5.5]] -> HIT
      gt_scenes:   [[0, 3], [4, 9]] -> gt_trans:   [[2.5, 4.5]] -> HIT
      gt_scenes:   [[0, 2], [3, 9]] -> gt_trans:   [[1.5, 3.5]] -> MISS
    """

    shift = n_frames_miss_tolerance / 2
    gt_scenes = gt_scenes.astype(np.float32) + np.array([[-0.5 + shift, 0.5 - shift]])
    pred_scenes = pred_scenes.astype(np.float32) + np.array([[-0.5 + shift, 0.5 - shift]])

    gt_trans = np.stack([gt_scenes[:-1, 1], gt_scenes[1:, 0]], 1)
    pred_trans = np.stack([pred_scenes[:-1, 1], pred_scenes[1:, 0]], 1)

    i, j = 0, 0
    tp, fp, fn = 0, 0, 0
    fp_mistakes, fn_mistakes = [], []

    while i < len(gt_trans) or j < len(pred_trans):
        if j == len(pred_trans):
            fn += 1
            fn_mistakes.append(gt_trans[i])
            i += 1
        elif i == len(gt_trans):
            fp += 1
            fp_mistakes.append(pred_trans[j])
            j += 1
        elif pred_trans[j, 1] < gt_trans[i, 0]:
            fp += 1
            fp_mistakes.append(pred_trans[j])
            j += 1
        elif pred_trans[j, 0] > gt_trans[i, 1]:
            fn += 1
            fn_mistakes.append(gt_trans[i])
            i += 1
        else:
            i += 1
            j += 1
            tp += 1

    if tp + fp != 0:
        p = tp / (tp + fp)
    else:
        p = 0

    if tp + fn != 0:
        r = tp / (tp + fn)
    else:
        r = 0

    if p + r != 0:
        f1 = (p * r * 2) / (p + r)
    else:
        f1 = 0

    assert tp + fn == len(gt_trans)
    assert tp + fp == len(pred_trans)

    if return_mistakes:
        return p, r, f1, (tp, fp, fn), fp_mistakes, fn_mistakes
    return p, r, f1, (tp, fp, fn)

def mAP_f1_p_fix_r(one_hot_pred, gt_scenes, fixed_r=0.70654, skip_map_miou=True):
    if fixed_r > 0:
        assert skip_map_miou
        eps = 0.001
        l_thr = 0.
        h_thr = 1.
        while h_thr - l_thr > eps:
            cur_thr = (l_thr + h_thr) / 2.
            precision = recall = f1 = tp = fp = fn = 0
            for file_name, pred in one_hot_pred.items():
                pred_scenes = predictions_to_scenes((pred > np.array([cur_thr])).astype(np.uint8))
                _, _, _, (tp_, fp_, fn_) = evaluate_scenes(gt_scenes[file_name], pred_scenes)
                tp += tp_
                fp += fp_
                fn += fn_

            if tp + fp == 0:
                precision = 0
            else:
                precision = tp * 1. / (tp + fp)
            if tp + fn == 0:
                recall = 0
            else:
                recall = tp * 1. / (tp + fn)

            if recall > fixed_r + eps:
                l_thr = cur_thr
            elif recall < fixed_r - eps:
                h_thr = cur_thr
            else:
                if precision + recall == 0:
                    f1 = 0
                else:
                    f1 = (precision * recall * 2) / (precision + recall)
                return 0, f1, precision, recall, cur_thr, 0
        precision = recall = f1 = tp = fp = fn = 0
        for file_name, pred in one_hot_pred.items():
            pred_scenes = predictions_to_scenes((pred > np.array([l_thr])).astype(np.uint8))
            _, _, _, (tp_, fp_, fn_) = evaluate_scenes(gt_scenes[file_name], pred_scenes)
            tp += tp_
            fp += fp_
            fn += fn_

        if tp + fp == 0:
            precision = 0
        else:
            precision = tp * 1. / (tp + fp)
        if tp + fn == 0:
            recall = 0
        else:
            recall = tp * 1. / (tp + fn)
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = (precision * recall * 2) / (precision + recall)
        return 0, f1, precision, recall, cur_thr, 0

    # f1 p r threshold
    thresholds = np.array([0.02, 0.06, 0.1, 0.15, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.255, 0.26, 0.265, 0.27, 0.275, 0.28, 0.2833, 0.2867, 0.29, 0.292, 0.294, 0.296, 0.298, 0.3, 0.302, 0.304, 0.306, 0.308, 0.31, 0.3133, 0.3167, 0.32, 0.325, 0.33, 0.335, 0.34, 0.345, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.5, 0.6, 0.7, 0.8,
                           0.9])
#     thresholds = np.array([0.02, 0.06, 0.1, 0.15, 0.2, 0.294, 0.2945, 0.295, 0.2952, 0.2954, 0.2956, 0.2958, 0.296, 0.2962, 0.2964, 0.2966, 0.2968, 0.297, 0.2975, 0.298, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
#                            0.9])
#     thresholds = np.array([0.02, 0.06, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
#                            0.9])
    precision, recall, f1, tp, fp, fn = np.zeros_like(thresholds), np.zeros_like(thresholds), \
                                        np.zeros_like(thresholds), np.zeros_like(thresholds), \
                                        np.zeros_like(thresholds), np.zeros_like(thresholds)
    for i in range(len(thresholds)):
        for file_name, pred in one_hot_pred.items():
            pred_scenes = predictions_to_scenes((pred > thresholds[i]).astype(np.uint8))
            _, _, _, (tp_, fp_, fn_) = evaluate_scenes(gt_scenes[file_name], pred_scenes)
            tp[i] += tp_
            fp[i] += fp_
            fn[i] += fn_

        if tp[i] + fp[i] == 0:
            precision[i] = 0
        else:
            precision[i] = tp[i] * 1. / (tp[i] + fp[i])
        if tp[i] + fn[i] == 0:
            recall[i] = 0
        else:
            recall[i] = tp[i] * 1. / (tp[i] + fn[i])
        if precision[i] + recall[i] == 0:
            f1[i] = 0
        else:
            f1[i] = (precision[i] * recall[i] * 2) / (precision[i] + recall[i])

    best_idx = np.argmax(f1)

    if skip_map_miou:
        return 0, f1[best_idx], precision[best_idx], recall[best_idx], thresholds[best_idx], 0

    # mAP
    # mIOU
    mious = []
    y_true_scene_list, y_pred_scene_list = [], []
    for file_name, pred in one_hot_pred.items():
        if len(pred) == 0:
            continue
        pred_scenes = predictions_to_scenes((one_hot_pred[file_name] > thresholds[best_idx]).astype(np.uint8))
        y_true_scene, y_pred_scene = evaluate_scenes_mAP(gt_scenes[file_name], pred_scenes, pred)
        for y_true, y_pred in zip(y_true_scene, y_pred_scene):
            y_true_scene_list.append(y_true)
            y_pred_scene_list.append(y_pred)

        mious.append(np.mean([
            cal_miou(gt_scenes[file_name], pred_scenes),
            cal_miou(pred_scenes, gt_scenes[file_name])
        ]))

    mAP = average_precision_score(y_true_scene_list, y_pred_scene_list)
    if np.isnan(mAP):
        mAP = 0
    return mAP, f1[best_idx], precision[best_idx], recall[best_idx], thresholds[best_idx], np.mean(mious)
