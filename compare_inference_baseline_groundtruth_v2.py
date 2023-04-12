import os
import pickle
import numpy as np
from utils import mAP_f1_p_fix_r
from utils import evaluate_scenes, predictions_to_scenes
from utils import get_frames, get_batches, scenes2zero_one_representation, visualize_predictions
import ffmpeg
# PT EVALUATION
import os
import pickle

import numpy as np
import torch
from transnetv2 import TransNetV2
### install and configure transnetv2 pretrained model https://github.com/soCzech/TransNetV2/tree/master/inference


if __name__ == "__main__":
    # load data & build fnm - path dict
    fnm_path_dict = {}
    dir_list = [
        "/share/ai_platform/lushun/dataset/original_videos/",
        "/share/ai_platform/zhuwentao/ads_game_videos/",
        "/share/ai_platform/zhuwentao/video_download/"
    ]
    for cur_dir in dir_list:
        for fnm in os.listdir(cur_dir):
            if fnm.endswith(".mp4"):
                fnm_path_dict[fnm[:-len(".mp4")]] = cur_dir + fnm

    # load test annotation keep only one shot
    with open('./gt_scenes_dict_baseline_v2.pickle', 'rb') as handle:
        gt_scenes_dict = pickle.load(handle)
    handle.close()
    
    print(sum( [len(annot) for _, annot in gt_scenes_dict.items()]))

    # load network
    # location of learned weights is automatically inferred
    # add argument model_dir="/path/to/transnetv2-weights/" to TransNetV2() if it fails
#     baseline_model = TransNetV2()

#     baseline_one_hot_pred_dict = {}
#     i = 0
#     for fnm, annot in gt_scenes_dict.items():
#         i += 1
#         print(i, fnm)
        
#         video_frames, single_frame_predictions, all_frame_predictions = baseline_model.predict_video(fnm_path_dict[fnm])

#         baseline_one_hot_pred_dict[fnm] = (single_frame_predictions > 0.5).astype(np.uint8)

#     with open('./baseline_one_hot_pred_dict_baseline.pickle', 'wb') as handle:
#         pickle.dump(baseline_one_hot_pred_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     handle.close()

    with open('./baseline_one_hot_pred_dict_baseline.pickle', 'rb') as handle:
        baseline_one_hot_pred_dict = pickle.load(handle)
    handle.close()
    
    print(gt_scenes_dict["20915607770"])

    mAP, metric_F1, precision, recall, threshold, miou = mAP_f1_p_fix_r(baseline_one_hot_pred_dict, gt_scenes_dict)
    print("Baseline 0.5", metric_F1, precision, recall, threshold)
    
#     Baseline 0.5 0.7992903082723443 0.9041645760160562 0.7162162162162162 0.9990234375


    # max F1
#     from supernet_flattransf_3_8_8_8_13_12_0_16_60 import TransNetV2Supernet
#     supernet_best_f1 = TransNetV2Supernet().eval()

#     if torch.cuda.is_available() is True:
#         device = "cuda"
#     else:
#         device = "cpu"
    
#     pretrained_path = os.path.join("./ckpt_0_200_0.pth")
#     if os.path.exists(pretrained_path):
#         print('Loading pretrained_path from %s' % pretrained_path)
#         model_dict = supernet_best_f1.state_dict()
#         pretrained_dict = torch.load(pretrained_path, map_location=device)
# #         key = ('net',)
# #         for k, v in pretrained_dict.items():
# #             print(k, type(v))
#         pretrained_dict = {k: v for k, v in pretrained_dict['net'].items() if k in model_dict}
#         print("Current model has %d paras, Update paras %d " % (len(model_dict), len(pretrained_dict)))
#         model_dict.update(pretrained_dict)
#         supernet_best_f1.load_state_dict(model_dict)
#     else:
#         raise Exception("Error: Can NOT find pretrained best model!!")

#     if device == "cuda":
#         supernet_best_f1 = supernet_best_f1.cuda(0)
#     supernet_best_f1.eval()

#     # Evaluation
#     def predict(batch):
#         batch = torch.from_numpy(batch.transpose((3, 0, 1, 2))[np.newaxis, ...]) * 1.0
#         batch = batch.to(device)
#         one_hot = supernet_best_f1(batch)
#         if isinstance(one_hot, tuple):
#             one_hot = one_hot[0]
#         return torch.sigmoid(one_hot[0])

#     supernet_best_f1_one_hot_pred_dict = {}
#     i = 0
#     for fnm, annot in gt_scenes_dict.items():
#         i += 1
#         print(i, fnm)
        
#         predictions = []
#         frames = get_frames(fnm_path_dict[fnm])

#         for batch in get_batches(frames):
#             one_hot = predict(batch)
#             one_hot = one_hot.detach().cpu().numpy()
            
#             predictions.append(one_hot[25:75])

#         predictions = np.concatenate(predictions, 0)[:len(frames)]
#         supernet_best_f1_one_hot_pred_dict[fnm] = predictions
    
#     with open('./supernet_best_f1.pickle', 'wb') as handle:
#         pickle.dump(supernet_best_f1_one_hot_pred_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     handle.close()

    with open('./supernet_best_f1.pickle', 'rb') as handle:
        supernet_best_f1_one_hot_pred_dict = pickle.load(handle)
    handle.close()

    mAP, metric_F1, precision, recall, threshold, miou = mAP_f1_p_fix_r(supernet_best_f1_one_hot_pred_dict, gt_scenes_dict, fixed_r=-1)
    print("supernet_best_f1", metric_F1, precision, recall, threshold)
#     supernet_best_f1 0.8405448717948718 0.8473344103392568 0.8338632750397457 0.296

    for fnm, annot in gt_scenes_dict.items():
        frames = get_frames(fnm_path_dict[fnm])
        img = visualize_predictions(
            frames,
            predictions=scenes2zero_one_representation(annot.astype(np.int), frames.shape[0]),
            predictions_2=baseline_one_hot_pred_dict[fnm],
            predictions_3=(supernet_best_f1_one_hot_pred_dict[fnm]>0.296).astype(np.uint8),
            show_frame_num=True
        )
        if not os.path.exists("./im_video_annotation_supernet_best_f1/"):
            os.mkdir("./im_video_annotation_supernet_best_f1/")
        im = img.save("./im_video_annotation_supernet_best_f1/" + fnm + ".png")
