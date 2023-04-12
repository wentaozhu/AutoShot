# AutoShot
AutoShot: A Short Video Dataset and State-of-the-Art Shot Boundary Detection - CVPR NAS 2023

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/autoshot-a-short-video-dataset-and-state-of/camera-shot-boundary-detection-on-clipshots)](https://paperswithcode.com/sota/camera-shot-boundary-detection-on-clipshots?p=autoshot-a-short-video-dataset-and-state-of)

Dataset link:<br>
Label: https://github.com/wentaozhu/AutoShot/blob/main/kuaishou_v2.txt <br>
Please read the paper https://github.com/wentaozhu/AutoShot/blob/main/CVPR23_AutoShot.pdf and supplementary https://github.com/wentaozhu/AutoShot/blob/main/CVPR23_AutoShot_Supplementary.pdf for the dataset, baseline and method. <br>
Format: In the dataset, for each short video, with the same name as the video, there exists the annotations of the shot boundaries, i.e., the beginning and the end frame number of each shot per line. <br>
Data: Baidu link: https://pan.baidu.com/s/1CdCVNzFdF3U6I4ajfejYNQ?pwd=sfkq passcode: sfkq <br>



Model link: TODO

Evaluation script:<br>
Step 1: Download the dataset using link (https://pan.baidu.com/s/1CdCVNzFdF3U6I4ajfejYNQ?pwd=sfkq passcode: sfkq), unzip. <br>
Step 2: Merge folders `ads_game_videos_2` and `ads_game_videos` into one folder with name `ads_game_videos`. <br>
Step 3: Merge folders `video_download`, `video_download_2`, `video_download_3`, `video_download_4`, `video_download_5` into one folder with name `video_download`. <br>
Step 3: Merge folders `video_download`, `video_download_2`, `video_download_3`, `video_download_4`, `video_download_5` into one folder with name `video_download`. <br>
Step 4: Change the data path `YOURDOWNLOADDATAPATH` in `compare_inference_baseline_groundtruth_v2.py` with your downloaded data path. <br>
Step 5: Download baseline model using link (https://pan.baidu.com/s/1CdCVNzFdF3U6I4ajfejYNQ?pwd=sfkq passcode: sfkq), and move the model `ckpt_0_200_0.pth` to this directory. <br>
Step 6: Download baseline existing prediction using link (https://pan.baidu.com/s/1CdCVNzFdF3U6I4ajfejYNQ?pwd=sfkq passcode: sfkq), and move the prediction `baseline_one_hot_pred_dict_baseline.pickle` and `gt_scenes_dict_baseline_v2.pickle` to this directory. <br>
Step 7: Run inference and evaluation on the AutoShot test set using `python compare_inference_baseline_groundtruth_v2.py` <br>

In the evaluation script `compare_inference_baseline_groundtruth_v2.py`, you can uncomment the inference part of the code, and conduct the real inference.

Please let me know if there is any issues. Thank you so much!

Contact: wentao.zhu16@gmail.com

If you find this is helpful to you, please add our work into your reference.

<pre><code>@inproceedings{zhuautoshot,
  title={AutoShot: A Short Video Dataset and State-of-the-Art Shot Boundary Detection},
  author={Zhu, Wentao and Huang, Yufang and Xie, Xiufeng and Liu, Wenxian and Deng, Jincan and Zhang, Debing and Wang, Zhangyang and Liu, Ji},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year={2023}
}</code></pre>
