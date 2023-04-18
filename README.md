# AutoShot
AutoShot: A Short Video Dataset and State-of-the-Art Shot Boundary Detection - CVPR NAS 2023

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/autoshot-a-short-video-dataset-and-state-of-1/camera-shot-boundary-detection-on-clipshots)](https://paperswithcode.com/sota/camera-shot-boundary-detection-on-clipshots?p=autoshot-a-short-video-dataset-and-state-of-1)

The short-form videos have explosive popularity and have dominated the new social media trends. Prevailing short-video platforms, e.g., Kuaishou (Kwai), TikTok, Instagram Reels, and YouTube Shorts, have changed the way we consume and create content. For video content creation and understanding, the shot boundary detection (SBD) is one of the most essential components in various scenarios. In this work, we release a new public Short video sHot bOundary deTection dataset, named SHOT, consisting of 853 complete short videos and 11,606 shot annotations, with 2,716 high quality shot boundary annotations in 200 test videos. Leveraging this new data wealth, we propose to optimize the model design for video SBD, by conducting neural architecture search in a search space encapsulating various advanced 3D ConvNets and Transformers. Our proposed approach, named AutoShot, achieves higher F1 scores than previous state-of-the-art approaches, e.g., outperforming TransNetV2 by 4.2\%, when being derived and evaluated on our newly constructed SHOT dataset. Moreover, to validate the generalizability of the AutoShot architecture, we directly evaluate it on another three public datasets: ClipShots, BBC and RAI, and the F1 scores of AutoShot outperform previous state-of-the-art approaches by 1.1\%, 0.9\% and 1.2\%, respectively.

Dataset link:<br>
Label: https://github.com/wentaozhu/AutoShot/blob/main/kuaishou_v2.txt <br>
Please read the paper https://github.com/wentaozhu/AutoShot/blob/main/CVPR23_AutoShot.pdf and supplementary https://github.com/wentaozhu/AutoShot/blob/main/CVPR23_AutoShot_Supplementary.pdf for the dataset, baseline and method. <br>
Format: In the dataset, for each short video, with the same name as the video, there exists the annotations of the shot boundaries, i.e., the beginning and the end frame number of each shot per line. <br>
Data: 
Google Drive: https://drive.google.com/drive/folders/1xZN6tvefXXmpZlIZ6GoSUUxpDQQOSNfJ?usp=sharing (Thank you, Dr. Haotian Jiang, so much!) <br>
Baidu link: https://pan.baidu.com/s/1CdCVNzFdF3U6I4ajfejYNQ?pwd=sfkq passcode: sfkq <br>



Model link: Baidu link: https://pan.baidu.com/s/1CdCVNzFdF3U6I4ajfejYNQ?pwd=sfkq passcode: sfkq <br>

Evaluation script:<br>
Step 1: Download the dataset using link (https://pan.baidu.com/s/1CdCVNzFdF3U6I4ajfejYNQ?pwd=sfkq passcode: sfkq), unzip. <br>
Step 2: Merge folders `ads_game_videos_2` and `ads_game_videos` into one folder with name `ads_game_videos`. <br>
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
