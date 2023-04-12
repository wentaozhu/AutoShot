# AutoShot
AutoShot: A Short Video Dataset and State-of-the-Art Shot Boundary Detection - CVPR NAS 2023

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/autoshot-a-short-video-dataset-and-state-of/camera-shot-boundary-detection-on-clipshots)](https://paperswithcode.com/sota/camera-shot-boundary-detection-on-clipshots?p=autoshot-a-short-video-dataset-and-state-of)

Dataset link:<br>
Label: https://github.com/wentaozhu/AutoShot/blob/main/kuaishou_v2.txt <br>
Please read the paper https://github.com/wentaozhu/AutoShot/blob/main/CVPR23_AutoShot.pdf and supplementary https://github.com/wentaozhu/AutoShot/blob/main/CVPR23_AutoShot_Supplementary.pdf for the dataset, baseline and method. <br>
Format: In the dataset, for each short video, with the same name as the video, there exists the annotations of the shot boundaries, i.e., the beginning and the end frame number of each shot per line. <br>
Data:


Model link: TODO

Evaluation script:<br>
Step 1: Download the dataset using link, unzip. <br>
Step 2: Merge folders `ads_game_videos_2` and `ads_game_videos` into one folder with name `ads_game_videos`. <br>
Step 3: Merge folders `video_download`, `video_download_2`, `video_download_3`, `video_download_4`, `video_download_5` into one folder with name `video_download`. <br>
Step 4: Change the data path `YOURDOWNLOADDATAPATH` in `compare_inference_baseline_groundtruth_v2.py` with your downloaded data path. <br>
Step 5: Download baseline model using link, and move the model to this directory. <br>
Step 6: Download baseline existing prediction using link, and move the prediction to this directory. <br>
Step 7: Run inference and evaluation on the AutoShot test set using `python compare_inference_baseline_groundtruth_v2.py` <br>

In the evaluation script `compare_inference_baseline_groundtruth_v2.py`, you can uncomment the inference part of the code, and conduct the real inference.

Please let me know if there is any issues. Thank you so much!

Contact: wentao.zhu16@gmail.com

If you find this is helpful to you, please add our work into your reference.

<pre><code>@inproceedings{zhuautoshot,
  title={AutoShot: A Short Video Dataset and State-of-the-Art Shot Boundary Detection},
  author={Zhu, Wentao and Xie, Xiufeng and Liu, Wenxian and Deng, Jincan and Zhang, Debing and Wang, Zhangyang and Liu, Ji},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year={2023}
}</code></pre>
