# Object Tracking

## Introduction
Object tracking is an essential task in all the applications concerning to *video analytics*. One has to track one particular or multiple objects in a video through time. Object of interest can be anything depending on the use case, like for security related scenarios, each person in video must be tracked while logging their facial data along with other crucial information. Another scenario is detecting traffic jams in cctv videos, in this case each vehicle must be tracked through time. Also counting total number of vehicles in a frame at a particular time, such type of information is crucial to detect traffic jams. Applications are endless for video analytics. In all of them, one has to track object/objects autonomously.

## Core Idea

* Task of tracking comes after all the objects are detected in a frame. Hence object detection is the first step. For the object detection, here I have chosen yolov4-tiny model. Which can run in real time on most of the GPUs.
* Now once all the objects are detected we have to track them in the next frame. So how? One can observe a tiny detail that *an object doesn't move significantly bw 2 consecutive frames* (most videos are recorded at *>=24 fps*).
* Now in the next frame, first we will do object detection to get the bounding boxes. So now we have bboxes from previous frame and current frame. Now we can make an association bw them, based on previously observerd fact.
* How to make such association? Like I said previously, since objects aren't moving significantly, we can calculate a IOU (intersection over union) table. In this table, iou[i, j] referes to IOU bw ith detection's bbox from current frame and jth detection's bbox from previous frame. Now this table can aid us in making associations.
* After calculating this table, we will make each entry 0, if it's lesser than *iou_threshold* else we will leave it as it is. *iou_threshold* is a hyperparameter chosen by us. Point of this step is, how we can tell if ith box (current frame) and jth box (previous frame) referes to the same object, those boxes have to overlap to some extent to tell that they point to the same object. So this minimum required overlap is our parameter called *iou_threshold*. Here I have set it to 0.3, you can change it depending on your needs.
* Now we can associate jth box in previous frame to ith box in current frame, if iou[i, j] is maximum from iou[i, :] (while also checking if it's > 0), else we will consider ith box in current frame as a new object.
* **Caveat:** What if our detector fails (due to some reason) to detect an object(s) in the current frame, which was detected in previous frame? One reason could be our object might have moved out from the scene, so it's not visible in camera, or it could be that our classifier failed to detect that in the first place. If its first case that means we have tracked that object throught the end :). It won't get detected in some of the upcoming frames. Now considering the second case, object from previous frame might get detected in upcoming frames, though it wasn't detected in current frame. Solution is to make track of all the tracking data along with a parameter called age. We will increment age when such case happens. We will delete all such tracked boxes which has age > *max_age* (Another hyperparameter, controlled by us, here I have set it to 20, change according to your needs).

## Pros

* It's very fast (I have written vectorized code for calculating iou_table :) . Only bottleneck is object detection. Since it's fast, it can be used in real time tracking scenarios.

## Cons
* In case of *occlusion*, this algorithm will clearly *fail*, by assigning false id to the visible object. One workaround is to consider convolution score (distance bw feature vectors of the objects) and shape along with iou_score, while making associations. Downside to this workaround is *feature vector calculation step* for each detected objects, it can slow down our algo, and hence now it can't be used in real time applications. Upside is, it will be very accurate, since we are taking many factors into the considerations.

* In case objects are moving fast in the video or video fps is low. We might fail to make associations bw frames. Above workaround will work in this case too. Or we can tune our hyperparameters to get the best out of it.

* Not so *accurate*.


## Conclusion
* Task of tracking relies heavily upon object detection. Detector might misclassify an object to different category / fail clearly to detect an object (false negative) / detect an object while there isn't any (false positive). All of such cases has to be handled while designing an intelligent video analytics system.
* Runtime of object detection step is also crucial when it comes to real time applications. Though after yolov4 introduced, object detectors now has become fast, and will be improved further in the future too. **Hats off to the authors of Yolo_v4 paper.**

## Usage & Results
### Prerequisites

* Clone https://github.com/Tianxiaomo/pytorch-YOLOv4.
* Go inside this cloned repo. Make 2 dirs named "Outputs" and "preds".
* Copy checkpoints/yolov4-tiny.weights from this repo to the {cloned repo}/checkpoints/yolov4-tiny.weights. You can download other yolov4 models from https://github.com/AlexeyAB/darknet, put it's cfg file and weights file inside {cloned repo}/cfg/ and {cloned repo}/checkpoints/ dirs respectively.
* Replace demo .py (From this repo) with {cloned repo}/demo.py
* Replace utils .py (From this repo) with {cloned repo}/tool/utils.py
* Put input videos you want to process inside data/ dir.

---
### Available command line args
* cfgfile (*required*): yolo config file path
* weightfile (*required*): yolo weights file path
* videofile (*required*): Input video file path
* classname (*required*): Class which you want to track e.g. car, person, etc. See list of available classes [here](./data/coco.names).
* outpath (*optinal*): Path on which to save processed video file
* savename (*optional*): For *debugging*, add this flag -savename preds/f while executing the code. This will save result of each frame inside preds dir. Listed as f0.jpg, f1.jpg, f2.jpg and so on.

### Hyperparameters
* iou_threshold (float): Read core idea part
* max_age (int): Read core idea part
* use_cuda (boolean): If you want to do detection on GPU, set it to *true* else *false*.
* width & height: Input image width and height, which is passed to the object detector. See available values at https://github.com/Tianxiaomo/pytorch-YOLOv4 inference section. To change edit width and height param in respective cfg file inside cfg/ dir. 
---

**Note**: Don't forget to give correct classname (One which you want to track).

```bash
foo@bar:~$ python demo.py -cfgfile ./cfg/yolov4-tiny.cfg -weightfile ./checkpoints/yolov4-tiny.weights -videofile ./data/sample2.mp4 -classname car -outpath Outputs/s2_yolov4_tiny.avi
```

I am able to get **~20 fps** with yolov4-tiny (width=512 and height=512) on my nvidia Geforce 940MX (*Entry level GPU, Only 2 GB DDR3 VRAM and 5.0 compute capability*), which is great :)

Checkout the results below.


![](Outputs/s2_yolov4_tiny.gif)

---
#### Misc.

Compressing output file and converting it to a gif using ffmpeg

```bash
foo@bar:~$ ffmpeg -i s2_yolov4_tiny.avi -vcodec libx264 s2_yolov4_tiny.mp4
foo@bar:~$ ffmpeg -ss 35 -t 20 -i s2_yolov4_tiny.mp4 -vf "fps=10,scale=720:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 s2_yolov4_tiny.gif
```
---

```bash
foo@bar:~$ python demo.py -cfgfile ./cfg/yolov4-tiny.cfg -weightfile ./checkpoints/yolov4-tiny.weights -videofile ./data/sample3.mp4 -classname person -outpath Outputs/s3_yolov4_tiny.avi
```
![](Outputs/s3_yolov4_tiny.gif)

---
#### Misc.

Compressing output file and converting it to a gif using ffmpeg

```bash
foo@bar:~$ ffmpeg -i s3_yolov4_tiny.avi -vcodec libx264 s3_yolov4_tiny.mp4
foo@bar:~$ ffmpeg -i s3_yolov4_tiny.mp4 -vf "fps=10,scale=720:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 s3_yolov4_tiny.gif
```
---

As you can observe in the results, there's some flickering due to object detector's failure to detect the object but yet our tracking algorithm is smart enough to identify that same object in upcoming frames.

You can always *tune hyperparametrs* to get the best out of it for your own input videos.

**Note**: You can find these input videos online by googling :)