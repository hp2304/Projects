# Feature Engineering for Person Identification given Dentition Images

## Objective

Given a dataset of dentition images of various individuals, extract features from them to uniquely identify a person using only Image Processing techniques. Kindly refer to the [report](./feature_extraction_dentition_images.pdf) for more details.

## Overview

![](./block_diagram.drawio.png)

1. Input Image

    ![](./results/3_5.png)

2. Gamma correction

    ![](./results/3_5_28_gamma.jpg)

3. Blurring

    ![](./results/3_5_28_blur.jpg)

4. Left

    ![](./results/3_5_28_left_binary.jpg)
    ![](./results/3_5_28_left_cntr.jpg)

    Incisor

    ![](./results/3_5_28_li_tooth_id.jpg)

    Canine

    ![](./results/3_5_28_lc_tooth_id.jpg)

5. Right

    ![](./results/3_5_28_right_binary.jpg)
    ![](./results/3_5_28_right_cntr.jpg)

    Incisor

    ![](./results/3_5_28_ri_tooth_id.jpg)

    Canine

    ![](./results/3_5_28_rc_tooth_id.jpg)

6. Incisor and Canine detection

    ![](./results/3_5_28_centroids.jpg)

7. Feature Calculation

    - Angular Features

        ![](./results/3_5_28_angle_0.jpg)
        ![](./results/3_5_28_angle_1.jpg)
        ![](./results/3_5_28_angle_2.jpg)
        ![](./results/3_5_28_angle_3.jpg)
        ![](./results/3_5_28_angle_4.jpg)
        ![](./results/3_5_28_angle_5.jpg)

    - Distance related features

        ![](./results/3_5_28_dists.jpg)
