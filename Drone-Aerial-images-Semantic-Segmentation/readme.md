# Semantic Segmentation of Aerial Drone Images

* As title suggests, semantic segmentation is performed on drone images (of size *4000x6000*). Dataset is available on [kaggle](https://www.kaggle.com/bulentsiyah/semantic-drone-dataset). Take a look at dataset link for more info (21 classes total, including background).

* Actually there are 23 classes, mask images contains 23 unique values starting from 0 to 22. There are total 400 images, I have split it in 80:10:10, train:val:test set. 

* I have finetuned deeplab_v3_resnet101 network on this dataset. Model was finetuned for 30 epochs (check notebook). After that it was evaluated on test data, it turned out to be .6711 average *dice score* (meaning 67.11 % test accuracy). Which is good enough given only 400 images and 23 classes.

* Checkout **aerial-drone-imagery-segmentation.ipynb** for more details, I have commented the code. Trained model is [here](https://drive.google.com/file/d/1EMANq7n2liJmf7FU59e4LZne9c0wNKo4/view?usp=sharing) (PyTorch).