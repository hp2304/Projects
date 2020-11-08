# Semantic Segmentation of Aerial Drone Images

* As title suggests, semantic segmentation is performed on drone images (of size *4000x6000*). Dataset is available on [kaggle](https://www.kaggle.com/bulentsiyah/semantic-drone-dataset). Take a look at dataset link for more info (21 classes total, including background).

* Actually there are total 23 classes, mask images contains 23 unique values starting from 0 to 22. Among them I have chosen top 5 classes (excluding background) to perform semantic segmentation, these top 5 classes constitutes aprrox. 80% pixels of the whole dataset. Remaining 18 classes were made to belong to background class. Hence, output segmentation map has total 6 classes (including background).

* There are total 400 images, I have split it in 80:10:10, train:val:test set. 

* I have finetuned *deeplab_v3_resnet101* network on this dataset. Model was finetuned (froze some of the initial layers) for **18 epochs** (check notebook). After that it was evaluated on test data, it turned out to be .8392 average *dice score* (meaning **83.92% test accuracy**). Which is good enough given only 320 images to train on. Various types of *data augmentation* techniques were also applied to reduce overfitting. A combination of **weighted cross entropy loss** (due to *class imbalance issue*) and **dice loss** was used as a loss function, which I think was game changer for achieving such great results.

* Checkout **aerial-drone-imagery-segmentation.ipynb** for more details, I have commented the code. Trained model is [here](https://drive.google.com/file/d/1m7tgh6Egmh_gwPpHKseGpX7ItibpwkPI/view?usp=sharing). Use torch.load(<file_path>) to load the model.