# Problem statement
This is just a fun side project. Task is to predict 68 facial landmarks given an RGB face image. Actually I found this problem statement from a [udacity's github repo](https://github.com/udacity/P1_Facial_Keypoints). This repo contains all the needed (helper notebooks) things to complete this project, it doesn't contain solution about the problem. You need to clone that repo and take a look at readme file there to understand problem definition and also to download the data. *It also contains helper notebooks, you can read it and finish it yourself by editing them* (of course you need to do some research to solve it). *Hats off to Udacity for making such repo open source !!*

#### Required Packages
* PyTorch and Torchvision
* Numpy
* Cv2 (Opencv stuff)
* Matplotlib
* Pandas

Copy *data* (from cloned udacity github repo) directory here.

*I have copied following classes from their notebook,*

* FacialKeypointsDataset (for loading data)
* Rescale (Resize input data to feed through model)
* Randomcrop (Data augmentation)
* Normalize (Normalize input data, meaning dividing pixel values by 255)

### Pipeline
* In *data* directory, *training_frames_keypoints.csv* contains filenames of training images (contained in *data/training*) and 136 values corresponding to (x, y) pairs of 68 facial landmarks. Equivalently there's testing data. Dataset class is there to load these data (Image and 68 landmark  points) given a input index.
* Classes like *ColorJitter (Change brightness, saturation and contrast randomly), Rescale, Random Crop, Normalize and ToTensorAndPreProcess (Mean normalization step and then Convert to torch tensor)* applies some necessary data augmentation and data transforms to our loaded input data from abve step. Augmentation to get good model (less overfitting and also to use deeper CNNs). And transforms to make our input augmented data compatible to flow through the model (Resize the image to 224x224 and mean normalize, etc.). You can add other data augmentation techniques like rotation, translation, etc. in *get_data_transform()* function of *helpers* python file.
* Creating data loaders to shuffle and batch input data and feed it to our model.
* I have tried two *pretrained models resnet18 and resnet50*. I added last linear layer to ouput 136 values corresponding to our 68 facial landmark points. You can edit *get_model()* in *helpers* to choose any model of your liking, pretrained deep CNN or custom CNN designed by yourself.
* Clearly we can understand that, this is a **Regression** problem. I utilized a combination of L1 loss (avg. mean absolute error) and L2 loss (avg. mean squared error) function called [smooth L1 loss](https://pytorch.org/docs/stable/nn.html#smoothl1loss). Also *Adam* optimizer was utilized for converging to minima at faster rate.
* I trained these models for 15 epochs with batch size of 4. You can edit *train .py* to change these values.

---
#### Note
* I chose to use shallower CNNs (And which can fit in < 2 GB of VRAM) with batch size of 4 and trained for only 15 epochs, because I have only 2 GB of VRAM in my laptop (Nvidia Geforce 940MX). You can try other VGG variants, Resnet variants with larger batch size and train for more number of epochs to get *superior* performance on test data.
* Since I am training it on GPU, in case you need to run it on CPU, you need to make some minor changes in *helpers .py*
---
* Running train .py will store trained model in *saved_models/* directory. Don't forget to change model file name in train .py
* Equivalently above steps are done for testing data. Except augmentation part.
* Run test .py , Don't forget to change filename of saved model (and model architecture) in it. It will display prediction results on a random test batch data.
* Try other VGG variants, Resnet variants with larger batch size and train for more number of epochs to get *superior* performance on test data :)