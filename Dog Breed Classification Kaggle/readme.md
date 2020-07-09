Code for once held [kaggle competition for dog breed identification](https://www.kaggle.com/c/dog-breed-identification). Check out metadata of dataset and problem statement on given link.

* I have used **PyTorch** framework for this task.
* I have also implemented various augmentation techniques.
* Transfer learning was utilized to head start training process (Training was done on GPU).
* I tried many variants of VGG, ResNet and DenseNet (Pretrained) to achieve good performance.
* Categorical cross entropy loss and Adam optimizer seemed obvious choice for this classification.
* I scored around 2.9 (which is less) on test data. Model was trained for only 50 epochs though. Deep models takes really long time to train, even on GPU.