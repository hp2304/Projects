Code for once held [kaggle competition for dog breed identification](https://www.kaggle.com/c/dog-breed-identification). Check out metadata of dataset and problem statement on given link.

* I have used **PyTorch** framework for this task.
* I have also implemented data augmentation pipeline.
* Transfer learning was utilized to head start training process (Training was done on GPU).
* I tried various models (including some of their variants) like ResNet, ResNext and DenseNet (Pretrained) to achieve good performance in shorter time.
* Weighted categorical cross entropy loss and Adam optimizer seemed good choice for this. LR scheduler was also utilized to decrease lr when *val_loss* was not improving.
* I scored (private score) around 1.12110 (which is good enough given the conditions) on test data, currently it ranks at 839th position from all public submissions. Model was finetuned for 45 epochs. This result was achieved using [resnext50_32x4d](https://pytorch.org/docs/stable/torchvision/models.html) model.