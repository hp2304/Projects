import torch.nn as nn
from torchvision import models, transforms

class DNN(nn.Module):
    def __init__(self, use_pretrained = False, nb_outs = 1):
        super(DNN, self).__init__()
        self.net = models.mobilenet_v2(pretrained = use_pretrained, progress=True)
        self.net.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(1280, 1, bias=False))
        
    def forward(self, x):
        x = self.net(x)
        return x

def get_test_loader():
    mean_nums = [0.485, 0.456, 0.406]
    std_nums = [0.229, 0.224, 0.225]
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean_nums, std_nums)
    ])
    return test_transform