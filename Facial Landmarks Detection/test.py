from helpers import get_model, get_test_loader, test_model
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

net = get_model("resnet50")
net.load_state_dict(torch.load('saved_models/keypoints_model_3_resnet50_epochs_15.pt'))
net.to(device)

batch_size = 4
test_loader = get_test_loader(batch_size)

test_model(net, test_loader, device, batch_size)

