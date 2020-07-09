from helpers import get_model, get_train_loader, train_net, net_sample_output, visualize_output
from torch import nn, optim, cuda, save, load

device = "cuda" if cuda.is_available() else "cpu"

net = get_model("resnet50")
net.load_state_dict(load('saved_models/keypoints_model_3_resnet50_epochs_15.pt'))
net.to(device)

batch_size = 4
train_loader = get_train_loader(batch_size)

images, predictions, gt_pts = net_sample_output(net, train_loader, device)
visualize_output(images, predictions, gt_pts, batch_size)

criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(net.parameters())

n_epochs = 5
cuda.empty_cache()
train_net(net, criterion, optimizer, train_loader, n_epochs, device)

# After training, save your model parameters in the dir 'saved_models'
model_dir = 'saved_models/'
model_name = 'keypoints_model_3_resnet50_epochs_' + str(n_epochs) + '.pt'
save(net.state_dict(), model_dir+model_name)

