import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from attacks import *
from models import LeNet
from torchvision import datasets, transforms
from utils import model_eval, model_train

epsilons = [0, .05, .1, .15, .2, .25, .3]
use_cuda = torch.cuda.is_available()
pretrained_model = 'resources/lenet_mnist_model.bin'

# MNIST Test dataset and dataloader declaration
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
    ])),
    batch_size=1, shuffle=True)

# Initialize the network
model = LeNet().cuda() if use_cuda else LeNet()

# Load the pretrained model
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()

accuracies = []
# Run test for each epsilon
for eps in epsilons:
    acc = model_eval(model, test_loader, F.nll_loss, use_cuda, attack_method=fgsm_attack, epsilon=eps)
    print("Epsilon: {}\tTest Accuracy = {}".format(eps, acc))
    accuracies.append(acc)

plt.figure(figsize=(5,5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.show()