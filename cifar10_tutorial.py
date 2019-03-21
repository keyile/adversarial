import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from attacks import *
from models import VGG
from utils import model_eval, model_train


def main():
    # check the configurations
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # prepare data for training
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, shuffle=True,
                                              batch_size=128, num_workers=4, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, shuffle=False,
                                             batch_size=128, num_workers=4, pin_memory=True)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # initilizae the model 
    net = VGG().cuda() if use_cuda else VGG()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)

    # training or loading the neural network
    # model_train(net, trainloader, criterion, optimizer, epochs=5)
    net.load_state_dict(torch.load('resources/vgg16_cifar10.bin', map_location='cpu'))
    print('Neural network ready.')

    # evaluate the model performance
    accuracy = model_eval(net, testloader, criterion)
    print('Accuracy of the network on the clean test images: %d %%' % (
        100 * accuracy))

    accuracy = model_eval(net, testloader, criterion, attack_method=fgsm_attack, epsilon=0.3)
    print('Accuracy of the network on the adversarial test images: %d %%' % (
        100 * accuracy))


if __name__ == '__main__':
    main()
