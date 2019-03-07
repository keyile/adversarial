r"""
The FGSM attacking tutorial on MNIST dataset.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim

from attacks import fgsm_attack
from attacks import noop_attack
from models import LeNet
from torchvision import datasets, transforms

MNIST_PATH = 'data'
MODEL_PATH = 'resources/lenet_mnist_model.bin'

def model_train(model, train_loader, device, epochs):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()


def model_eval(model, test_loader, device, attack):
    model.eval()
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data = attack(model, data, target) # produce the perturbed data
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    return 1. * correct / len(test_loader.dataset)


def main():

    # Define what device we are using
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    # MNIST Test dataset and dataloader declaration
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(MNIST_PATH, train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor()
        ])),
        batch_size=64, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(MNIST_PATH, train=False, transform=transforms.Compose([
            transforms.ToTensor()
        ])),
        batch_size=128, shuffle=True, **kwargs)

    # Initialize the network
    model = LeNet().to(device)

    # Load the pretrained model
    # model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))

    # Train an MNIST model
    model_train(model, train_loader, device, 10)

    # Evaluate the accuracy of the MNIST model on legitimate test examples
    accuracy = model_eval(model, test_loader, device, noop_attack)
    print('Test accuracy on legitimate test examples: ' + str(accuracy))

    # Evaluate the accuracy of the MNIST model on adversarial examples
    accuracy = model_eval(model, test_loader, device, fgsm_attack)
    print('Test accuracy on adversarial examples: ' + str(accuracy))

"""     # Accuracy counter
    n_1, n_2 = 0, 0

    # Loop over all examples in test set
    for image, label in test_loader:

        # Send the data and label to the device
        image, label = image.to(device), label.to(device)

        # Classify the input
        init_output = model(image)

        # Get the index of the max log-probability
        init_pred = init_output.max(1, keepdim=True)[1]

        # Check for success
        if init_pred.item() == label.item():
            n_1 += 1

        # Call Attack Method
        perturbed = bim_attack(model, image, label)

        # Repeat
        final_output = model(perturbed)
        final_pred = final_output.max(1, keepdim=True)[1]

        if final_pred.item() == label.item():
            n_2 += 1

    # Calculate final accuracy for this epsilon
    init_acc = n_1 / float(len(test_loader))
    final_acc = n_2 / float(len(test_loader))
    print("Accuracy: {} -> {}".format(init_acc, final_acc)) """


if __name__ == '__main__':
    main()
