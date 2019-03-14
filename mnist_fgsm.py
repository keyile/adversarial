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


def model_train(model, train_loader, device, epochs, attack):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            data = attack(model, data, target)  # produce the perturbed data
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
        data = attack(model, data, target)  # produce the perturbed data
        output = model(data)
        # get the index of the max log-probability
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    return 1. * correct / len(test_loader.dataset)


def main():

    # Define what device we are using
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    # MNIST Test dataset and dataloader declaration
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor()
        ])),
        batch_size=64, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor()
        ])),
        batch_size=128, shuffle=True, **kwargs)

    # Initialize the network
    model = LeNet().to(device)

    # Load the pretrained model
    # model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))

    # Train an MNIST model
    model_train(model, train_loader, device, 10, noop_attack)

    # Evaluate the accuracy of the MNIST model on benign examples
    accuracy = model_eval(model, test_loader, device, noop_attack)
    print('Test accuracy on benign examples: ' + str(accuracy))

    # Evaluate the accuracy of the MNIST model on adversarial examples
    accuracy = model_eval(model, test_loader, device, fgsm_attack)
    print('Test accuracy on adversarial examples: ' + str(accuracy))

    print("Repeating the process, using adversarial training")
    # Perform adversarial training
    model_train(model, train_loader, device, 10, fgsm_attack)

    # Evaluate the accuracy of the adversarialy trained MNIST model on
    # benign examples
    accuracy = model_eval(model, test_loader, device, noop_attack)
    print('Test accuracy on benign examples: ' + str(accuracy))

    # Evaluate the accuracy of the adversarially trained MNIST model on
    # adversarial examples
    accuracy_adv = model_eval(model, test_loader, device, fgsm_attack)
    print('Test accuracy on adversarial examples: ' + str(accuracy_adv))


if __name__ == '__main__':
    main()
