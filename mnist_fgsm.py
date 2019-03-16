r"""
The FGSM attacking tutorial on MNIST dataset.
"""
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim

from attacks import *
from models import LeNet
from torchvision import datasets, transforms


def model_train(model, train_loader, device, epochs, attack, **attack_params):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    for epoch in range(epochs):
        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            data = attack(model, data, label, **attack_params)  # produce the perturbed data
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, label)
            loss.backward()
            optimizer.step()


def model_eval(model, test_loader, device, attack, **attack_params):
    model.eval()
    correct = 0
    for data, label in test_loader:
        data, label = data.to(device), label.to(device)
        data = attack(model, data, label, **attack_params)  # produce the perturbed data
        output = model(data)
        # get the index of the max log-probability
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(label.view_as(pred)).sum().item()

    return 1. * correct / len(test_loader.dataset)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Adversarial MNIST Example')
    parser.add_argument('--use-pretrained', action='store_true', default=False,
                        help='uses the pretrained model')
    parser.add_argument('--no-adversarial-training', action='store_true', default=False,
                        help='takes the adversarial training process')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    args = parser.parse_args()

    # Define what device we are using
    use_cuda = torch.cuda.is_available()
    # print("CUDA Available: ", use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")

    # MNIST Test dataset and dataloader declaration
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor()
        ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor()
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # Initialize the network
    model = LeNet().to(device)
    if args.use_pretrained:
        # Load the pretrained model
        model.load_state_dict(torch.load('resources/lenet_mnist_model.bin', map_location='cpu'))
    else:
        # Train an MNIST model
        model_train(model, train_loader, device, 10, noop_attack)

    # Evaluate the accuracy of the MNIST model on clean examples
    accuracy = model_eval(model, test_loader, device, noop_attack)
    print('Test accuracy on clean examples: ' + str(accuracy))

    # Evaluate the accuracy of the MNIST model on adversarial examples
    accuracy = model_eval(model, test_loader, device, fgsm_attack)
    print('Test accuracy on adversarial examples: ' + str(accuracy))

    if not args.no_adversarial_training:
        print("Repeating the process, using adversarial training")
        # Perform adversarial training
        model_train(model, train_loader, device, 10, fgsm_attack)

        # Evaluate the accuracy of the adversarially trained MNIST model on
        # clean examples
        accuracy = model_eval(model, test_loader, device, noop_attack)
        print('Test accuracy on clean examples: ' + str(accuracy))

        # Evaluate the accuracy of the adversarially trained MNIST model on
        # adversarial examples
        accuracy_adv = model_eval(model, test_loader, device, fgsm_attack)
        print('Test accuracy on adversarial examples: ' + str(accuracy_adv))


if __name__ == '__main__':
    main()
