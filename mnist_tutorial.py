r"""
The FGSM attacking tutorial on MNIST dataset.
"""
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim

from attacks import *
from models import LeNet
from utils import model_eval, model_train
from torchvision import datasets, transforms


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Adversarial MNIST Example')
    parser.add_argument('--use-pretrained', action='store_true', default=False,
                        help='uses the pretrained model')
    parser.add_argument('--adversarial-training', action='store_true', default=False,
                        help='takes the adversarial training process')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    args = parser.parse_args()

    # Define what device we are using
    use_cuda = torch.cuda.is_available()
    print("CUDA Available: ", use_cuda)
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
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    if args.use_pretrained:
        print('Loading the pretrained model')
        model.load_state_dict(torch.load('resources/lenet_mnist_model.bin', map_location='cpu'))
    else:
        print('Training on the MNIST dataset')
        model_train(model, train_loader, F.nll_loss, optimizer, epochs=10)

    print('Evaluating the neural network')
    # Evaluate the accuracy of the MNIST model on clean examples
    accuracy = model_eval(model, test_loader, F.nll_loss)
    print('Test accuracy on clean examples: ' + str(accuracy))

    # Evaluate the accuracy of the MNIST model on adversarial examples
    accuracy = model_eval(model, test_loader, F.nll_loss, attack_method=fgsm_attack)
    print('Test accuracy on adversarial examples: ' + str(accuracy))

    if args.adversarial_training:
        print("Repeating the process, with adversarial training")
        # Perform adversarial training
        model_train(model, train_loader, F.nll_loss, optimizer, epochs=10, attack_method=fgsm_attack)

        # Evaluate the accuracy of the adversarially trained MNIST model on
        # clean examples
        accuracy = model_eval(model, test_loader, F.nll_loss)
        print('Test accuracy on clean examples: ' + str(accuracy))

        # Evaluate the accuracy of the adversarially trained MNIST model on
        # adversarial examples
        accuracy_adv = model_eval(model, test_loader, F.nll_loss, attack_method=fgsm_attack)
        print('Test accuracy on adversarial examples: ' + str(accuracy_adv))


if __name__ == '__main__':
    main()
