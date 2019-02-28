import torch

from attacks import fgsm_attack
from model import LeNet
from torchvision import datasets, transforms

MNIST_PATH = 'data'
MODEL_PATH = 'resources/lenet_mnist_model.bin'

if __name__ == '__main__':
    # MNIST Test dataset and dataloader declaration
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(MNIST_PATH, train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
        ])),
        batch_size=1, shuffle=True)

    # Define what device we are using
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    # Initialize the network
    model = LeNet().to(device)

    # Load the pretrained model
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))

    # Set the model in evaluation mode. In this case this is for the Dropout layers
    model.eval()

    # Accuracy counter
    n1, n2 = 0, 0

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
            n1 += 1

        # Call Attack Method
        perturbed = fgsm_attack(model, image, label)

        # Repeat
        final_output = model(perturbed)
        final_pred = final_output.max(1, keepdim=True)[1]

        if final_pred.item() == label.item():
            n2 += 1

    # Calculate final accuracy for this epsilon
    init_acc = n1 / float(len(test_loader))
    final_acc = n2 / float(len(test_loader))
    print("Accuracy: {} -> {}".format(init_acc, final_acc))
