import torch
import torch.nn.functional as F


# FGSM attack code
def fgsm_attack(model, data, target, epsilon=0.1):

    # Set requires_grad attribute of tensor. Important for Attack
    data.requires_grad = True

    # Forward pass the data through the model
    output = model(data)

    # Calculate the loss
    loss = F.nll_loss(output, target)

    # Zero all existing gradients
    model.zero_grad()

    # Calculate gradients of model in backward pass
    loss.backward()

    # Collect datagrad
    data_grad = data.grad.data

    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()

    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = data + epsilon*sign_data_grad

    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    # Return the perturbed image
    return perturbed_image


# BIM attack code
def bim_attack(model, data, target, alpha=0.05, nb_iter=3, epsilon=0.05):
    eta = 0
    for _ in range(nb_iter):
        # Call FGSM Attack
        perturbed_data = fgsm_attack(
            model, (data+eta).detach(), target, alpha)

        # Clip intermediate results
        eta = perturbed_data - data
        eta = torch.clamp(eta, -epsilon, epsilon)

    # Return the perturbed image
    return data + eta
