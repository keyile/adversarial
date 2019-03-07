r"""
The attack package contains adversarial attack methods inplements.
"""

import torch
import torch.nn.functional as F


def fgsm_attack(model, data, target, epsilon=0.1):
    r"""The Fast Gradient Sign Method attack.
    """
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
    data_grad = data.grad.data.clone().detach()

    # Don't record anymore
    data.requires_grad = False

    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign_()

    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = data + epsilon*sign_data_grad

    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    # Return the perturbed image
    return perturbed_image


def bim_attack(model, data, target, epsilon=0.1, nb_iter=3, alpha=0.1):
    r"""The Basic Iterate Method attack, also names as I-FGSM.
    """
    eta = 0
    data.requires_grad = False

    for _ in range(nb_iter):
        # Call FGSM Attack
        perturbed_data = fgsm_attack(model, data + eta, target, alpha)

        # Clip intermediate results
        eta = perturbed_data - data
        eta = torch.clamp(eta, -epsilon, epsilon)

    # Return the perturbed image
    return data + eta
