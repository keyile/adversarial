r"""
The attack package contains adversarial attack methods inplements.
"""

import torch
import torch.nn.functional as F

def noop_attack(model, data, label):
    return data

def fgsm_attack(model, data, label, epsilon=0.25):
    r"""The Fast Gradient Sign Method attack.
    """

    # Set requires_grad attribute of tensor. Important for Attack
    data.requires_grad = True

    # Forward pass the data through the model
    output = model(data)

    # Calculate the loss
    loss = F.nll_loss(output, label)

    # Calculate gradients of model in backward pass
    loss.backward()

    # Collect datagrad
    data_grad = data.grad.data

    # Stop recording gradients
    data.requires_grad = False

    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()

    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = data + epsilon*sign_data_grad

    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    # Return the perturbed image
    return perturbed_image


def bim_attack(model, data, label, epsilon=0.1, nb_iter=3, alpha=0.1):
    r"""The Basic Iterate Method attack, also names as I-FGSM.
    """
    eta = 0
    data.requires_grad = False

    for _ in range(nb_iter):
        # Call FGSM Attack
        perturbed_data = fgsm_attack(model, data + eta, label, alpha)

        # Clip intermediate results
        eta = perturbed_data - data
        eta = torch.clamp(eta, -epsilon, epsilon)

    # Return the perturbed image
    return data + eta


def onestep_attack(model, data, label, target=None, epsilon=0.25):
    r"""The One-step target class methods
    """

    # Set requires_grad attribute of tensor. Important for Attack
    data.requires_grad = True

    # Forward pass the data through the model
    output = model(data)

    if target is None:
        # Choose the least-likely class
        with torch.no_grad():
            target = output.argmin(dim=1)

    # Calculate the loss
    loss = F.nll_loss(output, target)

    # Calculate gradients of model in backward pass
    loss.backward()

    # Collect datagrad
    data_grad = data.grad.data

    # Stop recording gradients
    data.requires_grad = False

    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()

    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = data - epsilon*sign_data_grad

    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    # Return the perturbed image
    return perturbed_image