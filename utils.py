r"""
The PyTorch routines.
"""
import torch


def model_train(model, train_loader, criterion, optimizer, epochs=1, use_cuda=True, attack_method=None, **attack_params):

    for epoch in range (epochs):
        for data, label in train_loader:
            if use_cuda:
                data, label = data.cuda(), label.cuda()
            # call the attack to produce adversarial data if possible
            if not attack_method is None:
                model.eval()
                data = attack_method(model, data, label, criterion, **attack_params)

            # set the model in training mode
            model.train()
            # ordinary training procedure 
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()


def model_eval(model, test_loader, criterion, use_cuda=True, attack_method=None, **attack_params):
    # set the model in eval mode
    model.eval()

    correct = 0
    prob_sum = 0.0
    for data, label in test_loader:
        if use_cuda:
            data, label = data.cuda(), label.cuda()
        # call the attack to produce adversarial data if possible
        if not attack_method is None:
            data = attack_method(model, data, label, criterion, **attack_params)

        output = model(data)
        # get the index and the max log-probability
        log_prob, predicted = output.max(dim=1, keepdim=True)
        correct += predicted.eq(label.view_as(predicted)).sum().item()
        prob_sum += log_prob.exp().sum().item()

    accuracy = 1. * correct / len(test_loader.dataset)
    prob_aver = prob_sum / len(test_loader.dataset)
    return accuracy, prob_aver
