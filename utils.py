r"""
The PyTorch routines.
"""

def model_train(model, train_loader, criterion, optimizer, epochs=1, use_cuda=True, attack_method=None, **attack_params):
    model.train()
    for epoch in range (epochs):
        for data, label in train_loader:
            if use_cuda:
                data, label = data.cuda(), label.cuda()
            # call the attack to produce adversarial data if possible
            if not attack_method is None:
                data = attack_method(model, data, label, criterion, **attack_params)
            # ordinary training procedure 
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()


def model_eval(model, test_loader, criterion, use_cuda=True, attack_method=None, **attack_params):
    model.eval()
    # accuracy counter
    correct = 0
    for data, label in test_loader:
        if use_cuda:
            data, label = data.cuda(), label.cuda()
        # call the attack to produce adversarial data if possible
        if not attack_method is None:
            data = attack_method(model, data, label, criterion, **attack_params)
        output = model(data)
        # get the index of the max log-probability
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(label.view_as(pred)).sum().item()

    return 1. * correct / len(test_loader.dataset)