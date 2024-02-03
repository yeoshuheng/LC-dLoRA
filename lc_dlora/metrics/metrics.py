import torch

def accuracy(model, evaluation_set):
    model.eval()
    no_correct, no_seen = 0, 0
    with torch.no_grad():
        for input, label in evaluation_set:
            _, output = torch.max(model(input), dim = 1)
            no_seen += label.size(0)
            no_correct += (output == label).sum().item()
    acc = no_correct / no_seen
    model.train()
    return acc

def accuracy_binary(model, evaluation_set):
    model.eval()
    no_correct, no_seen = 0, 0
    with torch.no_grad():
        for input, label in evaluation_set:
            output = torch.sigmoid(model(input))
            output = torch.where(output > 0.5, 1, 0)
            no_seen += label.size(0)
            no_correct += (output == label).sum().item()
    acc = no_correct / no_seen
    model.train()
    return acc
