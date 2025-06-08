# 在验证集上评估模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels.unsqueeze(1)).sum().item()

    print(f'Accuracy of the network on the dataset: {100 * correct / total:.2f}%')