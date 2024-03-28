import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import top_k_accuracy_score


torch.manual_seed(42)


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)


model = torchvision.models.resnet101(pretrained=True)


for param in model.parameters():
    if isinstance(param, nn.Conv2d):
        param.requires_grad = False


num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


optimizers = {
    "Adam": optim.Adam(model.fc.parameters(), lr=0.001),
    "Adagrad": optim.Adagrad(model.fc.parameters(), lr=0.001),
    "Adadelta": optim.Adadelta(model.fc.parameters(), lr=0.001)
}

criterion = nn.CrossEntropyLoss()

results = {}

for opt_name, optimizer in optimizers.items():
    print(f"Finetuning with {opt_name}...")
    best_val_acc = 0.0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(5):

        model.train()
        running_loss = 0.0
        running_corrects = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs.data, 1)
            running_corrects += (preds == labels).sum().item()

        epoch_loss = running_loss / len(trainset)
        epoch_acc = running_corrects / len(trainset)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        # Validation
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        with torch.no_grad():
            for val_images, val_labels in testloader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)

                val_outputs = model(val_images)
                val_loss = criterion(val_outputs, val_labels)
                val_running_loss += val_loss.item() * val_images.size(0)
                _, val_preds = torch.max(val_outputs, 1)
                val_running_corrects += (val_preds == val_labels).sum().item()

        val_epoch_loss = val_running_loss / len(testset)
        val_epoch_acc = val_running_corrects / len(testset)
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc)

        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc


    top5_corrects = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.topk(outputs, k=5, dim=1)

            top5_corrects += top_k_accuracy_score(labels.cpu().numpy(), preds.cpu().numpy(), k=5) * len(labels)
            total_samples += len(labels)

    top5_accuracy = top5_corrects / total_samples
    print(f"Top-5 Testing Accuracy: {top5_accuracy:.4f}")


    results[opt_name] = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc
    }


plt.figure(figsize=(12, 5))


plt.subplot(1, 2, 1)
for opt_name, result in results.items():
    plt.plot(np.arange(1, len(result['train_losses']) + 1), result['train_losses'], label=opt_name)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()


plt.subplot(1, 2, 2)
for opt_name, result in results.items():
    plt.plot(np.arange(1, len(result['val_accs']) + 1), result['val_accs'], label=opt_name)
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


print("This is version-1")
