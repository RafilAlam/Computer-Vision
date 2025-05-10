import torch as T
import torch.nn as nn
import torch.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
device = T.device('cuda' if T.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 100
batch_size = 32
learning_rate = 0.001

# dataset has PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

# Data loader
train_loader = T.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = T.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        # N, 3, 32, 32
        x = T.relu(self.conv1(x))  # -> N, 32, 30, 30
        x = self.pool(x)           # -> N, 32, 15, 15
        x = T.relu(self.conv2(x))  # -> N, 64, 13, 13
        x = self.pool(x)           # -> N, 64, 6, 6
        x = T.relu(self.conv3(x))  # -> N, 64, 4, 4
        x = T.flatten(x, 1)    # -> N, 1024
        x = T.relu(self.fc1(x))    # -> N, 64
        x = self.fc2(x)            # -> N, 10
        return x
    
model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimiser = T.optim.Adam(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):

    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/n_total_steps:.3f}')

print('Finished Training')
PATH = './cnn.pth'
T.save(model.state_dict(), PATH)

loaded_model = ConvNet()
loaded_model.load_state_dict(T.load(PATH)) # it takes the loaded dictionary, not the path
loaded_model.to(device)
loaded_model.eval() # optimise for evaluation mode (remove unnecessary training prerequisites)

with T.no_grad():
    n_correct = 0
    n_correct2 = 0
    n_samples = len(test_loader.dataset)

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        # max returns (value ,index)
        _, predicted = T.max(outputs, 1)
        n_correct += (predicted == labels).sum().item()

        outputs2 = loaded_model(images)
        _, predicted2 = T.max(outputs.data, 1)
        n_correct2 += (predicted2 == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    acc2 = 100.0 * n_correct2 / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')
    print(f'Accuracy of the network on the 10000 test images: {acc2} %')
