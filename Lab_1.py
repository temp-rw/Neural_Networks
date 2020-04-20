import torch
import torchvision
import torchvision.transforms as transforms


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net()
net.to(device)


import torch.optim as optim


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# Training neuro network

loss_array = list()
accuracy_array = list()

dataiter = iter(testloader)
images = dataiter.next()[0].to(device)

for epoch in range(50):  # loop over the dataset multiple times

    epoch_loss = 0.0
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        epoch_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, lbls = data[0].to(device), data[1].to(device)
            output = net(images)
            _, predicted = torch.max(output.data, 1)
            total += lbls.size(0)
            correct += (predicted == lbls).sum().item()

    loss_array.append(epoch_loss / 12500)  # LOSS DATA !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    accuracy_array.append(100 * correct / total)  # ACCURACY DATA !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


print('Finished Training')

loss_array = tuple(loss_array)
accuracy_array = tuple(accuracy_array)


# Making plot
import matplotlib.pyplot as plt
from matplotlib.pyplot import grid


epochs = [i for i in range(1, 51)]

plt_acc = plt.plot(epochs, loss_array)

plt.xlabel("Номер эпохи")
plt.ylabel("Ошибка")
plt.grid()

plt.show()

plt_acc = plt.plot(epochs, accuracy_array)

plt.xlabel("Номер эпохи")
plt.ylabel("Точность (%)")
plt.grid()

plt.show()
