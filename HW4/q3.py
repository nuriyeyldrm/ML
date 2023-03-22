import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch.optim as optim


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.uniform_(m.weight, 0, 0)
        m.bias.data.fill_(0)


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(784, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, 10)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        x = self.linear3(x)
        return x


rnd_seed = 8
torch.manual_seed(rnd_seed)
torch.cuda.manual_seed(rnd_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

model = NeuralNetwork(128, 300, 200, 10).to('cuda')
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

summary = {}
n_total_steps = len(train_loader)
for epoch in range(20):
    loss_epoch = []
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28 * 28).to('cuda')
        labels = labels.to('cuda')
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{20}], Step[{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
            loss_epoch.append(loss.item())
    summary[epoch + 1] = sum(loss_epoch) / len(loss_epoch)

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to('cuda')
        labels = labels.to('cuda')
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        n_errors = n_samples - n_correct
        acc = 100.0 * n_correct / n_samples
        err = 100.0 * n_errors / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')
    print(f'Error rate of the network on the 10000 test images: {err} %')

plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.title(' Learning Curve')

lists = sorted(summary.items())

x, y = zip(*lists)
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))

plt.plot(x, y)
plt.savefig("q3_3.png")

plt.close()

# model = NeuralNetwork(784, 300, 200, 10).to('cuda')
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.1)
# model.apply(init_weights)
#
# summary = {}
# n_total_steps = len(train_loader)
# for epoch in range(20):
#     loss_epoch = []
#     for i, (images, labels) in enumerate(train_loader):
#         images = images.reshape(-1, 28 * 28).to('cuda')
#         labels = labels.to('cuda')
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         if (i + 1) % 100 == 0:
#             print(f'Epoch [{epoch + 1}/{20}], Step[{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
#             loss_epoch.append(loss.item())
#     summary[epoch + 1] = sum(loss_epoch) / len(loss_epoch)
#
# with torch.no_grad():
#     n_correct = 0
#     n_samples = 0
#     for images, labels in test_loader:
#         images = images.reshape(-1, 28 * 28).to('cuda')
#         labels = labels.to('cuda')
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         n_samples += labels.size(0)
#         n_correct += (predicted == labels).sum().item()
#         n_errors = n_samples - n_correct
#         acc = 100.0 * n_correct / n_samples
#         err = 100.0 * n_errors / n_samples
#     print(f'Accuracy of the network on the 10000 test images: {acc} %')
#     print(f'Error rate of the network on the 10000 test images: {err} %')
#
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
#
# plt.title(' Learning Curve')
#
# lists = sorted(summary.items())
#
# x, y = zip(*lists)
# plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
#
# plt.plot(x, y)
# plt.savefig("q3_4b.png")

# model = NeuralNetwork(784, 300, 200, 10).to('cuda')
# # Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = optim.SGD(model.parameters(), lr=0.1)
# model.apply(init_weights)
#
# summary = {}
# n_total_steps = len(train_loader)
# for epoch in range(20):
#     loss_epoch = []
#     for i, (images, labels) in enumerate(train_loader):
#         images = images.reshape(-1, 28 * 28).to('cuda')
#         labels = labels.to('cuda')
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         if (i + 1) % 100 == 0:
#             print(f'Epoch [{epoch + 1}/{20}], Step[{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
#             loss_epoch.append(loss.item())
#     summary[epoch + 1] = sum(loss_epoch) / len(loss_epoch)
#
# with torch.no_grad():
#     n_correct = 0
#     n_samples = 0
#     for images, labels in test_loader:
#         images = images.reshape(-1, 28 * 28).to('cuda')
#         labels = labels.to('cuda')
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         n_samples += labels.size(0)
#         n_correct += (predicted == labels).sum().item()
#         n_errors = n_samples - n_correct
#         acc = 100.0 * n_correct / n_samples
#         err = 100.0 * n_errors / n_samples
#     print(f'Accuracy of the network on the 10000 test images: {acc} %')
#     print(f'Error rate of the network on the 10000 test images: {err} %')
#
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
#
# plt.title('Learning Curve')
#
# lists = sorted(summary.items())
#
# x, y = zip(*lists)
# plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
#
# plt.plot(x, y)
# plt.savefig("q3_4a.png")

