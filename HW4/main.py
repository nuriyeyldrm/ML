import glob
import math
from collections import defaultdict
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.ticker as mticker
import torch.optim as optim
import numpy as np
from torchvision import datasets
import matplotlib.pyplot as plt


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

def Language(path, languages):
    lang = {'e': {}, 's': {}, 'j': {}}

    for language in languages:
        temp = {}
        for file in glob.glob(f"{path}/{language}[0-9].txt"):
            f_name = f"{file}"

            with open(f_name, 'r') as f:
                line = f.read()

            for c in line:
                if c != "\n":
                    if temp.get(c) is None:
                        temp[c] = 1
                    else:
                        temp[c] = temp[c] + 1
                else:
                    continue

        lang[language] = temp

    return lang


languages = ['e', 'j', 's']
path = "languageID"
lang = Language(path, languages)

dictionary = {'e': {}, 's': {}, 'j': {}}
for language in languages:
    count = {}
    tot = 0
    for char in sorted(lang[language].keys()):
        tot += lang[language][char]
    for char in sorted(lang[language].keys()):
        if count.get(char) is None:
            count[char] = float(lang[language][char] + 0.5) / (tot + (27 * 0.5))
    dictionary[language] = count

    print(f"{language}, propability: {count}\n")

test = f"{path}/e10.txt"

cnt = {}
with open(test, 'r') as f:
    line = f.read()
for c in line:
    if c == "\n":
        continue
    if cnt.get(c) is None:
        cnt[c] = 1
    else:
        cnt[c] += 1
for i in sorted(cnt):
    print(i, ": ", cnt[i])

log_likelihood = {'e': float(0), 'j': float(0), 's': float(0)}
for language in languages:
    val = dictionary[language]
    tot = 0
    for char in cnt:
        if val.get(char) is None:
            val[char] = 0.5 / 27 * 0.5
        tot = tot + math.log(val[char]) * cnt[char]
    print(f"p_hat {language} : {tot}")
    log_likelihood[language] = tot

prior = float((10 + 0.5)) / (30 + 3 * 0.5)
posterior = [log_likelihood[i] * prior for i in log_likelihood]
for i in range(3):
    print(f"posterior {languages[i]} : {posterior[i]}")


def Performance(fname):
    x_e = 0
    x_j = 0
    x_s = 0

    for name in fname:
        cnts = defaultdict(int)
        f = open("languageID/" + name, "r")
        text = f.read()
        f.close()
        for char in text:
            if char.isalpha() or char == " ":
                cnts[char] += 1

        prob_e = 0
        for key in cnts.keys():
            prob_e += (math.log(dictionary['e'][key]) * cnts[key])

        prob_j = 0
        for key in cnts.keys():
            prob_j += (math.log(dictionary['j'][key]) * cnts[key])

        prob_s = 0
        for key in cnts.keys():
            prob_s += (math.log(dictionary['s'][key]) * cnts[key])
        posterior_e = prob_e + math.log(1 / 3)
        posterior_j = prob_j + math.log(1 / 3)
        posterior_s = prob_s + math.log(1 / 3)

        if posterior_e > posterior_j and posterior_e > posterior_s:
            x_e += 1
        elif posterior_j > posterior_e and posterior_j > posterior_s:
            x_j += 1
        elif posterior_s > posterior_j and posterior_s > posterior_e:
            x_s += 1

    return x_e, x_s, x_j


e_test = ["e10.txt", "e11.txt", "e12.txt", "e13.txt", "e14.txt", "e15.txt", "e16.txt", "e17.txt", "e18.txt", "e19.txt"]
s_test = ["s10.txt", "s11.txt", "s12.txt", "s13.txt", "s14.txt", "s15.txt", "s16.txt", "s17.txt", "s18.txt", "s19.txt"]
j_test = ["j10.txt", "j11.txt", "j12.txt", "j13.txt", "j14.txt", "j15.txt", "j16.txt", "j17.txt", "j18.txt", "j19.txt"]
e_e, e_s, e_j = Performance(e_test)
print(e_e, e_s, e_j)

s_e, s_s, s_j = Performance(s_test)
print(s_e, s_s, s_j)

j_e, j_s, j_j = Performance(j_test)
print(j_e, j_s, j_j)


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

MNIST_train = datasets.MNIST(root="data", download=True, transform=None, train=True)
MNIST_test = datasets.MNIST(root="data", download=True, transform=None, train=False)


class NeuralNetwork:
    def __init__(self, d=784, d1=300, d2=200, k=10, lr=0.5, num_epochs=15, batch_size=64):
        self.W1 = np.random.randn(d1, d) / np.sqrt(d)
        self.W2 = np.random.randn(d2, d1) / np.sqrt(d1)
        self.W3 = np.random.randn(k, d2) / np.sqrt(d2)

        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def inter_forward(self, x, y):
        z1 = np.dot(self.W1, x)
        a1 = Sigmoid(z1)
        z2 = np.dot(self.W2, a1)
        a2 = Sigmoid(z2)
        return a1.T, a2.T

    def forward(self, x, y):
        z1 = np.dot(self.W1, x)
        a1 = Sigmoid(z1)
        z2 = np.dot(self.W2, a1)
        a2 = Sigmoid(z2)
        z3 = np.dot(self.W3, a2)
        y_hat = Sigmoid(z3)
        return y_hat.T

    def save_ckpt(self, id):
        np.savez(f'{id}_weights.npz', W1=self.W1, W2=self.W2, W3=self.W3)

    def backward(self, x, y, y_hat):
        a1, a2 = self.inter_forward(x, y)
        delta4 = -(y - y_hat)  # 32, 10
        dW3 = np.einsum('ij,ik->ijk', delta4, a2)
        delta3 = np.dot(delta4, self.W3) * a2 * (1 - a2)  # 32, 200
        dW2 = np.einsum('ij,ik->ijk', delta3, a1)  # 32, 200, 300
        delta2 = np.dot(delta3, self.W2) * a1 * (1 - a1)
        dW1 = np.einsum('ij,ik->ijk', delta2, x.T)  # 32, 300, 784
        self.W1 -= self.lr * np.mean(dW1, axis=0)
        self.W2 -= self.lr * np.mean(dW2, axis=0)
        self.W3 -= self.lr * np.mean(dW3, axis=0)


def Sigmoid(data):
    return 1.0 / (1.0 + np.exp(-data))


def Softmax(data):
    exp = np.exp(data)
    return exp / np.sum(exp, axis=0)


def CrossEntropyLoss(y_hat, y):
    return -np.sum(np.log(y_hat) * y, axis=1).mean()


def Convert(labels):
    length = len(labels)
    num_labels = len(np.unique(labels))
    arr = np.zeros((length, num_labels))
    for i, label in enumerate(labels):
        arr[i, label] = 1
    return arr


def NeuralNetworkTrain(nn, x_train, y_train, x_test, y_test, batch_size, lr, num_epochs, d, save=False, id=None):
    num_train_batches = len(x_train) // batch_size
    training_loss = []
    test_loss = []
    for epoch in range(num_epochs):
        train_correct = 0
        epoch_train_loss = 0
        for i in range(num_train_batches):
            idx = range(i * batch_size, (i + 1) * batch_size)
            batch_x = np.reshape(x_train[idx], (batch_size, d))
            batch_y = np.array(y_train[idx])
            y_hat = nn.forward(batch_x.T, batch_y)
            loss = CrossEntropyLoss(y_hat, batch_y)
            predictions = np.argmax(y_hat, axis=1)
            true_labels = np.argmax(batch_y, axis=1)
            train_correct += np.sum(predictions == true_labels)
            nn.backward(batch_x.T, batch_y, y_hat)

            epoch_train_loss += loss * batch_size
        epoch_train_loss /= (num_train_batches * batch_size)
        epoch_train_accuracy = train_correct / (num_train_batches * batch_size)
        training_loss.append(epoch_train_loss)
        print(
            f"Epoch: {epoch + 1}/{num_epochs}, Training loss: {epoch_train_loss:.4f}, Training Accuracy: {epoch_train_accuracy:.4f}")

        epoch_test_loss, epoch_test_accuracy = NeuralNetworkTest(nn, x_test, y_test, batch_size, lr, d)
        test_loss.append(epoch_test_loss)

    if save:
        nn.save_ckpt(id)
    return training_loss, test_loss


def NeuralNetworkTest(nn, x_test, y_test, batch_size, lr, d):
    num_test_batches = len(x_test) // batch_size
    test_loss = 0.0
    test_correct = 0
    for i in range(num_test_batches):
        idx = range(i * batch_size, (i + 1) * batch_size)
        batch_x = np.reshape(x_test[idx], (batch_size, d))
        batch_y = np.array(y_test[idx])

        y_hat = nn.forward(batch_x.T, batch_y)
        loss = CrossEntropyLoss(y_hat, batch_y)
        test_loss += loss * batch_size

        predictions = np.argmax(y_hat, axis=1)
        true_labels = np.argmax(batch_y, axis=1)
        test_correct += np.sum(predictions == true_labels)
    test_accuracy = test_correct / (num_test_batches * batch_size)
    test_loss = test_loss / (num_test_batches * batch_size)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    return test_loss, test_accuracy


x_train = MNIST_train.data / 255
x_test = MNIST_test.data / 255

y_train = Convert(MNIST_train.targets)
y_test = Convert(MNIST_test.targets)

lr = 0.5
batch_size = 64
num_epochs = 15
d = 784
nn = NeuralNetwork(lr=lr, batch_size=batch_size)
training_loss_64, test_loss_64 = NeuralNetworkTrain(nn, x_train, y_train, x_test, y_test, batch_size, lr,
                                                    num_epochs, d)

print(training_loss_64, test_loss_64)

plt.plot(range(1, num_epochs + 1), training_loss_64, label='Training Loss')
plt.plot(range(1, num_epochs + 1), test_loss_64, label='Test Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Learning Curve")
plt.legend()
plt.savefig("q3_2.png")

