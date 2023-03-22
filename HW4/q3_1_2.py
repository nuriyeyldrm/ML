from torchvision import datasets, transforms
from scipy.special import softmax, expit
import matplotlib.pyplot as plt
import numpy as np


def Predictions(w1, w2, w3, data):
    ep = expit(np.matmul(w1, data))
    ep2 = expit(np.matmul(w2, ep))
    return softmax(np.matmul(w3, ep2))


def InputsLabels(data):
    length = len(data)
    x = np.zeros(length * 28 * 28).reshape(length, 784)
    y = np.zeros(length)

    for i in range(length):
        x[i] = np.array(data[i][0]).reshape(1, 784)[0]
        y[i] = data[i][1]

    return x.T, y


def Loss(labels, predictions):
    loss = []
    for i in range(len(labels)):
        ind = int(labels[i] - 1)
        y_hat = np.log(predictions[ind, i])
        loss.append(y_hat)
    return -sum(loss)


MNIST_train = datasets.MNIST(root="data", download=True, transform=None, train=True)
MNIST_test = datasets.MNIST(root="data", download=True, transform=None, train=False)

x_train, y_train = InputsLabels(MNIST_train)
x_test, y_test = InputsLabels(MNIST_test)

w1 = np.random.uniform(low=0, high=1, size=(784 * 300)).reshape(300, 784)
w2 = np.random.uniform(low=0, high=1, size=(300 * 200)).reshape(200, 300)
w3 = np.random.uniform(low=0, high=1, size=(200 * 10)).reshape(10, 200)

y = np.zeros((10 * len(y_train))).reshape(10, len(y_train))
for i in range(len(y_train)):
    ind = int(y_train[i] - 1)
    y[ind, i] = 1

batch_size = 64
epochs = 20

train_loss = [Loss(y_train, Predictions(w1, w2, w3, x_train)) / 60000]
test_loss = [Loss(y_test, Predictions(w1, w2, w3, x_test)) / 10000]
error = []
inc = 1

print(Predictions(w1, w2, w3, x_test[:, 0]))

while epochs >= inc:
    length = len(y_train)
    bids = np.random.choice(range(length), size=60000, replace=False)
    batches = length / batch_size
    for b in range(int(batches)):
        start = b * batch_size
        stop = start + batch_size
        ids = bids[start:stop]
        X_batch = x_train[:, ids]
        y_batch = y[:, ids]
        a1 = expit(np.matmul(w1, X_batch))
        a2 = expit(np.matmul(w2, a1))
        g = Predictions(w1, w2, w3, X_batch)
        d3 = (g - y_batch)
        d2 = np.matmul(w3.T, d3) * a2 * (1 - a2)
        d1 = np.matmul(w2.T, d2) * a1 * (1 - a1)
        w3 = w3 - np.matmul(d3, a2.T) * .000001
        w2 = w2 - np.matmul(d2, a1.T) * .000001
        w1 = w1 - np.matmul(d1, X_batch.T) * .000001
    print(Predictions(w1, w2, w3, x_test[:, 0]))
    print(inc, Loss(y_train, Predictions(w1, w2, w3, x_train)) / 60000)
    train_loss.append(Loss(y_train, Predictions(w1, w2, w3, x_train)) / 60000)
    test_loss.append(Loss(y_test, Predictions(w1, w2, w3, x_test)) / 10000)
    pred = Predictions(w1, w2, w3, x_test)
    e = 0
    for i in range(len(y_test)):
        p = np.argmax(pred[:, i])
        if p != int(y_test[i]): e += 1
    error.append(e / len(y_test))
    inc += 1

for i in range(len(error)):
    print(i + 1, "&", round(error[i] * 100, 4), "\%", "\\\\")

plt.plot(range(epochs + 1), train_loss)
plt.title("Learning Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("q3_2new.png")
