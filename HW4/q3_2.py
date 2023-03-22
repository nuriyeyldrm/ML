import numpy as np
from torchvision import datasets
import matplotlib.pyplot as plt

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
