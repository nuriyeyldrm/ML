import numpy as np
import matplotlib.pyplot as plt


def CalculateProb(dataset, n):
    prob1 = 0
    for d in dataset:
        if d == 1:
            prob1 += 1
    prob0 = n - prob1
    return prob0, prob1


def EntropyFormula(p1, p2):
    return - p1 * np.log2(p1) - p2 * np.log(p2)


def CalculateEntropy(dataset):
    n = len(dataset)
    prob = CalculateProb(dataset, n)

    if prob[0] != 0 and prob[1] != 0:
        p0 = prob[0] / n
        p1 = prob[1] / n
        return EntropyFormula(p1, p0)
    else:
        return 0


def randomData(dataset):
    np.random.seed(2)
    np.random.shuffle(dataset)
    D8192 = dataset[0:8191]
    D2048 = D8192[0:2047]
    D512 = D2048[0:511]
    D128 = D512[0:127]
    D32 = D128[0:31]

    return D8192, D2048, D512, D128, D32


# def CalculateError(dataset, tree):
#     predictions = []
#     err = 0
#     for i in range(0, len(dataset)):
#         predict = Prediction(tree, dataset[i])
#         predictions.append(predict)
#         if predict != dataset[i]:
#             err = err + 1
#
#     n = CalculateNodeNumber(tree)
#     return n, err, predictions


def SubTree(dataset):
    cns_gain_ratio = 0
    cns_entropy = 1
    sub_tree = set
    column = set
    dat = set

    number_of_features = dataset.shape[1] - 1

    for i in range(0, number_of_features):
        feature_data = dataset[:, i]
        for data in feature_data:
            right, left = SplitTree(dataset, data, i)
            gain_ratio = CalculateGainRatio(dataset, right, left)

            entropy = max(CalculateEntropy(left[:, -1]), CalculateEntropy(right[:, -1]))
            # need recommend for # Q 2.3
            # print("Info Gain Ratio: {}  Entropy: {}  Info Gain: {}".format(gain_ratio, entropy, gain_ratio*entropy))
            if entropy < cns_entropy:
                cns_entropy = entropy

            if gain_ratio > cns_gain_ratio:
                cns_gain_ratio = gain_ratio
                column = ("column", i)
                dat = ("data", data)
                sub_tree = (column, dat)
                column = ()
                dat = ()

    return cns_gain_ratio, cns_entropy, sub_tree


def MakeTree(dataset):
    # base case
    if len(dataset) == 0:
        return None

    result = SubTree(dataset)

    sub_tree = result[2]

    # print('data: {}  gain ratio: {}'.format(1, result[1]))

    if result[0] == 0 or result[1] == 1:
        data = dataset[:, -1]
        datum = {}
        for d in data:
            if d not in datum.keys():
                datum[d] = 0
            datum[d] += 1
        return max(datum, key=datum.get)

    right, left = SplitTree(dataset, sub_tree[1][1], sub_tree[0][1])

    return {sub_tree: (MakeTree(left), MakeTree(right))}


def CalculateGainRatio(dataset, right, left):
    ln = len(left)
    rn = len(right)
    dn = len(dataset)

    # base case
    if rn == 0 or ln == 0:
        return 0

    else:
        num_left = ln / dn
        num_right = rn / dn

        info_gain = CalculateEntropy(dataset[:, -1]) - (num_left * CalculateEntropy(left[:, -1]) +
                                                        num_right * CalculateEntropy(right[:, -1]))
        entropy = EntropyFormula(num_left, num_right)

        return info_gain / entropy


def SplitTree(dataset, data, index):

    return dataset[dataset[:, index] < data], dataset[dataset[:, index] >= data]


def plot(data, name):
    x1 = data[:, 0]
    x2 = data[:, 1]
    label = data[:, 2]
    plt.scatter(x1, x2, c=2 * label - 1)
    # plt.title('D1.txt Decision Boundary')
    file_name = "plot/DecisionTree{}.png".format(name)
    plt.savefig(file_name)
    plt.show()


# Q 2.2
print("Q 2.2")
arr = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
arr = np.array(arr)
# MakeTree(arr)

# Q 2.3
print("\nQ 2.3")
filepath = "data/Druns.txt"
dataset = np.loadtxt(filepath, delimiter=" ")
# MakeTree(dataset)

# Q 2.4
print("\nQ 2.4")
filepath = "data/D3leaves.txt"
dataset = np.loadtxt(filepath, delimiter=" ")
# MakeTree(dataset)


# Q 2.5
print("\nQ 2.5")
filepath = "data/D1.txt"
D1 = np.loadtxt(filepath, delimiter=" ")
# MakeTree(D1)

filepath = "data/D2.txt"
D2 = np.loadtxt(filepath, delimiter=" ")
# MakeTree(D2)


# Q 2.6
print("\nQ 2.6")
# plot(D1, "D1")
# plot(D2, "D2")


# Q 2.7
print("\nQ 2.7")
filepath = "data/Dbig.txt"
Dbig = np.loadtxt(filepath, delimiter=" ")
result = randomData(Dbig)

# ans = CalculateError(result[3], tree)
# print(ans[0], ans[1], ans[2])




