from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt


def DecisionTree(train_x, train_y, test_x, test_y):
    clf = DecisionTreeClassifier(random_state=760)
    fit = clf.fit(train_x, train_y)
    n = fit.tree_.node_count
    err = 1 - fit.score(test_x, test_y)

    return n, err


filepath = "data/Dbig.txt"
dataset = np.loadtxt(filepath, delimiter=" ")
dataset_x = dataset[:, 0:2]
dataset_y = dataset[:, -1]

rng = np.random.RandomState(0)
# D32 ⊂ D128 ⊂ D512 ⊂ D2048 ⊂ D8192
D8192_x, test8192_x, D8192_y, test8192_y = train_test_split(dataset_x, dataset_y, train_size=8192, random_state=rng)
D2048_x, test2048_x, D2048_y, test2048_y = train_test_split(D8192_x, D8192_y, train_size=2048, random_state=rng)
D512_x, test512_x, D512_y, test512_y = train_test_split(D2048_x, D2048_y, train_size=512, random_state=rng)
D128_x, test128_x, D128_y, test128_y = train_test_split(D512_x, D512_y, train_size=128, random_state=rng)
D32_x, test32_x, D32_y, test32_y = train_test_split(D128_x, D128_y, train_size=32, random_state=rng)

number_of_node = []
err = []

r8192 = DecisionTree(D8192_x, D8192_y, test8192_x, test8192_y)
number_of_node.append(r8192[0])
err.append(r8192[1])

r2048 = DecisionTree(D2048_x, D2048_y, test2048_x, test2048_y)
number_of_node.append(r2048[0])
err.append(r2048[1])

r512 = DecisionTree(D512_x, D512_y, test512_x, test512_y)
number_of_node.append(r512[0])
err.append(r512[1])

r128 = DecisionTree(D128_x, D128_y, test128_x, test128_y)
number_of_node.append(r128[0])
err.append(r128[1])

r32 = DecisionTree(D32_x, D32_y, test32_x, test32_y)
number_of_node.append(r32[0])
err.append(r32[1])

print(number_of_node, err)

n = [8192, 2048, 512, 128, 32]
plt.title('Learning Curve')
plt.plot(n, err)
plt.xlabel('n')
plt.ylabel('Error')
plt.grid()
plt.savefig("plot/LeraningCurveSklearn.png")

plt.show()
