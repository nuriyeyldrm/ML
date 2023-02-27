import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

correct_class = [1, 1, 0, 1, 1, 0, 1, 1, 0, 0]
confidence_pve = [0.95, 0.85, 0.8, 0.7, 0.55, 0.45, 0.4, 0.3, 0.2, 0.1]
false_pve_rate, true_pve_rate, threshold = roc_curve(correct_class, confidence_pve)
print("False +ve rate: {}\nTrue +ve rate: {}\nThreshold: {}".format(false_pve_rate, true_pve_rate, threshold))

"""
False +ve rate: [0.   0.   0.   0.25 0.25 0.5  0.5  1.  ]
True +ve rate: [0.         0.16666667 0.33333333 0.33333333 0.66666667 0.66666667 1.         1.        ]
Threshold: [1.95 0.95 0.85 0.8  0.55 0.45 0.3  0.1 ]
"""

fpr = [0, 0, 1/4, 2/4, 4/4]
tpr = [0, 2/6, 4/6, 6/6, 6/6]

plt.plot(fpr, tpr, marker=".")
plt.grid()
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.savefig("roc5.png")

