import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def OneNN(x1, x2, df):
    shortest_path = None
    y = 1
    for index, row in df.iterrows():
        distance = np.sqrt(pow(row[0] - x1, 2) + pow(row[1] - x2, 2))
        if shortest_path is None:
            shortest_path = distance
            y = row[2]
        if distance < shortest_path:
            shortest_path = distance
            y = row[2]
    return y


df = pd.read_csv("data/D2z.txt", sep=" ", names=["x1", "x2", "y"])

grid = np.linspace(-2, 2, 40)

data = {"x1": [], "x2": [], "y": []}

for x1 in grid:
    for x2 in grid:
        y = OneNN(x1, x2, df)
        data["x1"].append(x1)
        data["x2"].append(x2)
        data['y'].append(int(y))

df_new = pd.DataFrame(data)
colors = {0: '#071076', 1: '#8C0000'}

fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
df_new.plot.scatter("x1", "x2", c=df_new['y'].map(colors), ax=ax, marker=".", s=20, alpha=0.4)
df.plot.scatter("x1", "x2", c=df["y"].map(colors), ax=ax, alpha=0.8)

plt.savefig("fig.png")




