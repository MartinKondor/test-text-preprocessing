"""
This file separates the title and label attributes into a different .csv file.

WARNING: Only run if there is no data/data.csv file.
"""
import pandas as pd


try:
    df = pd.read_csv("data/fake_or_real_news.csv")
except FileNotFoundError as e:
    print("FileNotFoundError: Download the data first! See the /data/README.md file for more instructions.")
    exit(1)


X = df["title"].values.reshape(-1, 1)
y = df["label"].values.reshape(-1, 1)
print("We are going to use only the titles from the database")
print("[i] X.shape =", X.shape)
print("[i] y.shape =", y.shape)


text = "title;label\n"
for xs, ys in zip(X, y):
    classe = 1 if ys[0].replace(";", "").upper() == "FAKE" else 0
    text += "{};{}\n".format(xs[0].replace(";", ""), classe)


print("[i] Writing to file...")
with open("data/data.csv", "w+") as file:
    file.write(text)

print("[i] data/data.csv created!")
