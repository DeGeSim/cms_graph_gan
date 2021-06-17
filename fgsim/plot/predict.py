import pandas as pd
import matplotlib.pyplot as plt


from ..config import conf
df = pd.read_csv(f'wd/{conf.tag}/prediction.csv')
plt.figure(num=None, figsize=(10, 5), dpi=80, facecolor="w", edgecolor="k")
plt.scatter(df["Energy "],df["Prediction"])
plt.xlabel("Enery")
plt.ylabel("Prediction")
plt.title(f"Prediction vs True Energy for {conf.tag}")
plt.savefig((f"wd/{conf.tag}/prediction.pdf"))
