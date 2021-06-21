import matplotlib.pyplot as plt
import pandas as pd

df_simple_forward = pd.read_csv(f"wd/simple_forward/prediction.csv")
df_simple_msgpass = pd.read_csv(f"wd/simple_msgpass/prediction.csv")
plt.figure(num=None, figsize=(10, 5), dpi=80, facecolor="w", edgecolor="k")
plt.scatter(
    df_simple_forward["Energy "],
    df_simple_forward["Prediction"],
    label="Layerwise Pass Network",
    alpha=0.5,
    s=12,
)
plt.scatter(
    df_simple_msgpass["Energy "],
    df_simple_msgpass["Prediction"],
    label="Message Pass Network",
    alpha=0.5,
    s=12,
)
plt.xlabel("Enery")
plt.ylabel("Prediction")
plt.title(f"Prediction vs True Energy")
plt.legend()
plt.savefig(f"wd/prediction.pdf")
