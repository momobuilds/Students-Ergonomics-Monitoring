import pandas as pd
import numpy as np

mean_col1 = 1.3
mean_col2 = 1 / 2
std_dev = 0.1

input_file = "../../data/raw/HR/Adquisition/Relaxed_3.csv"
output_file = "../../data/raw/HR/Adquisition/Stressed_3.csv"

df = pd.read_csv(input_file)

if df.shape[1] < 2:
    raise ValueError("El CSV debe tener al menos 2 columnas.")

factor_col1 = np.random.normal(mean_col1, std_dev, len(df))
factor_col2 = np.random.normal(mean_col2, std_dev, len(df))

df.iloc[:, 0] = df.iloc[:, 0] * factor_col1
df.iloc[:, 1] = df.iloc[:, 1] * factor_col2

df.to_csv(output_file, index=False)

print(f"File saved in: {output_file}")