#%%%
import pandas as pd
import random

df = pd.read_csv('dia.csv')


total_rows = len(df)
target_rows = 10000

if total_rows > target_rows:
    
    rows_to_drop = total_rows - target_rows

    
    rows_to_drop_indices = random.sample(range(total_rows), rows_to_drop)

    df = df.drop(rows_to_drop_indices)

df.to_csv('dia_reduced_10000.csv', index=False)

# %%
import pandas as pd
import matplotlib.pyplot as plt
dataset_path = "data/dia.csv"
df = pd.read_csv(dataset_path)
CCS_values = df.iloc[:, 1]  
#Plots
#Distribution Dataset
plt.hist(CCS_values, bins=10)  
plt.title("RT Distribution")
plt.xlabel("RT Values (sec)")
plt.ylabel("Frequency")
plt.show()
# %%
