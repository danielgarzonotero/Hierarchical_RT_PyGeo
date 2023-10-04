
#%%
import pandas as pd
import random

df = pd.read_csv('LUNA_SILICA.csv' )

total_rows = len(df)
target_rows = 10000

if total_rows > target_rows:
    
    rows_to_drop = total_rows - target_rows

    
    rows_to_drop_indices = random.sample(range(total_rows), rows_to_drop)

    df = df.drop(rows_to_drop_indices)

df.to_csv('/home/vvd9fd/Documents/Bilodeau Group/Codes/0.Research/Hierarchical_RT_PyGeo/data/LUNA_SILICA_10000.csv', index=False, quoting=None)
df = pd.read_csv('/home/vvd9fd/Documents/Bilodeau Group/Codes/0.Research/Hierarchical_RT_PyGeo/data/LUNA_SILICA_10000.csv')
filas, columnas = df.shape
print(f"El DataFrame shuffle tiene {filas} filas y {columnas} columnas.")


# %%
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/home/vvd9fd/Documents/Bilodeau Group/Codes/0.Research/Hierarchical_RT_PyGeo/data/RT_dataset_shuffle_reduced_30000.csv')
print(df.shape)
RT_values = df.iloc[:, 1]  

#Distribution Dataset
plt.hist(RT_values, bins=100)  
plt.title("RT Distribution")
plt.xlabel("RT Values (sec)")
plt.ylabel("Frequency")
plt.show()
# %%

# %%
import math
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/home/vvd9fd/Documents/Bilodeau Group/Codes/0.Research/Hierarchical_RT_PyGeo/data/Xbridge_10000.csv')
column_index = 1
column_title = df.columns[column_index]

print("Título de la columna número 6:", column_title)
column_title = 'log(RT)'
df = df.rename(columns={df.columns[column_index]: column_title})
df.iloc[:, column_index] = df.iloc[:, column_index].apply(lambda x: math.log(x))
df.to_csv("log_RT_dataset_shuffle_reduced_30000.csv", index=False) 

plt.hist(df.iloc[:, column_index], bins=10)
plt.xlabel(' Values Log (RT)')
plt.ylabel('Frequency')
plt.title('Histogram of Log(RT)')
plt.show()
# %%
