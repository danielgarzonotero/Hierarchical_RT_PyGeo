
#%%
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('ccs_dataset_completed.csv')

peptides_with_ox = df['Modified sequence'].str.contains('\(ox\)').sum()
peptides_no_ox = len(df) - peptides_with_ox

porcentaje_con_ox = (peptides_with_ox / len(df)) * 100
porcentaje_sin_ox = (peptides_no_ox / len(df)) * 100

etiquetas = ['With (ox)', 'No (ox)']
valores = [peptides_with_ox, peptides_no_ox]

plt.bar(etiquetas, valores)
plt.xlabel('Type of peptides')
plt.ylabel('Amount')
plt.title('Dataset distribution in terms (ox) Modification')

for i in range(len(etiquetas)):
    porcentaje = porcentaje_con_ox if i == 0 else porcentaje_sin_ox
    plt.text(i, valores[i], f'{valores[i]} ({porcentaje:.2f}%)',
             horizontalalignment='center', verticalalignment='bottom')

plt.show()  


dataset_path = "ccs_dataset_reduced_10000_log.csv"
df = pd.read_csv(dataset_path)
CCS_values = df.iloc[:, 6]  
#Plots
#Distribution Dataset
plt.hist(CCS_values, bins=100)  
plt.title("CCS Distribution")
plt.xlabel("CCS Values")
plt.ylabel("Frequency")
plt.show()
''' 
df = df[~df['Modified sequence'].str.contains('\(ox\)')]

df.to_csv('data/ccs_dataset_modified.csv', index=False)
 '''
# %%
