#%%
import pandas as pd
import random

archivos_csv = ["ATLANTIS_SILICA.csv", 
                "LUNA_HILIC.csv", 
                "LUNA_SILICA.csv", 
                "mod.csv",
                "SCX.csv",
                "unmod.csv",
                "Xbridge.csv"]

# Método para unir los archivos CSV de manera ordenada
def unir_csv_ordenados(archivos):
    
    datos_combinados = pd.DataFrame()
    
    for archivo in archivos:
        df = pd.read_csv(archivo)
        
        datos_combinados = pd.concat([datos_combinados, df])
    
    
    datos_combinados.to_csv("RT_dataset_order.csv", index=False)


def unir_csv_aleatorios(archivos):
    
    lista_dataframes = []
    
    
    for archivo in archivos:
        df = pd.read_csv(archivo)
        lista_dataframes.append(df)
    
    random.shuffle(lista_dataframes)
    
    datos_combinados = pd.concat(lista_dataframes, ignore_index=True)
    
    datos_combinados.to_csv("RT_dataset_shuffle.csv", sep=',', index=False, quoting=None)
    


unir_csv_ordenados(archivos_csv)

df = pd.read_csv("RT_dataset_order.csv")
filas, columnas = df.shape
print(f"El DataFrame ordenado tiene {filas} filas y {columnas} columnas.")

unir_csv_aleatorios(archivos_csv)

df = pd.read_csv("RT_dataset_shuffle.csv")
filas, columnas = df.shape
print(f"El DataFrame shuffle tiene {filas} filas y {columnas} columnas.")


# %%
