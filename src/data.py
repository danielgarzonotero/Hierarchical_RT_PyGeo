import pandas as pd
import os
import sys
import torch
from torch_geometric.data import InMemoryDataset

from src.utils import sequences_geodata, get_features
from src.device import device_info

class GeoDataset(InMemoryDataset):
    def __init__(self, root='../data', raw_name='10000_unmod.csv', processed_name='rt_processed.pt', transform=None, pre_transform=None):
        self.filename = os.path.join(root, raw_name)
        
        self.df = pd.read_csv(self.filename)
        self.x = self.df[self.df.columns[0]].values
        self.y = self.df[self.df.columns[1]].values   
        
        super(GeoDataset, self).__init__(root, transform, pre_transform)
        
        self.data, self.slices = torch.load(self.processed_paths[0])
        

    def processed_file_names(self):
        return ['data.pt']
    

    def process(self):
        peptide_ft_dic, amino_ft_dict, node_ft_dict, edge_ft_dict = get_features(self.x)
        
        total_samples = len(self.x)
        processed_samples = 0
        csv_file_path = 'results/progress.csv'  
        progress_data = {'Iteration': [], 'Progress %': []}
        
        data_list = []
        
        cc = 0 #This is an ID for each peptide in the dataset to then be able to call each dictionary
        aminoacids_features_dict = {} #We use this dictionaries in the forward to call the amino and peptide features for each peptide in the batch
        peptides_features_dict = {}
        
        for i, (x, y) in enumerate(zip(self.x, self.y)):
            device_info_instance = device_info()
            device = device_info_instance.device

            data_list.append(sequences_geodata(cc, x, y, peptide_ft_dic, amino_ft_dict, node_ft_dict, edge_ft_dict, device)[0])
            aminoacids_features_dict[cc] = sequences_geodata(cc, x, y, peptide_ft_dic, amino_ft_dict, node_ft_dict, edge_ft_dict, device)[1]
            peptides_features_dict[cc] = sequences_geodata(cc, x, y, peptide_ft_dic, amino_ft_dict, node_ft_dict, edge_ft_dict, device)[2]
            
            cc += 1
            
            processed_samples += 1
            progress = round(processed_samples / total_samples * 100, 3)
            
            progress_data['Iteration'].append(i)
            progress_data['Progress %'].append(progress)
        
            if processed_samples % 1 == 5:  
                pd.DataFrame(progress_data).to_csv(csv_file_path, index=False)
            
        
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        # directorio del script de Python actual
        script_directory = os.path.dirname(os.path.abspath(__file__ if "__file__" in locals() else sys.argv[0]))

        # directorio relativo para los diccionarios
        dictionaries_dir = os.path.join(script_directory, 'data/dictionaries')

        # crea el directorio 
        os.makedirs(dictionaries_dir, exist_ok=True)

        # rutas relativas para los diccionarios
        aminoacids_features_dict_path = os.path.join(dictionaries_dir, 'aminoacids_features_dict.pt')
        peptides_features_dict_path = os.path.join(dictionaries_dir, 'peptides_features_dict.pt')

        # Guarda los diccionarios 
        torch.save(aminoacids_features_dict, aminoacids_features_dict_path)
        torch.save(peptides_features_dict, peptides_features_dict_path)
                
        

     

