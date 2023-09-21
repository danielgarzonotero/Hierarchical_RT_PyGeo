import pandas as pd
import os

import torch
from torch_geometric.data import InMemoryDataset

from src.utils import smiles2geodata, get_atom_features

class GeoDataset(InMemoryDataset):
    def __init__(self, root='../data', raw_name='dia_reduced_10000.csv', processed_name='rt_processed.pt', transform=None, pre_transform=None):
        self.filename = os.path.join(root, raw_name)
        
        self.df = pd.read_csv(self.filename)
        self.x = self.df[self.df.columns[0]].values
        self.y = self.df[self.df.columns[1]].values   
        
        super(GeoDataset, self).__init__(root, transform, pre_transform)
        
        self.data, self.slices = torch.load(self.processed_paths[0])
        

    def processed_file_names(self):
        return ['data.pt']
    

    def process(self):
        node_features_dict, edge_features_dict = get_atom_features(self.x)
        total_samples = len(self.x)
        processed_samples = 0
        
        csv_file_path = 'results/progress.csv'  
        
        progress_data = {'Iteration': [], 'Progress %': []}
        
        data_list = []
        
        for i, (x, y) in enumerate(zip(self.x, self.y)):
            
            data_list.append(smiles2geodata(x, y, node_features_dict, edge_features_dict))
            
            
            processed_samples += 1
            progress = round(processed_samples / total_samples * 100, 3)
            
            progress_data['Iteration'].append(i)
            progress_data['Progress %'].append(progress)
        
            if processed_samples % 1 == 5:  
                pd.DataFrame(progress_data).to_csv(csv_file_path, index=False)
        
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

