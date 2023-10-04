
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, ARMAConv
from torch_geometric.nn import aggr
from torch_scatter import scatter
from src.data import GeoDataset
from src.device import device_info


dataset = GeoDataset(root='data')
aminoacids_features_dict = torch.load('/home/vvd9fd/Documents/Bilodeau Group/Codes/0.Research/Hierarchical_RT_PyGeo/data/aminoacids_features_dict.pt')
class GCN_Geo(torch.nn.Module):
    def __init__(self,
                initial_dim_gcn,
                edge_dim_feature,
                hidden_dim_nn_1,
                p1,
                hidden_dim_nn_2,
                p2,
                hidden_dim_nn_3,
                p3,
                hidden_dim_gat_0,
                hidden_dim_gat_1,
                hidden_dim_fcn_1,
                hidden_dim_fcn_2,
                hidden_dim_fcn_3):
        super(GCN_Geo, self).__init__()

        self.nn_conv_1 = NNConv(initial_dim_gcn, hidden_dim_nn_1,
                                nn=torch.nn.Sequential(torch.nn.Linear(edge_dim_feature, initial_dim_gcn * hidden_dim_nn_1)), 
                                aggr='add' )
        self.dropout_1 = nn.Dropout(p=p1)
        
        self.nn_conv_2 = NNConv(hidden_dim_nn_1, hidden_dim_nn_2,
                                nn=torch.nn.Sequential(torch.nn.Linear(edge_dim_feature, hidden_dim_nn_1 * hidden_dim_nn_2)), 
                                aggr='add')
        self.dropout_2 = nn.Dropout(p=p2)
        
        self.nn_conv_3 = NNConv(hidden_dim_nn_2, hidden_dim_nn_3,
                                nn=torch.nn.Sequential(torch.nn.Linear(edge_dim_feature, hidden_dim_nn_2 * hidden_dim_nn_3)), 
                                aggr='add')
        self.dropout_3 = nn.Dropout(p=p3)
        
        #The 4 comes from the four amino acid features that were concatenated
        self.nn_gat_1 = ARMAConv(hidden_dim_nn_3+4, hidden_dim_gat_0, num_stacks = 3, dropout=0, num_layers=7, shared_weights = False ) #TODO
        
        
        self.readout = aggr.SumAggregation()
        
        self.linear1 = nn.Linear(hidden_dim_gat_0, hidden_dim_fcn_1)
        self.linear2 = nn.Linear(hidden_dim_fcn_1, hidden_dim_fcn_2)
        self.linear3 = nn.Linear(hidden_dim_fcn_2, hidden_dim_fcn_3)
        self.linear4 = nn.Linear(hidden_dim_fcn_3, 1)

    def forward(self, data):
        cc, x, edge_index,  edge_attr, monomer_labels = data.cc, data.x, data.edge_index, data.edge_attr, data.monomer_labels

        x = self.nn_conv_1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.dropout_1(x)
        
        x = self.nn_conv_2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.dropout_2(x)
        
        x = self.nn_conv_3(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.dropout_3(x)
        
        results_list = []
        
        for i in range(data.num_graphs):
            xi = x[data.batch == i]
            monomer_labels_i = monomer_labels[data.batch == i]
            cc_i = cc[i].item()
            
            num_aminoacid = torch.max(monomer_labels_i).item()
            amino_index_i = get_amino_indices(num_aminoacid)

            xi = scatter(xi, monomer_labels_i, dim=0, reduce="sum")
            
            aminoacids_features_i = aminoacids_features_dict[cc_i]
            combined_ft = torch.cat((xi, aminoacids_features_i), dim=1)
            
            xi = self.nn_gat_1(combined_ft, amino_index_i) 
            
            xi = self.readout(xi)
            results_list.append(xi)
            
        p = torch.cat(results_list, dim=0)
            
        p = self.linear1(p)
        p = F.relu(p)
        p = self.linear2(p)
        p = F.relu(p)
        p = self.linear3(p)
        p = F.relu(p)
        p = self.linear4(p)

            
        return p.view(-1,)


device_info_instance = device_info()
device = device_info_instance.device

def get_amino_indices(num_aminoacid):
    edges = []
    for i in range(num_aminoacid-1):
        edges.append((i, i + 1))
    
    graph_edges = [[x[0] for x in edges], [x[1] for x in edges]]
    
    return torch.tensor(graph_edges, dtype=torch.long, device = device) 

