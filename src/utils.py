from collections import defaultdict
import numpy as np

import torch
from torch_geometric.data import Data

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG

from sklearn.preprocessing import OneHotEncoder
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils import MeltingTemp as mt


def sequences_geodata(cc, sequence, y, peptide_ft_dict, amino_ft_dict, node_ft_dict, edge_ft_dict, device):
    
    #Atoms
    polymer_id = "PEPTIDE1" 
    helm_notation = peptide_to_helm(sequence, polymer_id)
    molecule = Chem.MolFromHELM(helm_notation)
    
    atomic_number = [atom.GetAtomicNum() for atom in molecule.GetAtoms()]
    aromaticity = [int(atom.GetIsAromatic()) for atom in molecule.GetAtoms()]
    num_bonds = [atom.GetDegree() for atom in molecule.GetAtoms()]
    bonded_hydrogens = [atom.GetTotalNumHs() for atom in molecule.GetAtoms()]
    hybridization = [atom.GetHybridization().real for atom in molecule.GetAtoms()]
    
    node_keys_features = [f"{atomic}_{aromatic}_{bonds}_{hydrogen}_{hybrid}" 
                          for atomic, aromatic, bonds, hydrogen, hybrid 
                          in zip(atomic_number, aromaticity, num_bonds, bonded_hydrogens, hybridization)]
    
    edge_key_features = []
    for bond in molecule.GetBonds():
        bond_type = bond.GetBondTypeAsDouble()
        in_ring = int(bond.IsInRing())
        conjugated = int(bond.GetIsConjugated())
        
        edge_key_features.append(f"{bond_type:.1f}_{in_ring:.1f}_{conjugated:.1f}") 
    
    nodes_features = torch.tensor(np.array([node_ft_dict[x] for x in node_keys_features]), dtype=torch.float32)
    edges_features = torch.tensor(np.array([edge_ft_dict[x] for x in edge_key_features]), dtype=torch.float32)  
    graph_edges = get_edge_indices(molecule)[0]
    
    edges_peptidic = get_edge_indices(molecule)[1]
    edges_nonpeptidic = get_non_peptide_idx(molecule)
    
    
    labels_aminoacid_atoms = get_label_aminoacid_atoms(edges_peptidic, edges_nonpeptidic, molecule)
    
    #Aminoacid:
    aminoacids =get_aminoacids(sequence)
    aminoacids_features = torch.tensor(np.array([amino_ft_dict[amino] for amino in aminoacids]), dtype=torch.float32, device=device)

    #Peptide:
    peptide_features = torch.tensor(np.array([peptide_ft_dict[sequence]]), dtype=torch.float32, device=device)

    geo_dp = Data(x=nodes_features,
                  edge_index=graph_edges, 
                  edge_attr=edges_features, 
                  monomer_labels=labels_aminoacid_atoms,
                  cc = cc,
                  y=y, )
    
    return geo_dp, aminoacids_features, peptide_features


#Atomic Features

#Convert to Helm notation to use Rdkit
def peptide_to_helm(peptide, polymer_id):
    sequence = peptide.replace("(ac)", "[ac].").replace("_", "").replace("1", "").replace("2", "").replace("3", "").replace("4", "")
    sequence_helm = "".join(sequence)
    
    sequence_helm = ''.join([c + '.' if c.isupper() else c for i, c in enumerate(sequence_helm)])
    sequence_helm = sequence_helm.rstrip('.')
    
    sequence_helm = f"{polymer_id}{{{sequence_helm}}}$$$$"
    
    return sequence_helm


    
#Creating the dictionaries based on the dataset information

def get_features(sequence_list):
    
    peptide_features_dict = defaultdict(list)
    peptides_list_helm = []
    aminoacid_list = []
    
    for i, peptide in enumerate(sequence_list):
        peptide_biopython = ProteinAnalysis(peptide)
        #molecular weight
        wt_peptide = peptide_biopython.molecular_weight()
        #aromaticity
        aromaticity_peptide = peptide_biopython.aromaticity()
        #hidrophobicity
        hydrophobicity_peptide = peptide_biopython.gravy()
        #Calculate the charge of a protein at given pH (pH 7)
        net_charge_peptide = peptide_biopython.charge_at_pH(7)
        # isoelectric_point
        p_iso_peptide = peptide_biopython.isoelectric_point()
        #inextability
        inextability_peptide = peptide_biopython.instability_index()
        #length Peptide
        length_peptide = peptide_biopython.length
    
        peptide_properties = [wt_peptide, 
                            aromaticity_peptide,
                            hydrophobicity_peptide,
                            net_charge_peptide,
                            p_iso_peptide,
                            inextability_peptide,
                            length_peptide]
        
        peptide_ft = np.array(peptide_properties)
        peptide_features_dict[peptide] = peptide_ft
        
        #To use Helm notation and RdKit for nodes and bonds features
        polymer_type = "PEPTIDE"  # Tipo de polímero (en este caso, PEPTIDE)
        polymer_id = f"{polymer_type}{i + 1}"
        simple_polymer_helm = peptide_to_helm(peptide, polymer_id)
        peptides_list_helm.append(simple_polymer_helm) #create the list of peptides in Helm notation
        
        #To use for amino acids features
        aminoacid_list.extend(get_aminoacids(peptide))
        
    aminoacid_set = list(set(aminoacid_list))
    amino_features_dict = defaultdict(list)
    
    for amino in aminoacid_set:
        amino_mol = get_molecule(amino)
        amino_biopython = ProteinAnalysis(amino)
        
        #molecular weight:
        wt_amino = Descriptors.MolWt(amino_mol)
        #aromaticity Calculate the aromaticity according to Lobry, 1994:
        aromaticity_amino = amino_biopython.aromaticity()
        #Calculate the GRAVY (Grand Average of Hydropathy) according to Kyte and Doolitle, 1982.
        hydrophobicity_amino = amino_biopython.gravy()
        #Calculate the charge of a protein at given pH (pH 7)
        net_charge_amino = amino_biopython.charge_at_pH(7)
        # isoelectric_point
        p_iso_amino = amino_biopython.isoelectric_point()
        #coeficiente de partición octanol-agua. Un LogP más alto indica una molécula más hidrofóbica (menos polar), mientras que un LogP más bajo indica una molécula más hidrofílica (más polar).
        logp_amino = Crippen.MolLogP(amino_mol)
        #number of atoms
        atoms_amino = float(amino_mol.GetNumAtoms())
        
        #all together
        amino_properties = [wt_amino,
                            aromaticity_amino,
                            hydrophobicity_amino,
                            net_charge_amino,
                            p_iso_amino,
                            logp_amino, 
                            atoms_amino]
        
        amino_ft = np.array(amino_properties)
        amino_features_dict[amino] = amino_ft
        
  
    #nodes
    atomic_number = []
    aromaticity = []
    num_bonds = []
    bonded_hydrogens = []
    hybridization = []
    
    #edges
    bond_type = []
    in_ring = []
    conjugated = []
    
    for helm in peptides_list_helm:
        molecule = Chem.MolFromHELM(helm)
        
        atomic_number.extend([atom.GetAtomicNum() for atom in molecule.GetAtoms()])
        aromaticity.extend([int(atom.GetIsAromatic()) for atom in molecule.GetAtoms()])
        num_bonds.extend([atom.GetDegree() for atom in molecule.GetAtoms()])
        bonded_hydrogens.extend([atom.GetTotalNumHs() for atom in molecule.GetAtoms()])
        hybridization.extend([atom.GetHybridization().real for atom in molecule.GetAtoms()])
        
        for bond in molecule.GetBonds():
            bond_type.extend([bond.GetBondTypeAsDouble()])
            in_ring.extend([int(bond.IsInRing())])
            conjugated.extend([int(bond.GetIsConjugated())])
            
    #nodes
    atomic_set = list(set(atomic_number))
    codificador_atomic = OneHotEncoder()
    codificador_atomic.fit(np.array(atomic_set).reshape(-1,1))
    
    aromatic_set = list(set(aromaticity))
    codificador_aromatic = OneHotEncoder()
    codificador_aromatic.fit(np.array(aromatic_set).reshape(-1,1))
    
    bonds_set = list(set(num_bonds))
    codificador_bonds = OneHotEncoder()
    codificador_bonds.fit(np.array(bonds_set).reshape(-1,1))
    
    hydrogen_set = list(set(bonded_hydrogens))
    codificador_hydrogen = OneHotEncoder()
    codificador_hydrogen.fit(np.array(hydrogen_set).reshape(-1,1))   
    
    hybrid_set = list(set(hybridization))
    codificador_hybrid = OneHotEncoder()
    codificador_hybrid.fit(np.array(hybrid_set).reshape(-1,1))
    
    #edges
    bond_type_set = list(set(bond_type))
    codificador_bond_type = OneHotEncoder()
    codificador_bond_type.fit(np.array(bond_type_set).reshape(-1,1))
    
    in_ring_set = list(set(in_ring))
    codificador_in_ring= OneHotEncoder()
    codificador_in_ring.fit(np.array(in_ring_set).reshape(-1,1))
    
    conjugated_set = list(set(conjugated))
    codificador_conjugated= OneHotEncoder()
    codificador_conjugated.fit(np.array(conjugated_set).reshape(-1,1))

    node_features_dict = defaultdict(list)
    edge_features_dict = defaultdict(list)
    
    for atom, aromatic, bonds, hydrogen, hybrid in zip(atomic_number, aromaticity, num_bonds, bonded_hydrogens, hybridization):
        node_key_features_combined = f"{atom}_{aromatic}_{bonds}_{hydrogen}_{hybrid}"
        
        atomic_feature  = codificador_atomic.transform([[atom]]).toarray()[0]
        aromatic_feature = codificador_aromatic.transform([[aromatic]]).toarray()[0]
        bonds_feature = codificador_bonds.transform([[bonds]]).toarray()[0]
        hydrogen_feature = codificador_hydrogen.transform([[hydrogen]]).toarray()[0]
        hybrid_feature = codificador_hybrid.transform([[hybrid]]).toarray()[0]
        
        feature_node = np.concatenate((atomic_feature, aromatic_feature, bonds_feature, hydrogen_feature, hybrid_feature))
        node_features_dict[node_key_features_combined] = feature_node
    
    for bond, ring, conjugat in zip(bond_type, in_ring, conjugated):
        edge_key_features_combined = f"{bond:.1f}_{ring:.1f}_{conjugat:.1f}" 
        
        bond_feature = codificador_bond_type.transform([[bond]]).toarray()[0]
        ring_feature = codificador_in_ring.transform([[ring]]).toarray()[0]
        conjugated_feature = codificador_conjugated.transform([[conjugat]]).toarray()[0] 
            
        feature_edge = np.concatenate((bond_feature, ring_feature, conjugated_feature))
        edge_features_dict[edge_key_features_combined] = feature_edge
        
    
    return  peptide_features_dict, amino_features_dict, node_features_dict, edge_features_dict

#Getting the edges to the molecule
def get_edge_indices(molecule):
    edges_peptidic=[]
    for bond in molecule.GetBonds():
        edges_peptidic.append((bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()))
        
    graph_edges = [[x[0] for x in edges_peptidic],[x[1] for x in edges_peptidic]]
    
    return torch.tensor(graph_edges, dtype=torch.long), edges_peptidic

#Getting the edges of the peptidic bonds
def get_non_peptide_idx(molecule):
    edges_nonpeptidic= []
    for bond in molecule.GetBonds():
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        atomic_num1 = atom1.GetAtomicNum()
        atomic_num2 = atom2.GetAtomicNum()
        neighbors_1 = list(atom1.GetNeighbors())
        neighbors_1_list = [neighbor.GetAtomicNum() for neighbor in neighbors_1]
        neighbors_2 = list(atom2.GetNeighbors())
        neighbors_2_list = [neighbor.GetAtomicNum() for neighbor in neighbors_2]  
        hibrid_1 = str(atom1.GetHybridization())
        hibrid_2 = str(atom2.GetHybridization())
        hidrog_1 = atom1.GetTotalNumHs()
        hidrog_2 = atom2.GetTotalNumHs()
        bond_type = str(bond.GetBondType())
        conjugated = str(bond.GetIsConjugated())
        #print(atom1.GetAtomicNum(), atom2.GetAtomicNum())

        if not(atomic_num1 == 6 and #C
            atomic_num2 == 7 and #N
            8 in neighbors_1_list and #O is neighbor of C
            hibrid_1 == 'SP2' and
            hibrid_2 == 'SP2' and
            hidrog_1 == 0 and  #C
            (hidrog_2 == 1 or  #N
            hidrog_2 == 0 )and  #N in Proline
            conjugated == 'True' and
            bond_type == 'SINGLE'): #ROC---NHR
            if not(atomic_num1 == 7 and   #N
                    atomic_num2 == 6 and  #C
                    8 in neighbors_2_list and #O is neighbor of C
                    hibrid_1 == 'SP2' and 
                    hibrid_2 == 'SP2' and
                    (hidrog_2 == 1 or  #N
                    hidrog_2 == 0 )and  #N in Proline
                    hidrog_2 == 0 and  #C
                    conjugated == 'True' and
                    bond_type == 'SINGLE'):
                edges_nonpeptidic.append((bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()))
                
    
    
    return  edges_nonpeptidic


#Getting the label of the atoms based on the aminoacids e.g: ABC --> [1,1,1,2,2,2,2,2,3,3,3,3,3,3,3]
def get_label_aminoacid_atoms(edges_peptidic, edges_nonpeptidic, molecule):
    
    set_with = set(edges_peptidic)
    set_witout = set(edges_nonpeptidic)
    tuplas_diferentes = list( set_with.symmetric_difference(set_witout))
    lista_atoms = [elemento for tupla in tuplas_diferentes for elemento in tupla]
    
    break_idx = []
    for tupla in tuplas_diferentes:
        atom_1, atom_2 = tupla
        bond = molecule.GetBondBetweenAtoms(atom_1, atom_2)
        break_idx.append(bond.GetIdx())
        
    mol_f = Chem.FragmentOnBonds(molecule, break_idx, addDummies=False)
    fragmentos = list(Chem.GetMolFrags(mol_f, asMols=True))
    peptide_idx = np.empty(0)
    
    
    for i, fragme in enumerate(fragmentos):
        atoms_in_fragme = fragme.GetNumAtoms()
        idx_vector = np.ones(atoms_in_fragme)*(i)
        peptide_idx = np.concatenate((peptide_idx, idx_vector ))
        
    
    return  torch.tensor(peptide_idx.tolist(), dtype=torch.long)
    

#Aminoacids Features

#To create aminoacid dictionary
def split_sequence_for_Helm(peptide):
    sequence = peptide.replace("(ac)", "[ac].").replace("_", "").replace("1", "").replace("2", "").replace("3", "").replace("4", "")
    sequence = "".join(sequence)
    
    sequence = ''.join([c + '.' if c.isupper() else c for i, c in enumerate(sequence)])
    sequence = sequence.rstrip('.')
    
    split_list = []
    temp = ''
    skip_next = False

    for i in range(len(sequence)):
        if skip_next:
            skip_next = False
            continue

        if sequence[i] == ']':
            temp += sequence[i]
            if i < len(sequence) - 1 and sequence[i + 1] == '.':
                temp += '.'
                skip_next = True
        elif sequence[i] == '.':
            if temp:
                split_list.append(temp)
                temp = ''
        else:
            temp += sequence[i]

    if temp:
        split_list.append(temp)
        
    return split_list


#Gives the aminoacids given a peptide
def get_aminoacids(sequence):
    aminoacids_list = []
    
    for i, sequence in enumerate(sequence):
        aminoacids_list.extend(split_sequence_for_Helm(sequence))
    
    return aminoacids_list

#Returns tmolecule from an aminoacid
def get_molecule(amino):
    polymer_id = "PEPTIDE1" 
    helm_notation = peptide_to_helm(amino, polymer_id)
    molecule = Chem.MolFromHELM(helm_notation)
    
    return molecule 