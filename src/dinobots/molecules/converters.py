from rdkit import Chem
import polars as pl
from dinobots.data_structures.PolarGraph import PolarGraph


def smiles_to_atom_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    vertices = [
        dict(
            ids=atom.GetIdx(),
            atomic_num=atom.GetAtomicNum(),
            formal_charge=atom.GetFormalCharge(),
            chiral_tag=atom.GetChiralTag().name,
            hybridization=atom.GetHybridization().name,
            num_explicit_hs=atom.GetNumExplicitHs(),
            is_aromatic=bool(atom.GetIsAromatic()),
        )
        for atom in mol.GetAtoms()
    ]
    vertices_df = pl.DataFrame(vertices)
    edges = [
        dict(
            src=bond.GetBeginAtomIdx(),
            dst=bond.GetEndAtomIdx(),
            bond_type=bond.GetBondType().name,
        )
        for bond in mol.GetBonds()
    ]
    edges_df = pl.DataFrame(edges)
    pg = PolarGraph(v=vertices_df, e=edges_df)
    return pg
