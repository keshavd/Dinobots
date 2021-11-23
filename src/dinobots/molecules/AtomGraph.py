from rdkit import Chem
import polars as pl
from dinobots.data_structures.PolarGraph import PolarGraph


def SmilesToPolarGraph(smiles):
    """Convert SMILES string to PolarGraph"""
    mol = Chem.MolFromSmiles(smiles)
    vertices = [
        dict(
            id=atom.GetIdx(),
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


def NetworkXToMol(G):
    """Convert NetworkX Graph generated from PolarGraph back to RDKit Mol"""
    import networkx as nx
    from rdkit.Chem.rdchem import ChiralType, HybridizationType, BondType

    mol = Chem.RWMol()
    atomic_nums = nx.get_node_attributes(G, "atomic_num")
    chiral_tags = {
        k: ChiralType.names[x]
        for k, x in nx.get_node_attributes(G, "chiral_tag").items()
    }
    formal_charges = nx.get_node_attributes(G, "formal_charge")
    node_is_aromatics = nx.get_node_attributes(G, "is_aromatic")
    node_hybridizations = {
        k: HybridizationType.names[x]
        for k, x in nx.get_node_attributes(G, "hybridization").items()
    }
    num_explicit_hss = nx.get_node_attributes(G, "num_explicit_hs")
    node_to_idx = {}
    for node in G.nodes():
        a = Chem.Atom(atomic_nums[node])
        a.SetChiralTag(chiral_tags[node])
        a.SetFormalCharge(formal_charges[node])
        a.SetIsAromatic(node_is_aromatics[node])
        a.SetHybridization(node_hybridizations[node])
        a.SetNumExplicitHs(num_explicit_hss[node])
        idx = mol.AddAtom(a)
        node_to_idx[node] = idx
    bond_types = {
        k: BondType.names[x] for k, x in nx.get_edge_attributes(G, "bond_type").items()
    }
    for edge in G.edges():
        first, second = edge
        ifirst = node_to_idx[first]
        isecond = node_to_idx[second]
        bond_type = bond_types[first, second]
        mol.AddBond(ifirst, isecond, bond_type)
    Chem.SanitizeMol(mol)
    return mol