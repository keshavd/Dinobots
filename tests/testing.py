from dinobots.molecules.converters import smiles_to_atom_graph, nx_to_mol
import networkx as nx

caffeine = smiles_to_atom_graph("CN1C=NC2=C1C(=O)N(C(=O)N2C)C").to_nx()
paraxanthine = smiles_to_atom_graph("O=C2Nc1ncn(c1C(=O)N2C)C").to_nx()

mcs = nx.isomorphism.ISMAGS(caffeine, paraxanthine)
nx_to_mol(mcs.subgraph)