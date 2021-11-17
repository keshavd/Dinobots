from dinobots.molecules.converters import smiles_to_atom_graph, nx_to_mol
import networkx as nx

caffeine = smiles_to_atom_graph("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
paraxanthine = smiles_to_atom_graph("O=C2Nc1ncn(c1C(=O)N2C)C")

mcs = nx.isomorphism.ISMAGS(caffeine.to_nx(), paraxanthine.to_nx())
nx_to_mol(mcs.subgraph)