import unittest
from dinobots.molecules.AtomGraph import SmilesToPolarGraph, NetworkXToMol
from dinobots.data_structures.PolarGraph import PolarGraph
import networkx as nx
from rdkit import Chem

caffeine_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
paraxanthine_smiles = "O=C2Nc1ncn(c1C(=O)N2C)C"


class AtomGraphTesting(unittest.TestCase):
    def test_convert_smiles_to_polargraph(self):
        self.assertEqual(type(SmilesToPolarGraph(caffeine_smiles)), PolarGraph)

    def test_convert_vertices(self):
        self.assertEqual(len(SmilesToPolarGraph(caffeine_smiles).vertices), 14)

    def test_convert_edges(self):
        self.assertEqual(len(SmilesToPolarGraph(caffeine_smiles).edges), 15)

    def test_convert_networkx(self):
        self.assertEqual(type(SmilesToPolarGraph(caffeine_smiles).to_nx()), nx.Graph)

    def test_mcs(self):
        caffeine = SmilesToPolarGraph(caffeine_smiles)
        paraxanthine = SmilesToPolarGraph(paraxanthine_smiles)
        mcs = nx.isomorphism.ISMAGS(caffeine.to_nx(), paraxanthine.to_nx())
        mol = NetworkXToMol(mcs.subgraph)
        self.assertEqual(Chem.MolToSmiles(mol), "Cn1c(=O)[nH]c2ncn(C)c2c1=O")


if __name__ == "__main__":
    unittest.main()
