import polars as pl
import pandas as pd


class PolarGraph:
    """
    Represents a graph with vertices and edges stored as DataFrames.
    :param v:  :class:`DataFrame` holding vertex information.
               Must contain a column named "id" that stores unique
               vertex IDs.
    :param e:  :class:`DataFrame` holding edge information.
               Must contain two columns "src" and "dst" storing source
               vertex IDs and destination vertex IDs of edges, respectively.

    (Borrowed from Spark's GraphFrames)

    """

    def __init__(
        self, v: pl.DataFrame, e: pl.DataFrame, vertex_encoders=None, edge_encoders=None
    ):
        if edge_encoders is None:
            edge_encoders = {}
        if vertex_encoders is None:
            vertex_encoders = {}
        self._vertices = v
        self._edges = e
        self.vertex_encoders = vertex_encoders
        self.edge_encoders = edge_encoders
        self.ID = "id"
        self.SRC = "src"
        self.DST = "dst"

        # Check that provided DataFrames contain required columns
        if self.ID not in v.columns:
            raise ValueError(
                "Vertex ID column {} missing from vertex DataFrame, which has columns: {}".format(
                    self.ID, ",".join(v.columns)
                )
            )
        if self.SRC not in e.columns:
            raise ValueError(
                "Source vertex ID column {} missing from edge DataFrame, which has columns: {}".format(
                    self.SRC, ",".join(e.columns)
                )
            )
        if self.DST not in e.columns:
            raise ValueError(
                "Destination vertex ID column {} missing from edge DataFrame, which has columns: {}".format(
                    self.DST, ",".join(e.columns)
                )
            )

    @property
    def vertices(self):
        """
        :class:`DataFrame` holding vertex information, with unique column "id"
        for vertex IDs.
        """
        return self._vertices

    @property
    def edges(self):
        """
        :class:`DataFrame` holding edge information, with unique columns "src" and
        "dst" storing source vertex IDs and destination vertex IDs of edges,
        respectively.
        """
        return self._edges

    # Convert To Data Types

    def to_nx(self):
        """Convert to NetworkX Graph Object"""
        import networkx as nx

        G = nx.Graph()
        vertex_cols = self.vertices.rename({self.ID: "node_for_adding"}).columns
        for row in self.vertices.rows():
            G.add_node(**dict(zip(vertex_cols, row)))
        edge_cols = self.edges.rename(
            {self.SRC: "u_of_edge", self.DST: "v_of_edge"}
        ).columns
        for edge in self.edges.rows():
            G.add_edge(**dict(zip(edge_cols, edge)))
        return G

    def to_gf(self, spark):
        """Convert PolarFrame to PySpark's GraphFrame

        Only usable in PySpark Environment
        """
        from graphframes import GraphFrame

        vdf = self.vertices
        edf = self.edges
        if type(vdf) == pl.DataFrame:
            vdf = vdf.to_pandas()
        if type(edf) == pl.DataFrame:
            edf = edf.to_pandas()
        vdf = spark.createDataFrame(vdf)
        edf = spark.createDataFrame(edf)
        return GraphFrame(vdf, edf)

    def to_cugraph(self):
        pass

    def to_pyg(
        self,
        vertex_attributes=None,
        edge_attributes=None,
        label_nodes=False,
        node_label_key=None,
        graph_label=None,
        node_pos_key=None,
    ):
        from torch_geometric.data import Data
        import torch
        from sklearn.preprocessing import LabelBinarizer
        import numpy as np

        if vertex_attributes is None:
            vertex_attributes = []

        if edge_attributes is None:
            edge_attributes = []

        # Node Attributes
        x = []
        for vertex_attribute in vertex_attributes:
            attr = self.vertices[vertex_attribute].to_numpy()
            enc = self.vertex_encoders.get(vertex_attribute)
            if enc is None:
                enc = LabelBinarizer()
                self.vertex_encoders[vertex_attribute] = enc
                enc.fit(attr)
            encoded_data = enc.transform(attr)
            x.append(encoded_data)
        if len(x) > 0:
            x = np.concatenate(x, axis=0)
        else:
            x = None

        # Edges
        vertex_lookup = {v: idx for idx, v in enumerate(self.vertices.id)}
        src = self.edges.src.apply(lambda x: vertex_lookup[x]).to_numpy()
        dst = self.edges.dst.apply(lambda x: vertex_lookup[x]).to_numpy()
        edge_index = np.concatenate([[src], [dst]], axis=0)

        # Edge Attributes
        edge_attr = (
            self.edges[[*edge_attributes]].to_numpy()
            if len(edge_attributes) > 0
            else None
        )

        # Labels
        if label_nodes and node_label_key is not None:
            y = self.vertices[[node_label_key]].to_numpy()
        elif graph_label is not None:
            y = graph_label
        else:
            y = None

        # Node position matrix
        if node_pos_key:
            pos = self.vertices[[node_pos_key]].to_numpy()
        else:
            pos = None

        return Data(
            x=torch.tensor(x, dtype=torch.float32) if x is not None else None,
            edge_index=torch.tensor(edge_index, dtype=torch.long)
            if edge_index is not None
            else None,
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32)
            if edge_attr is not None
            else None,
            y=torch.tensor(y, dtype=torch.float32) if y is not None else None,
            pos=torch.tensor(pos, dtype=torch.float32) if pos is not None else None,
        )

    # Convert From Data Types

    @staticmethod
    def from_nx(G):
        all_edges = G.edges().items()
        edge_dicts = []
        for k, v in all_edges:
            edge_dict = {"src": k[0], "dst": k[1]}
            for vk, vv in v.items():
                if isinstance(vv, set):  # recast sets
                    edge_dict[vk] = list(vv)
                else:
                    edge_dict[vk] = vv
            edge_dicts.append(edge_dict)
        if len(edge_dicts) > 0:
            edges_df = pl.DataFrame(pd.DataFrame(edge_dicts))
        else:
            edges_df = pl.DataFrame()
        all_nodes = G.nodes().items()
        node_dicts = []
        for k, v in all_nodes:
            node_dict = {"id": k}
            for vk, vv in v.items():
                if isinstance(vv, set):
                    node_dict[vk] = list(vv)
                else:
                    node_dict[vk] = vv
            node_dicts.append(node_dict)
        if len(node_dicts) > 0:
            node_df = pl.DataFrame(pd.DataFrame(node_dicts))
        else:
            node_df = pl.DataFrame()
        return PolarGraph(v=node_df, e=edges_df)
