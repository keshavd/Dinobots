import polars as pl
import networkx as nx


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

    def __init__(self, v: pl.DataFrame, e: pl.DataFrame):
        self._vertices = v
        self._edges = e

        self.ID = "ids"
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

    def to_nx(self):
        """Convert to NetworkX Graph Object"""
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

    def to_gf(self):
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

    def to_pyg(self):
        pass
