from graph_tool.all import *  # FYI import * is not a python convention
g = Graph()
ug = Graph(directed=False)
ug = Graph()
ug.set_directed(False)
assert ug.is_directed() == False
v1 = g.add_vertex()
v2 = g.add_vertex()
graph_tool.draw.graph_draw(g, vertex_text=g.vertex_index, vertex_font_size=18,
                           output_size=(200, 200), output="two-nodes.png")
