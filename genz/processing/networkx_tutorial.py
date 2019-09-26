#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" """

import random

import matplotlib.pyplot as plt
import networkx as nx
from numpy import random as nprand

seed = hash("Network Science in Python") % 2 ** 32
nprand.seed(seed)
random.seed(seed)

G = nx.Graph()
G.add_node('A')
G.add_nodes_from(['B', 'C'])
G.add_edge('A', 'B')
G.add_edges_from([('B', 'C'), ('A', 'C')])

plt.figure(figsize=(7.5, 7.5))
nx.draw_networkx(G)
plt.show()
G = nx.karate_club_graph()
karate_pos = nx.spring_layout(G, k=0.3)
nx.draw_networkx(G, karate_pos)
mr_hi = 0
list(G.neighbors(mr_hi))
member_id = 1
(mr_hi, member_id) in G.edges
G.has_edge(mr_hi, member_id)
john_a = 33
(mr_hi, john_a) in G.edges
G.has_edge(mr_hi, john_a)

member_club = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
    0, 0, 0, 0, 1, 1, 0, 0, 1, 0,
    1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1]
for node_id in G.nodes:
    G.nodes[node_id]["club"] = member_club[node_id]

node_colors = [
    '#1f78b4' if G.nodes[v]["club"] == 0
    else '#33a02c' for v in G]
nx.draw_networkx(G, karate_pos, label=True, node_color=node_colors)

# Iterate through all edges
for v, w in G.edges:
    # Compare `club` property of edge endpoints
    # Set edge `internal` property to True if they match
    if G.nodes[v]["club"] == G.nodes[w]["club"]:
        G.edges[v, w]["internal"] = True
    else:
        G.edges[v, w]["internal"] = False

internal = [e for e in G.edges if G.edges[e]["internal"]]
external = [e for e in G.edges if ~G.edges[e]["internal"]]

# Draw nodes and node labels
nx.draw_networkx_nodes(G, karate_pos, node_color=node_colors)
nx.draw_networkx_labels(G, karate_pos)

# Draw internal edges as solid lines
nx.draw_networkx_edges(G, karate_pos, edgelist=internal)
# Draw external edges as dashed lines
nx.draw_networkx_edges(G, karate_pos, edgelist=external, style="dashed")


def tie_strength(G, v, w):
    # Get neighbors of nodes v and w in G
    v_neighbors = set(G.neighbors(v))
    w_neighbors = set(G.neighbors(w))
    # Return size of the set intersection
    return 1 + len(v_neighbors & w_neighbors)


# Calculate weight for each edge
for v, w in G.edges:
    G.edges[v, w]["weight"] = tie_strength(G, v, w)
# Store weights in a list
edge_weights = [G.edges[v, w]["weight"] for v, w in G.edges]
weighted_pos = nx.spring_layout(G, pos=karate_pos, k=0.3, weight="weight")
# Draw network with edge color determined by weight
nx.draw_networkx(
    G, weighted_pos, width=8, node_color=node_colors,
    edge_color=edge_weights, edge_cmap=plt.cm.Blues,
    edge_vmin=0, edge_vmax=6)

# Draw solid/dashed lines on top of internal/external edges
nx.draw_networkx_edges(
    G, weighted_pos, edgelist=internal, edge_color="gray")
nx.draw_networkx_edges(
    G, weighted_pos, edgelist=external, edge_color="gray", style="dashed")

