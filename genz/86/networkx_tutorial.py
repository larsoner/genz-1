#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" """

import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.connectivity as nxcon
from networkx import bipartite
from numpy import random as nprand


def tie_strength(G, v, w):
    # Get neighbors of nodes v and w in G
    v_neighbors = set(G.neighbors(v))
    w_neighbors = set(G.neighbors(w))
    # Return size of the set intersection
    return 1 + len(v_neighbors & w_neighbors)


def path_length_histogram(G, title=None):
    """distribution of shortest path length"""
    # Find path lengths
    length_source_target = dict(nx.shortest_path_length(G))
    # Convert dict of dicts to flat list
    all_shortest = sum([
        list(length_target.values())
        for length_target
        in length_source_target.values()],
        [])
    # Calculate integer bins
    high = max(all_shortest)
    bins = [-0.5 + i for i in range(high + 2)]
    # Plot histogram
    plt.hist(all_shortest, bins=bins, rwidth=0.8)
    plt.title(title)
    plt.xlabel("Distance")
    plt.ylabel("Count")


def centrality_histogram(x, title=None):
    """plots histograms of the eigenvector centralities"""
    plt.hist(x, density=True)
    plt.title(title)
    plt.xlabel("Centrality")
    plt.ylabel("Density")


def entropy(x):
    """return the entropy of a list of numbers.
    same as scipy.stats.entropy
    """
    # Normalize
    total = sum(x)
    x = [xi / total for xi in x]
    H = sum([-xi * math.log2(xi) for xi in x])
    return H


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
G_karate = nx.karate_club_graph()
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

# Create empty affiliation network and list of people
B = nx.Graph()
people = set()
# Load data file into network
data_dir = Path(
    '/home/ktavabi/Dropbox/work/Network_Sci_Networkx_quick') / 'data'
with open(data_dir / 'crossley2012' / '50_ALL_2M.csv') as f:
    # Parse header
    events = next(f).strip().split(",")[1:]
    # Parse rows
    for row in f:
        parts = row.strip().split(",")
        person = parts[0]
        people.add(person)
        for j, value in enumerate(parts[1:]):
            if value != "0":
                B.add_edge(person, events[j], weight=int(value))
# Project into person-person co-affiliation network
Gs = bipartite.projected_graph(B, people)

#####################################
# Small Scale – Nodes and Centrality#
#####################################
betweenness = nx.betweenness_centrality(Gs, normalized=False)
sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[0:10]
eigenvector = nx.eigenvector_centrality(Gs)
sorted(eigenvector.items(), key=lambda x: x[1], reverse=True)[0:10]
closeness = nx.closeness_centrality(Gs)
sorted(closeness.items(), key=lambda x: x[1], reverse=True)[0:10]
triangles = nx.triangles(Gs)
sorted(triangles.items(), key=lambda x: x[1], reverse=True)[0:10]
clustering = nx.clustering(Gs)
[(x, clustering[x]) for x in
 sorted(people, key=lambda x: eigenvector[x], reverse=True)[0:10]]

########################################
# Large Scale – characterizing networks#
########################################
# Load Germany electrical grid
with open(data_dir / 'mureddu2016' / '0.2' / 'branches.csv', 'rb') as f:
    next(f)
    # Read edgelist format
    G_electric = nx.read_edgelist(
        f,
        delimiter="\t",
        create_using=nx.Graph,
        data=[('X', float), ('Pmax', float)])

G_internet = nx.read_graphml(data_dir / 'UAITZ' / 'Geant2012.graphml')

# Create a figure
plt.figure(figsize=(7.5, 2.75))
# Plot networks
plt.subplot(1, 3, 1)
plt.title("Karate")
nx.draw_networkx(G_karate, node_size=0, with_labels=False)
plt.subplot(1, 3, 2)
plt.title("Electric")
nx.draw_networkx(G_electric, node_size=0, with_labels=False)
plt.subplot(1, 3, 3)
plt.title("Internet")
nx.draw_networkx(G_internet, node_size=0, with_labels=False)
# Adjust layout
plt.tight_layout()

# Create figure
plt.figure(figsize=(7.5, 2.5))
# Plot path length histograms
plt.subplot(1, 3, 1)
path_length_histogram(G_karate, title="Karate")
plt.subplot(1, 3, 2)
path_length_histogram(G_electric, title="Electric")
plt.subplot(1, 3, 3)
path_length_histogram(G_internet, title="Internet")
# Adjust layout
plt.tight_layout()

# characteristic length - small worldness (short paths)
nx.average_shortest_path_length(G_karate)
nx.average_shortest_path_length(G_electric)
nx.average_shortest_path_length(G_internet)

# diameter
nx.diameter(G_karate)
nx.diameter(G_karate)
nx.diameter(G_karate)

# transitivity
nx.transitivity(G_karate)
nx.transitivity(G_electric)
nx.transitivity(G_internet)

# Global clustering coeff
nx.average_clustering(G_karate)
nx.average_clustering(G_karate)
nx.average_clustering(G_karate)

# density - crudest measure of resilience
nx.density(G_karate)
nx.density(G_karate)
nx.density(G_karate)

# minimum cut - min nodes/edges removed to separate into two unconnected parts.
nxcon.minimum_st_node_cut(G_karate, mr_hi, john_a)

# connectivity
nxcon.minimum_node_cut(G_karate)
nxcon.minimum_node_cut(G_electric)
nxcon.minimum_node_cut(G_internet)
nx.node_connectivity(G_karate)  # size of the smallest min-cut to disconnect
nx.node_connectivity(G_electric)
nx.node_connectivity(G_internet)

# connectivity over all nodes
nx.average_node_connectivity(G_karate)
nx.average_node_connectivity(G_electric)
nx.average_node_connectivity(G_internet)


# centralization - how much of centrality is concentrated in one or a few nodes.
plt.figure(figsize=(7.5, 2.5))
# Calculate centralities for each example and plot
plt.subplot(1, 3, 1)
centrality_histogram(
    nx.eigenvector_centrality(G_karate).values(), title="Karate")
plt.subplot(1, 3, 2)
centrality_histogram(
    nx.eigenvector_centrality(G_electric, max_iter=1000).values(),
    title="Electric")
plt.subplot(1, 3, 3)
centrality_histogram(
    nx.eigenvector_centrality(G_internet).values(), title="Internet")
plt.tight_layout()

# entropy
entropy(nx.eigenvector_centrality(G_karate).values())
entropy(nx.eigenvector_centrality(G_electric, max_iter=1000).values())
entropy(nx.eigenvector_centrality(G_internet).values())

########################################
# Meso Scale – community interrelations#
########################################
