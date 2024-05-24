import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

options = {
	"with_labels": False,
	"node_size": 3,
	"node_color": "black",
	"edgecolors": "black",
	"linewidths": 2,
	"width": 3,
}

def pos_from_lat_lon(lat, lon):
	pos = {}
	for node in range(len(lat)):
		pos[node] = (lon[node], lat[node])
	return pos

def plot_graph(graph, graph_for_pos=None, pos=None, lon=None, lat=None, colors=None):
	if pos is None:
		if graph_for_pos is None:
			graph_for_pos = graph

		if lon is not None and lat is not None:
			pos = pos_from_lat_lon(lat, lon)
		else:
			pos = nx.nx_agraph.graphviz_layout(graph_for_pos, prog="neato")

	if colors is None:
		C = (graph.subgraph(c) for c in nx.connected_components(graph))
		for g in C:
			c = [np.random.random()] * nx.number_of_nodes(g) # random color...
			nx.draw(g, pos, node_size=40, node_color=c, vmin=0.0, vmax=1.0, with_labels=False)

	else:
		nx.draw(graph, pos, node_size=40, node_color=colors, vmin=0.0, vmax=1.0, with_labels=False)
	
	plt.gca().set_aspect('equal', adjustable='box')