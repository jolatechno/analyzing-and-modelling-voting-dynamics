import matplotlib.pyplot as plt
import numpy as np

def plot_graph_from_scratch(neighbors, lon, lat, colors=None, ax=None):
	target = plt
	if ax is not None:
		target = plt

	if colors is None:
		colors = "k"
	
	for node,node_neighbors in enumerate(neighbors):
		for neighbor in node_neighbors:
			target.plot([lon[node], lon[neighbor]], [lat[node], lat[neighbor]],
				        "k-", linewidth=0.8)

	target.scatter(lon, lat, c=colors)