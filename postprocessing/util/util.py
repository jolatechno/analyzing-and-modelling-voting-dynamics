from matplotlib import pyplot as plt
import numpy as np
import json
import sys


def get_config(filename):
	with open(filename) as raw_json:
	    json_file = json.load(raw_json)

	    if len(sys.argv) > 1:
	    	key = sys.argv[1]
	    	return json_file[key]
	    else:
	    	key = list(json_file.keys())[0]
	    	print(f"no config selected, selecting \"{ key }\"")
	    	return json_file[key]

def get_map_ratio(lon, lat):
	min_lat, max_lat = np.min(lat), np.max(lat)

	lat_width = max_lat - min_lat

	lat_mean  = (max_lat + min_lat)/2
	lon_width = (np.max(lat) - np.min(lat))*np.sin(lat_mean*np.pi/180)

	return lat_width/lon_width

def get_map_dist(lon1, lat1, lon2, lat2):
	return np.sqrt(
		 (lat1 - lat2)**2 +
		((lon1 - lon2)**2)*np.sin(lat1*np.pi/180)*np.sin(lat2*np.pi/180)
	)

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

	map_ratio = get_map_ratio(lon, lat)
	ax.set_aspect(map_ratio)