library(sf)
library(nngeo)
suppressMessages(library(dplyr))

sf_use_s2(FALSE)

compute_knn_from_map <- function(map, n=-1, idxs=NULL) {
	centroid_map <- suppressWarnings(st_point_on_surface(st_geometry(vote_map)))

	map_size <- length(centroid_map)
	if (is.null(idxs)) {
		idxs <- 1:map_size
	}
	if (n <= 0) {
		n <- map_size
	}

	knn_mat <- t(matrix(
			unlist(suppressMessages(st_nn(centroid_map[idxs], centroid_map, k=n, progress=FALSE))),
			nrow=n, ncol=length(idxs)))
	return(knn_mat)
}

geoEncoded_to_map <- function(map, long_field="lon", lat_field="lat") {
	map <- map[!is.na(map[[long_field]]), ]
	map <- map[!is.na(map[[lat_field]]),  ]

	return(suppressWarnings(st_centroid(
		st_as_sf(map, coords=c(long_field, lat_field)))
	))
}

get_distance_matrix <- function(map, idxs=NULL) {
	centroid_map <- suppressWarnings(st_point_on_surface(st_geometry(vote_map)))

	map_size <- length(centroid_map)
	if (is.null(idxs)) {
		idxs <- 1:map_size
	}
	
	distance_mat <- st_distance(centroid_map[idxs], centroid_map)
	return(distance_mat)
}

get_areas <- function(map) {
	return(as.numeric(st_area(map)))
}

intersect_dfs <- function(df1, name1, field_list) {
	return(df1[df1[[name1]] %in% field_list, ])
}
