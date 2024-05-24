library(stringr)
suppressMessages(library(weights))

source("source/util/readWrite_util.R")
source("source/util/algo.R")
source("source/util/map_util.R")


root        <- "../../data/"
vote_file    <- "resultats-par-niveau-burvot-t1-france-entiere.xlsx"
adress_file  <- "table-bv-reu.geocoded.csv"
geoVote_file <- "bureaux-de-votes.geojson"
iris_file    <- "CONTOURS-IRIS.shp"

ref_crs <- st_crs(st_read(paste(root, geoVote_file, sep=""), quiet=TRUE))

output_root <- "../computation/output/"
config_file <- "config.json"
config <- read_config(paste(output_root, config_file, sep=""))

compute_knn <- config$preprocessing$compute_knn

N_neighbors        <- config$preprocessing$N_neighbors
num_commune_sample <- config$preprocessing$num_commune_sample
list_departement   <- config$preprocessing$list_departement


lat_min <- config$preprocessing$lat_lim[1]
lat_max <- config$preprocessing$lat_lim[2]
lon_min <- config$preprocessing$lon_lim[1]
lon_max <- config$preprocessing$lon_lim[2]

output_file <- paste(output_root, config$preprocessed_file, sep="")

invisible(try(file.remove(output_file)))
h5createFile(output_file)

# fix the seed:
set.seed(639245)


##################################################
# Read files
##################################################


print(paste("Reading file's colnames '", vote_file, "'...", sep=""))
vote_data_colnames <- colnames(read_excel(paste(root, vote_file, sep=""), n_max=1))
candidate_list <- unique(unlist(str_extract_all(vote_data_colnames, "(?<=_)[A-ZÉÈ_]*")))
candidate_list <- candidate_list[lengths(candidate_list) != 0]


##################################################
# Process geocoded adresses 
# from: https://adresse.data.gouv.fr/csv
##################################################


vote_map <- read_file_or_compute(
	function() { stop() }, function(data) {},
	#read_map(merged_map_file), write_map(merged_map_file),
	function() {
		## Read vote adress:
		print(paste("Reading file '", adress_file, "'...", sep=""))
		adress_data <- read_csv(file=paste(root, adress_file, sep=""), show_col_types=FALSE)
		adress_data <- adress_data[adress_data$result_status == "ok", ]

		# cut down to departments:
		adress_data_regParis <- adress_data[substr(adress_data$cp_reu, 1, 2) %in% list_departement, ]

		adress_data_regParis <- adress_data_regParis[which(adress_data_regParis$longitude<lon_max), ]
		adress_data_regParis <- adress_data_regParis[which(adress_data_regParis$longitude>lon_min), ]
		adress_data_regParis <- adress_data_regParis[which(adress_data_regParis$latitude <lat_max), ]
		adress_data_regParis <- adress_data_regParis[which(adress_data_regParis$latitude >lat_min), ]

		print("Converting to map...")
		geo_encoded_sf <- geoEncoded_to_map(adress_data_regParis,
			long_field="longitude", lat_field="latitude")
		geo_encoded_sf <- st_set_crs(geo_encoded_sf, ref_crs)

		return(geo_encoded_sf)
	})


##################################################
# Compute vote proportion
##################################################


field_names <- paste("Voix_", candidate_list, sep="")
vote_map <- cbind(vote_map, 
	read_file_or_compute(
		function() { stop() }, function(data) {},
		#read_vector_or_df_or_matrix(vote_prop_file), write_vector_or_df_or_matrix(vote_prop_file),
		function() {
			## Read vote data:
			print(paste("Reading file '", vote_file, "'...", sep=""))
			vote_data <- read_excel(paste(root, vote_file, sep=""))

			vote_data$code_commune <- paste(vote_data$"Code du département", vote_data$"Code de la commune", sep="")
			vote_data$id_brut_miom <- paste(vote_data$code_commune, vote_data$"Code du b.vote", sep="_")


			print("Computing vote proportions...")
			return(compute_accumulated_pop(vote_data,
				"id_brut_miom", vote_map$"id_brut_miom",
				field_names))
		})
	)
vote_map <- clear_df_NAs(vote_map, field_names)

total_vec <- compute_total_accumulated_pop(vote_map, field_names)
total     <- sum(total_vec)
prop_vec  <- total_vec/total

vote_map <- cbind(vote_map, compute_prop_from_pop(vote_map, field_names))


h5createGroup(output_file, "geo_data")
coordinates <- st_coordinates(vote_map)
h5write(coordinates[,1], output_file, "geo_data/lon")
h5write(coordinates[,2], output_file, "geo_data/lat")

h5createGroup(output_file, "demo_data")
h5write(as.numeric(unlist(vote_map[["total_pop"]])), output_file, "demo_data/voter_population")

h5createGroup(output_file, "vote_data")
for (field in field_names) {
	h5write(as.numeric(unlist(vote_map[[     field]])), output_file, paste("vote_data/",      field, sep=""))
	prop_field <- paste("PROP_", field, sep="")
	h5write(as.numeric(unlist(vote_map[[prop_field]])), output_file, paste("vote_data/", prop_field, sep=""))
}


##################################################
# Compute nearest neighbors
##################################################


if (!compute_knn) {
	quit()
}

num_commune_sample    <- if (num_commune_sample > 0) min(num_commune_sample, nrow(vote_map)) else nrow(vote_map)
nearest_neighbors_idx <- read_file_or_compute(
	function() { stop() }, function(data) {},
	#read_vector_or_df_or_matrix(knn_file), write_vector_or_df_or_matrix(knn_file),
	function() {
		commune_name_sample_list <- sort(c(sample(1:nrow(vote_map), size=num_commune_sample, replace=FALSE)))

		print("Computing k closests neighbors...")
		return(compute_knn_from_map(vote_map, -1, commune_name_sample_list))
	})


##################################################
# Write network shape
##################################################


h5createGroup(output_file, "network")

neighbors               <- c()
neighbors_begin_end_idx <- as.integer(seq(0, N_neighbors*num_commune_sample, by=N_neighbors))
for (node_idx in 1:num_commune_sample) {
	offset <- 0
	for (ineighbor in 1:N_neighbors) {
		while (nearest_neighbors_idx[node_idx, ineighbor+offset] == node_idx) {
			offset <- offset + 1
		}
		neighbors[neighbors_begin_end_idx[node_idx] + ineighbor] = as.integer(nearest_neighbors_idx[node_idx, ineighbor+offset] - 1)
	}
}

h5write(neighbors,               output_file, "network/neighbors")
h5write(neighbors_begin_end_idx, output_file, "network/neighbors_begin_end_idx")