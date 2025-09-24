#library(foreach)
#library(parallel)
library(pbmcapply)

accumulate <- function(x) {
	return(cumsum(x))
}
colCumSums <- function(x) {
	return(matrix(accumulate(rbind(x,-colSums(x))), ncol=ncol(x))[1:nrow(x),])
}
rowCumSums <- function(x) {
	return(t(colCumSums(t(x))))
}

clear_df_NAs <- function(df, field_names_list=NULL) {
	if (is.null(field_names_list)) {
		field_names_list <- colnames(df)
	}

	for (field_name in field_names_list) {
		df <- df[!is.na(df[[field_name]]), ]
	}
	return(df)
}

compute_accumulated_pop <- function(data, field_name, field_value_list, field_names_list, threadPerCore=0.5) {
	output_data <- data.frame(mat=rep(0, length(field_value_list)))
	for (name in field_names_list) {
		output_data[[name]] <- rep(0, length(field_value_list))
	}

	accumulation_matrix <- t(matrix(unlist(
		pbmclapply(1:length(field_value_list), function(i) {coincidence_value <- field_value_list[i]
			coincidence_value <- field_value_list[i]
			local_list <- data[data[[field_name]] == coincidence_value, ]

			return(unlist(
				lapply(field_names_list, function(name) {
					return(sum(local_list[[name]]) + 1e-6) # + 1e-6 to avoid dividing by 0 later on
				})))
		}, mc.cores=round(detectCores()*threadPerCore), ignore.interactive=TRUE)),
		nrow=length(field_names_list), ncol=length(field_value_list)))

	output_data           <- data.frame(accumulation_matrix)
	colnames(output_data) <- field_names_list

	return(output_data)
}

compute_total_accumulated_pop <- function(data, field_names_list) {
	total_vec <- rep(0, length(field_names_list))
	for (i in 1:length(field_names_list)) {
		total_vec[i] <- sum(data[[field_names_list[i]]])
	}

	return(total_vec)
}

compute_prop_from_pop <- function(data, field_names_list) {
	output_data <- data.frame(mat=rep(0, length(data[[field_names_list[1]]])))

	prop_names <- paste("PROP_", field_names_list, sep="")
	output_data$total_pop <- rep(0, length(data[[field_names_list[1]]]))

	for (name in field_names_list) {
		output_data$total_pop <- output_data$total_pop + data[[name]]
	}

	for (i in 1:length(field_names_list)) {
		output_data[[prop_names[i]]] <- data[[field_names_list[i]]]/output_data$total_pop
	}

	return(output_data)
}

compute_knn_from_distances <- function(distances, n=-1, idxs=NULL) {
	if (n <= 0) {
		n <- ncol(distances)
	}
	if (is.null(idxs)) {
		idxs <- 1:nrow(distances)
	}

	## parallel, mclapply:
	trajectories <- t(matrix(
		unlist(pbmclapply(idxs, function(i) {
			return(order(distances[i, ])[1:n])
		}, mc.cores=8)),
		nrow=n, ncol=length(idxs)
	))

	return(trajectories)
}

compute_trajectories <- function(data, trajectories, field_names_list) {
	output_data <- data.frame(mat=rep(0, nrow(trajectories)))

	traj_names <- paste("TRAJ_", field_names_list, sep="")
	for (traj_name in traj_names) {
		output_data[[traj_name]] <- matrix(0, nrow=nrow(trajectories), ncol=ncol(trajectories))
	}

	total <- 0
	for (name in field_names_list) {
		traj_name <- paste("TRAJ_", name, sep="")
		output_data[[traj_name]][] <- matrix(
			data[[name]][unlist(trajectories)],
			nrow(trajectories), ncol(trajectories))

		output_data[[traj_name]] <- rowCumSums(output_data[[traj_name]])

		total <- output_data[[traj_name]] + total
	}

	for (traj_name in traj_names) {
		output_data[[traj_name]] <- output_data[[traj_name]]/total
	}

	return(output_data)
}

kl_divergence <- function(p_list1, p_list2) {
	p_list1[p_list2 <= 0] <- 1
	p_list2[p_list2 <= 0] <- 1
	p_list1[p_list1 <= 0] <- 1
	p_list2[p_list1 <= 0] <- 1

	return(max(
		sum(p_list1*log2(p_list1/p_list2)),
		0))
}

compute_kl_divergence <- function(trajectories, field_names_list, total_proportion) {
	traj_dim <- dim(trajectories[[field_names_list[1]]])

	p_list <- matrix(0, nrow=length(field_names_list), ncol=traj_dim[1]*traj_dim[2])
	for (i in 1:length(field_names_list)) {
		p_list[i, ] <- unlist(trajectories[[field_names_list[i]]])
	}

	p_list          [total_proportion <= 0, ] <- 1
	total_proportion[total_proportion <= 0]   <- 1
	p_list          [p_list           <= 0]   <- 1e-20

	p_list <- p_list*log2(p_list/total_proportion)
	KL_div_trajectory <- matrix(colSums(p_list),
		nrow=traj_dim[1], ncol=traj_dim[2])

	return(KL_div_trajectory)
}

compute_focal_distance <- function(focal_lim, kl_traj, trajectories, x_accumulation_axis) {
	focal_distances <- matrix(0, nrow=nrow(trajectories), ncol=length(focal_lim))
	for (it in 1:nrow(trajectories)) {
		neighbor_idxs <- unlist(trajectories[it, ])
		accumulated <- accumulate(x_accumulation_axis[neighbor_idxs])
		
		idx <- 1
		for (i in length(focal_lim):1) {
			while (kl_traj[it, idx] > focal_lim[i] && idx < ncol(kl_traj)) {
				idx <- idx + 1
			}

			focal_distances[it, i] <- accumulated[idx]
		}
	}

	return(focal_distances)
}

dis_coef_norm_factor <- function(focal_lim, total_proportion, total_population) {
	## normalisation factor
	# housing status trajectory:  total_hlm_number social housing -> total_housing_number-total_hlm_number non-social housing
	# proportion trajectory:      100% for x in [0, total_hlm_number] then x/total_hlm_number for x in [total_hlm_number, total_housing_number]
	# kl_divergence trajectory:   kl_divergence(1, total_hlm_number/total_housing_number) for x in [0, total_hlm_number], then kl_divergence(...)
	# focal distance:             compute numerically
	# normalizing factor:         compute numerically
	## constants
	num_point   <- 10000 
	scale       <- total_population/num_point

	proportion   <- sort(total_proportion)
	populations  <- proportion*total_population
	midle_points <- cumsum(populations/scale)
	# upper limit:
	midle_points[length(midle_points)] <- num_point

	## trajectories:
	trajectory    <- matrix(0, nrow=length(total_proportion), ncol=num_point)
	for (j in 1:midle_points[1]) {
		trajectory[1, j] <- 1
	}
	for (i in 2:length(midle_points)) {
		for (j in midle_points[i-1]:midle_points[i]) {
			pop <- j*scale
			for (k in 1:i-1) {
				this_pop         <- populations[k]
				trajectory[k, j] <- this_pop/pop
			}
			this_pop         <- (j - midle_points[i-1])*scale
			trajectory[i, j] <- this_pop/pop
		}
	}

	## KL divergence:
	max_KL_div_trajectory <- unlist(lapply(1:num_point, function(j) {
		return(kl_divergence(as.numeric(trajectory[, j]), proportion))
	}))

	## focal distance -> dis coef:
	dis_coef <- 0
	# compute focal distances "as normal":
	idx    <- 1
	for (i in length(focal_lim):1) {
		while (max_KL_div_trajectory[idx] > focal_lim[i] && idx < length(max_KL_div_trajectory)) {
			idx <- idx + 1
		}

		# integrate the focal distances "on the fly" rather then storing it:
		i_df     <- max(i, 2)
		df       <- focal_lim[i_df] - focal_lim[i_df-1]
		dis_coef <- dis_coef + idx*scale*df
	}

	return(dis_coef)
}

compute_dist_coef <- function(focal_lim, focal_distances) {
	return(c(unlist(lapply(1:nrow(focal_distances), function(it) {
		dis_coef <- 0

		# compute focal distances "as normal":
		for (i in 1:length(focal_lim)) {
			i_df     <- max(i, 2)
			df       <- focal_lim[i_df] - focal_lim[i_df-1]
			dis_coef <- dis_coef + focal_distances[it, i]*df
		}

		return(dis_coef)
	}))))
}

compute_dis_coef_from_scratch <- function(map, focal_lim, field_names_list, x_accumulation_axis, kernel_size=10, threadPerCore=0.5) {
	traj_field_names_list <- paste("TRAJ_", field_names_list, sep="")

	total_vec <- compute_total_accumulated_pop(map, field_names_list)
	total_pop <- sum(total_vec)
	prop_vec  <- total_vec/total_pop

	dis_coef_norm_factor <- dis_coef_norm_factor(focal_lim, prop_vec, total_pop)

	map_size <- nrow(map)
	n_kernel <- ceiling(map_size/kernel_size)

	# function that computes the distortion coefficient without storing intermediary data:
	compute_dis_coef_onTheFly <- function(it, KL_div_traj, neighbor_idxs) {
		accumulated_x_axis <- accumulate(x_accumulation_axis[unlist(neighbor_idxs[it, ])])
		
		dis_coef <- 0

		# compute focal distances "as normal":
		idx    <- 1
		for (i in length(focal_lim):1) {
			while (KL_div_traj[it, idx] > focal_lim[i] && idx < ncol(KL_div_traj)) {
				idx <- idx + 1
			}

			# integrate the focal distances "on the fly" rather then storing it:
			i_df     <- max(i, 2)
			df       <- focal_lim[i_df] - focal_lim[i_df-1]
			dis_coef <- dis_coef + accumulated_x_axis[idx]*df
		}
		dis_coef <- dis_coef/dis_coef_norm_factor

		return(dis_coef)
	}

	num_cores <- max(1, round(detectCores()*threadPerCore))
	return(unlist(
		pbmclapply(1:n_kernel, function(i) {
			begin <-     round((i-1)*kernel_size) + 1
			end   <- min(round( i   *kernel_size), map_size)
			size  <- end - begin + 1

			# compute neighbors:
			nearest_neighbors_idx <- compute_knn_from_map(map, -1, begin:end)

			# compute trajectories:
			trajectories_data <- compute_trajectories(map, nearest_neighbors_idx, field_names_list)

			# compute KL-divergence:
			KL_div_trajectory <- compute_kl_divergence(trajectories_data, traj_field_names_list, prop_vec)
			rm(trajectories_data)

			# compute distortion coeffecicients:
			dis_coefs <- lapply(1:size, function(it) {
				return(compute_dis_coef_onTheFly(it, KL_div_trajectory, nearest_neighbors_idx))
			})

			return(dis_coefs)
		}, mc.cores=num_cores, ignore.interactive=TRUE)
	))
}