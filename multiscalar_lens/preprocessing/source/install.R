install.packages("remotes")
install.packages("BiocManager")

library(remotes)

install.packages("readr")
install.packages("readxl")
install.packages("stringr")
install.packages("rjson")
BiocManager::install("rhdf5")

install_github("r-spatial/sf")
install.packages("osrm")
install.packages("dplyr")
install.packages("weights")
BiocManager::install("nngeo")



## legacy:
#install.packages("foreach")
#install.packages("parallel")
#install.packages("pbmcapply")
#install.packages("tmaptools")
#install.packages("pbapply")