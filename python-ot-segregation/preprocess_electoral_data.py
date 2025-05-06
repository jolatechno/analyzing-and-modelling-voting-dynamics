#!/usr/bin/env python3

import urllib.request
import os.path
import pandas as pd

election_id = "france_pres_tour1_2022"

election_data_file_names = {
	"france_pres_tour1_2022" : "data/resultats-par-niveau-burvot-t1-france-entiere.xlsx"
}
election_data_urls       = {
	"france_pres_tour1_2022" : "https://www.data.gouv.fr/fr/datasets/r/98eb9dab-f328-4dee-ac08-ac17211357a8"
}

output_file_names = {
	"france_pres_tour1_2022" : "data/france_pres_tour1_2022_preprocessed.csv"
}
bvote_position_output_file_name  = "data/table-adresses-preprocessed.csv"

read_bvote_position_elector      = False
bvote_position_elector_file_name = "data/table-adresses-reu.parquet"
bvote_position_elector_url       = "https://www.data.gouv.fr/fr/datasets/r/8b5c75df-24ea-43ae-9f4c-6f5c633e942b"

base_field_election_database          = [
	"id_brut_bv_reu", "code_commune", "Libellé de la commune", "Libellé de la circonscription",
	"Inscrits", "Abstentions", "Votants", "Blancs"
]
field_per_candidate_election_database = ["Voix", "% Voix/Ins", "% Voix/Exp"]
field_bvote_position_database         = ["longitude", "latitude"]

""" ##################################
######################################
read field from the election data file
######################################
################################## """

if not os.path.isfile(election_data_file_names[election_id]):
	print(f"\"{ election_data_file_names[election_id] }\" not found, downloading from { election_data_urls[election_id] }")
	urllib.request.urlretrieve(election_data_urls[election_id], election_data_file_names[election_id])

print(f"Reading data from \"{ election_data_file_names[election_id] }\"")
election_database = pd.read_excel(election_data_file_names[election_id])

""" ##############
process the fields
############## """

all_fieldnames           = list(election_database.columns)
fieldnames_per_candidate = ["N°Panneau", "Sexe", "Nom", "Prénom", "Voix", "% Voix/Ins", "% Voix/Exp"]
fieldnames_idx           = {fieldnames_per_candidate[i] : i for i in range(len(fieldnames_per_candidate)) }
candidate_offset         = all_fieldnames.index(fieldnames_per_candidate[0])
candidate_length         = len(fieldnames_per_candidate)

candidate_begin_idx = list(range(candidate_offset, len(all_fieldnames), candidate_length))
candidate_list      = [election_database[all_fieldnames[i + fieldnames_idx["Nom"]]][0] for i in candidate_begin_idx]

numerical_mask    = election_database["Code du b.vote"].str.isnumeric()
election_database = election_database[numerical_mask]

# Modifs pour les trois villes à arrondissements :
## Paris :
paris_mask                                              = election_database["Libellé de la commune"] == "Paris"
arrondissements_paris                                   = election_database[paris_mask]["Code du b.vote"].str[0:2]
election_database.loc[paris_mask, "Code de la commune"] = ("1" + arrondissements_paris).astype(int)
election_database.loc[paris_mask, "Code du b.vote"    ] = election_database[paris_mask]["Code du b.vote"].str[2: ]
## Marseille :
marseille_mask                                              = election_database["Libellé de la commune"] == "Marseille"
arrondissements_marseille                                   = election_database[marseille_mask]["Code du b.vote"].str[0:2].str.lstrip('0')
election_database.loc[marseille_mask, "Code de la commune"] = ("2" + arrondissements_marseille.str.zfill(2)).astype(int)
election_database.loc[marseille_mask, "Code du b.vote"    ] = arrondissements_marseille + election_database[marseille_mask]["Code du b.vote"].str[2: ]
## Lyon :
lyon_mask                                              = election_database["Libellé de la commune"] == "Lyon"
arrondissements_lyon                                   = election_database[lyon_mask]["Code du b.vote"].str[1]
election_database.loc[lyon_mask, "Code de la commune"] = ("38" + arrondissements_lyon).astype(int)
election_database.loc[lyon_mask, "Code du b.vote"    ] = arrondissements_lyon + election_database[lyon_mask]["Code du b.vote"].str[2: ]


election_database["code_commune"  ] = election_database["Code du département"] +       election_database["Code de la commune"].astype(str).str.zfill(3)
election_database["id_brut_bv_reu"] = election_database["code_commune"       ] + "_" + election_database["Code du b.vote"    ].str.lstrip('0')

field_election_database = {}
for field in base_field_election_database:
	field_election_database[field] = field
for i,candidate in enumerate(candidate_list):
	for field in field_per_candidate_election_database:
		base_field  = all_fieldnames[candidate_offset + candidate_length*i + fieldnames_idx[field]]
		final_field = candidate + " " + field
		field_election_database[base_field] = final_field

election_database.drop(
	columns=list(set(election_database.columns) - set(field_election_database.keys())),
	inplace=True)
election_database.rename(columns=field_election_database, inplace=True)
election_database.set_index("id_brut_bv_reu")

""" ##########################################
##############################################
read field from the elector position data file
##############################################
########################################## """

if not os.path.isfile(bvote_position_elector_file_name):
	print(f"\"{ bvote_position_elector_file_name }\" not found, downloading from { bvote_position_elector_url }")
	urllib.request.urlretrieve(bvote_position_elector_url, bvote_position_elector_file_name)

print(f"Reading data from \"{ bvote_position_elector_file_name }\"")
bvote_position_database = pd.read_parquet(bvote_position_elector_file_name, engine='pyarrow')

""" ##############
process the fields
############## """

bvote_position_database = bvote_position_database.dropna(subset=["longitude", "latitude"])
bvote_position_database.drop(
	columns=list(set(bvote_position_database.columns) - set(field_bvote_position_database + ["id_brut_bv_reu"])),
	inplace=True)

bvote_grouped_database  = bvote_position_database.groupby("id_brut_bv_reu", as_index=False)
bvote_averaged_database = bvote_grouped_database.mean()

""" ############################################
################################################
merge both database into one steamlined database
################################################
############################################ """

final_database = election_database.merge(bvote_averaged_database, how="left", on="id_brut_bv_reu")

""" #########
write to file
######### """

print(f"Write to \"{ output_file_names[election_id] }\"")
final_database.to_csv(output_file_names[election_id], index=False)

print(f"Write to \"{ bvote_position_output_file_name }\"")
bvote_position_database.to_csv(bvote_position_output_file_name, index=False)