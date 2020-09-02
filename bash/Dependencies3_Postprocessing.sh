IDs_MI03C=$(./MI03C_Predictions_merge_parallel.sh)
IDs_MI03D=$(./MI03D05B_Predictions_eids_parallel.sh False $IDs_MI03C)
IDs_MI04A=$(./MI04A_Performances_generate_parallel.sh $IDs_MI03D)
IDs_MI04B=$(./MI04B05D_Performances_merge_parallel.sh False $IDs_MI04A)
IDs_MI04C=$(./MI04C_Performances_tuning_parallel.sh $IDs_MI04C)
IDs_MI05A=$(./MI05A_Ensembles_predictions_generate_and_merge_parallel.sh $IDs_MI04C)
IDs_MI05B=$(./MI03D05B_Predictions_eids_parallel.sh True $IDs_MI05A )
IDs_MI05C=$(./MI05C_Ensembles_performances_generate_parallel.sh $IDs_MI05B)
IDs_MI05D=$(./MI04B05D_Performances_merge_parallel.sh True $IDs_MI05C)
IDs_MI06A=$(./MI06A_Residuals_generate_parallel.sh $IDs_MI05D)
IDs_MI06B=$(./MI06B_Residuals_correlations_parallel.sh $IDs_MI06A) 
IDs_MI07A=$(./MI07A_Select_best_parallel.sh $IDs_MI06B)
IDs_MI07B=$(./MI07B_Select_correlationsNAs_parallel.sh $IDs_MI07A)
IDs_MI08=$(./MI08_Attentionmaps_parallel.sh $IDs_MI07A)
IDs_MI09A=$(./MI09A_GWAS_preprocessing_parallel.sh $IDs_MI07A)
IDs_MI09B=$(./MI09B_GWAS_remove_parallel.sh $IDs_MI09A)
IDs_MI09C=$(./MI09C_GWAS_bolt_parallel.sh $IDs_MI09B)
IDs_MI09D=$(./MI09D_GWAS_correlations_parallel.sh $IDs_MI09C)
IDs_MI09E=$(./MI09E_GWAS_postprocessing_parallel.sh $IDs_MI09D)

