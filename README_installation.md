pip install --upgrade pip
pip install -r requirements.txt

Files to have : 

MI01A_Preprocessing_main:
Inputs:
    data/fake_short_ukb41230.csv
    data/fake_all_eids.csv
    data/missing_samples.csv
    data/fake_PA_visit_data.csv
Outputs:
    data/MI01A_Preprocessing_main/data-features_eids.csv
    data/MI01A_Preprocessing_main/data-features_instances.csv


MI01B_Preprocessing_imagesIDs:
Inputs:
    data/FoldsAugmented/*
Outputs:
    data/MI01B_Preprocessing_imagesIDs/instances23_eids_{0..9}.csv

MI01C_Preprocessing_folds:
Inputs:
    data/Abdomen/*
    Arguments: Age Abdomen
Outputs:
    data/MI01C_Preprocessing_folds/data-features_*


# TO SKIP BECAUSE A GPU IS NEEDED TO TRAIN THE ALGORITHMS
MI02_Training_parallel:
Inputs:
    Arguments: Age Abdomen Pancreas Contrast InceptionResNetV2 1 1024 Adam 0.0001 0.1 0.5 1.0 9
               The last argument is the outer_fold.
Outputs:
    data/MI02_Training/model-weights_Age_Abdomen_*
    Weights of the neural network from the training. <- They are already given with the suffix *trained_* for all the outer folds of Pancreas Contrast so that you don't need to train the algorithm by your self


# YOU CAN TRY BY YOUR SELF, AS IT MIGHT TAKE SOME TIME, WE PROVIDE THE OUTPUTS OF PANCREAS CONTRAST
MI03A_Predictions_generate_parallel
Inputs:
    Arguments: Age Abdomen Pancreas Contrast InceptionResNetV2 1 1024 Adam 0.0001 0.1 0.5 1.0 9
               The last argument is the outer_fold.
    data/MI02_Training/trained_model-weights_Age_Abdomen_*
Outputs:
    data/MI03A_Predictions_generate/Predictions_instances_Age_Abdomen_*
    Prediction of the specified outer_fold. <- They are already given with the suffix *short_*


MI03B_Predictions_concatenate
Inputs:
    Arguments: Age Abdomen Pancreas Contrast InceptionResNetV2 1 1024 Adam 0.0001 0.1 0.5 1.0
    data/MI03A_Predictions_generate/Predictions_instances_Age_Abdomen_*
Outputs:
    data/MI03B_Predictions_concatenate/Predictions_instances_Age_Abdomen_*


MI03C_Predictions_merge
Inputs:
    Arguments: Age val
               The last argument has to be changed to *test* too.
    data/MI03B_Predictions_concatenate/Predictions_instances_Age_Abdomen_*
Outputs:
    data/MI03C_Predictions_merge/PREDICTIONS_withoutEnsembles_instances_Age_*


MI03D_Predictions_eids
Inputs:
    Arguments: Age val
               The last argument has to be changed to *test* too.
    data/MI03C_Predictions_merge/PREDICTIONS_withoutEnsembles_instances_Age_*
Outputs:
    data/MI03D_Predictions_eids/Predictions_eids_concatenate/Predictions_instances_Age_Abdomen_*
    data/MI03D_Predictions_eids/PREDICTIONS_withoutEnsembles_eids_Age_*


MI04A_Performances_generate
Inputs:
    Arguments: Age Abdomen Pancreas Contrast InceptionResNetV2 1 1024 Adam 0.0001 0.1 0.5 1.0 val instances
               The second last argument has to be changed to *test* too.
    data/MI03B_Predictions_concatenate/Predictions_instances_Age_Abdomen_*
Outputs:
    data/MI04A_Performances_generate/Predictions_instances_Age_Abdomen_*


MI04B_Performances_merge
Inputs:
    Arguments: Age val instances False
               The second argument has to be changed to *test* too.
    data/MI04A_Performances_generate/Predictions_instances_Age_Abdomen_*
Outputs:
    data/MI04B_Performances_merge/PERFORMANCES_withoutEnsembles_*_instances_Age_*


MI04C_Performances_tuning
Inputs:
    Arguments: Age instances
    data/MI04B_Performances_merge/PERFORMANCES_withoutEnsembles_ranked_instances_Age_*
    data/MI03C_Predictions_merge/PREDICTIONS_withoutEnsembles_instances_Age_*
Outputs:
    data/MI04C_Performances_tuning/PERFORMANCES_tuned_*


MI05A_Ensembles_predictions
Inputs:
    Arguments: Age instances
    data/MI04C_Performances_tuning/PERFORMANCES_tuned_*
Outputs:
    data/MI05A_Ensembles_predictions/Predictions_instances_Age_*
    data/MI05A_Ensembles_predictions/PREDICTIONS_withEnsembles_instances_Age_*


MI05B_Performances_generate
Inputs:
    Arguments: Age "\*" "\*" "\*" "\*" "\*" "\*" "\*" "\*" "\*" "\*" "\*" val instances
               The second argument has to be changed to *"\*instances23"* too.
               The second last argument has to be changed to *test* too.
    data/MI05A_Ensembles_predictions/Predictions_instances_Age_*
Outputs:
    data/MI05B_Performances_generate/Predictions_instances_Age_*


MI05C_Performances_merge
Inputs:
    Arguments: Age val instances True
               The second argument has to be changed to *test* too.
    data/MI04A_Performances_generate/Predictions_instances_Age_Abdomen_*
    data/MI05B_Performances_generate/Predictions_instances_Age_*
Outputs:
    data/MI05C_Performances_merge/PERFORMANCES_withoutEnsembles_*_instances_Age_*


