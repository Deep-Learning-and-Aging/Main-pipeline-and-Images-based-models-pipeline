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


MI03C_Predictions_merge_parallel
Inputs:
    Arguments: Age val
               The last argument has to be changed to *val* and *test* too.
    data/MI03B_Predictions_concatenate/Predictions_instances_Age_Abdomen_*
Outputs:
    data/MI03C_Predictions_merge/Predictions_instances_Age_Abdomen_*
