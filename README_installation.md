pip install --upgrade pip
pip install -r requirements.txt

Files to have : 

MI01A_Preprocessing_main:
Inputs:
    short_ukb41230.csv
    All_eids.csv
    missing_samples.csv
    PA_visit_data.csv
Outputs:
    data-features_eids.csv
    data-features_instances.csv


MI01B_Preprocessing_imagesIDs:
Inputs:
    FoldsAugmented/*
Outputs:
    instances23_eids_{0..9}.csv

MI01C_Preprocessing_folds:
Inputs:
    Arguments: Age Abdomen
