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
    data/data-features_eids.csv
    data/data-features_instances.csv


MI01B_Preprocessing_imagesIDs:
Inputs:
    data/FoldsAugmented/*
Outputs:
    data/MI01B_Preprocessing_imagesIDs/instances23_eids_{0..9}.csv

MI01C_Preprocessing_folds:
Inputs:
    Arguments: Age Abdomen
