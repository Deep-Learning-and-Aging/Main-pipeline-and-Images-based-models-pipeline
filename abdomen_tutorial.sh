#!/bin/bash

python scripts/MI01A_Preprocessing_main.py

python scripts/MI01B_Preprocessing_imagesIDs.py

python scripts/MI01C_Preprocessing_folds.py Age Abdomen

# python scripts/MI03A_Predictions_generate.py Age Abdomen Pancreas Contrast InceptionResNetV2 1 1024 Adam 0.0001 0.1 0.5 1.0 0

python scripts/MI03B_Predictions_concatenate.py Age Abdomen Pancreas Contrast InceptionResNetV2 1 1024 Adam 0.0001 0.1 0.5 1.0

python scripts/MI03C_Predictions_merge.py Age val
python scripts/MI03C_Predictions_merge.py Age test

python scripts/MI03D_Predictions_eids.py Age val
python scripts/MI03D_Predictions_eids.py Age test

python scripts/MI04A05B_Performances_generate.py Age Abdomen Pancreas Contrast InceptionResNetV2 1 1024 Adam 0.0001 0.1 0.5 1.0 val instances
python scripts/MI04A05B_Performances_generate.py Age Abdomen Pancreas Contrast InceptionResNetV2 1 1024 Adam 0.0001 0.1 0.5 1.0 test instances

python scripts/MI04B05C_Performances_merge.py Age val instances False
python scripts/MI04B05C_Performances_merge.py Age test instances False

python scripts/MI04C_Performances_tuning.py Age instances

python scripts/MI05A_Ensembles_predictions.py Age instances

python scripts/MI04A05B_Performances_generate.py Age "*" "*" "*" "*" "*" "*" "*" "*" "*" "*" "*" val instances
python scripts/MI04A05B_Performances_generate.py Age "*instances23" "*" "*" "*" "*" "*" "*" "*" "*" "*" "*" val instances
python scripts/MI04A05B_Performances_generate.py Age "*" "*" "*" "*" "*" "*" "*" "*" "*" "*" "*" test instances
python scripts/MI04A05B_Performances_generate.py Age "*instances23" "*" "*" "*" "*" "*" "*" "*" "*" "*" "*" test instances

python scripts/MI04B05C_Performances_merge.py Age val instances True
python scripts/MI04B05C_Performances_merge.py Age test instances True

python scripts/MI06A_Residuals_generate.py Age test instances

python scripts/MI06B_Residuals_correlations.py Age test instances

python scripts/MI07A_Select_best.py Age instances
