# Multidimensionality of Aging - a tutorial on the Abdomen dimension
The project consist on predicting age using different types of medical images from UK Biobank, and deep learning.
This pipeline generates the images-based predictors, then merges them and ensemble them with the predictors build on images. Finally, it performs postprocessing steps.


**In order not to disclose any personal information from the participants of UK Biobank, all the data accessible here is fake. You should not expect to get any interesting results from the outputs.**

## Project architecture
Throught this tutorial you will use two directories: 

- scripts: where the python scripts can be found. This is the core of the pipeline.

- data: where the results used and generated by the pipeline are saved.

An exact list of all the important files to have at the begining of the tutorial can be seen [here](##Structure-at-the-begin-of-the-tutorial). At the end, you should have a structure that looks like [that](##Structure-of-the-data-folder-at-the-end-of-the-tutorial).


## Working environment
The project has been developped with Python 3.6.14, using a Debian Buster distribution. Please choose from the three different ways to setup your environment:
- using the docker image ?? [TO DO] | __tested__.
- using VS Code with Remote-Containers. The files to setup the container are stored [here](./.devcontainer) | __tested__.
- downloading python 3.6.14 from the website with this [link](https://www.python.org/downloads/release/python-3614/) | __not tested__.

Only a CPU is required for this tutorial with 2 Gb of RAM. All the results needing heavy and long lasting computations are given. To host the project on your local computer, 7 Gb of memory is needed: 2.5 Gb for the dependencies and 3 Gb for the data.


## Install packages
If you use the docker image created for this tutorial, you don't need to do anything.

Otherwise, please type those four lines of code in your terminal being at the top directory: (Expected time 6 min)
```bash
python3 -m venv env_tutorial
source env_tutorial/bin/activate
pip install --upgrade "pip==21.1.3"
pip install -r requirements.txt
```

##  Step by step
If you want to execute the pipeline in a single line of code, you can simply source this [bash](./abdomen_tutorial.sh) script in a terminal where a local environment is activated (Expected time 12 min):
```bash
source ./abdomen_tutorial.sh
```

<br/>
<br/>

- Each one of the following steps is presented the same way.

Command line:
> The command line to execute to proceed the step.\
> Sometimes, you will need to execute the step several times with different arguments.\
> When in doubt, you can refer to the [bash](./abdomen_tutorial.sh) script that gathers all the command lines to execute.

Inputs:
> List of the inputs to have to make the step working. Normally, they should come along when you clone the repository.

Outputs:
> List of the outputs that you should get from the command line.

<br/>
<br/>

- MI01A_Preprocessing_main: Preprocesses the main dataframe (Expected time 1 min 30 sec).

Command line:
> python scripts/MI01A_Preprocessing_main.py

Inputs:
> data/fake_short_ukb41230.csv\
> data/fake_all_eids.csv\
> data/missing_samples.csv\
> data/fake_PA_visit_data.csv

Outputs:
> data/MI01A_Preprocessing_main/data-features_eids.csv\
> data/MI01A_Preprocessing_main/data-features_instances.csv

<br/>
<br/>

- MI01B_Preprocessing_imagesIDs: Splits the different images datasets into folds for the future cross validation (Expected time 1 min).

Command line:
> python scripts/MI01B_Preprocessing_imagesIDs.py

Inputs:
> data/FoldsAugmented/*
    
Outputs:
> data/MI01B_Preprocessing_imagesIDs/instances23_eids_{0..9}.csv

<br/>
<br/>

- MI01C_Preprocessing_folds: Splits the data into training, validation and testing sets for all CV folds (Expected time 25 sec).

Command line:
> python scripts/MI01C_Preprocessing_folds.py Age Abdomen

Inputs:
> data/Abdomen/*

Outputs:
> data/MI01C_Preprocessing_folds/data-features_*

<br/>
<br/>

#### TO SKIP BECAUSE A GPU IS NEEDED TO TRAIN THE ALGORITHMS
MI02_Training: This step should be repeated until all models have converged (Expected time depends on your computer).

Command line:
> python scripts/MI02_Training.py Age Abdomen Pancreas Contrast InceptionResNetV2 1 1024 Adam 0.0001 0.1 0.5 1.0 9\
> The last argument is the outer_fold.

Inputs:
> data/Abdomen/*
> data/MI01C_Preprocessing_folds/data-features_*

Outputs:
> data/MI02_Training/model-weights_Age_Abdomen_*\
> Weights of the neural network from the training. <- They are already given with the suffix *trained_* for all the outer folds of Pancreas Contrast so that you don't need to train the algorithm by your self

<br/>
<br/>

#### YOU CAN TRY BY YOUR SELF, AS IT MIGHT TAKE SOME TIME, WE PROVIDE THE OUTPUTS OF PANCREAS CONTRAST
- MI03A_Predictions_generate: Generates the predictions from all models (Expected time 6 min 30 sec for each outer fold).

Command line:
> python scripts/MI03A_Predictions_generate.py Age Abdomen Pancreas Contrast InceptionResNetV2 1 1024 Adam 0.0001 0.1 0.5 1.0 9\
> The last argument is the outer_fold.

Inputs:
> data/Abdomen/*
> data/MI02_Training/trained_model-weights_Age_Abdomen_*

Outputs:
> data/MI03A_Predictions_generate/Predictions_instances_Age_Abdomen_* <- They are already given with the suffix *short_*

<br/>
<br/>

- MI03B_Predictions_concatenate: Concatenates the predictions from the different cross-validation folds (Expected time 10 sec).

Command line:
> python scripts/MI03B_Predictions_concatenate.py Age Abdomen Pancreas Contrast InceptionResNetV2 1 1024 Adam 0.0001 0.1 0.5 1.0

Inputs:
> data/MI03A_Predictions_generate/short_Predictions_instances_Age_Abdomen_*

Outputs:
> data/MI03B_Predictions_concatenate/Predictions_instances_Age_Abdomen_*

<br/>
<br/>

- MI03C_Predictions_merge: Merges the predictions from all models into a unified dataframe (Expected time 5 sec for each run).

Command line:
> python scripts/MI03C_Predictions_merge.py Age val\
> The last argument has to be changed to *test* too.

Inputs:
> data/MI03B_Predictions_concatenate/Predictions_instances_Age_Abdomen_*

Outputs:
> data/MI03C_Predictions_merge/PREDICTIONS_withoutEnsembles_instances_Age_*

<br/>
<br/>

- MI03D_Predictions_eids: Computes the average age prediction across samples from different instances for every participant (Expected time 5 sec for each run).

Command line:
> python scripts/MI03D_Predictions_eids.py Age val\
> The last argument has to be changed to *test* too.

Inputs:
> data/MI03C_Predictions_merge/PREDICTIONS_withoutEnsembles_instances_Age_*

Outputs:
> data/MI03D_Predictions_eids/Predictions_eids_concatenate/Predictions_instances_Age_Abdomen_*
> data/MI03D_Predictions_eids/PREDICTIONS_withoutEnsembles_eids_Age_*

<br/>
<br/>

- MI04A_Performances_generate: Computes the performances for each model (Expected time 1 min 30 sec for each run).

Command line:
> python scripts/MI04A05B_Performances_generate.py Age Abdomen Pancreas Contrast InceptionResNetV2 1 1024 Adam 0.0001 0.1 0.5 1.0 val instances\
> The second last argument has to be changed to *test* too.

Inputs:
> data/MI03B_Predictions_concatenate/Predictions_instances_Age_Abdomen_*

Outputs:
> data/MI04A_Performances_generate/Predictions_instances_Age_Abdomen_*

<br/>
<br/>

- MI04B_Performances_merge: False Merges the performances of the different models into a unified dataframe (Expected time 5 sec for each run).

Command line:
> python scripts/MI04B05C_Performances_merge.py Age val instances False\
> The second argument has to be changed to *test* too.

Inputs:
> data/MI04A_Performances_generate/Predictions_instances_Age_Abdomen_*

Outputs:
> data/MI04B_Performances_merge/PERFORMANCES_withoutEnsembles_*_instances_Age_*

<br/>
<br/>

- MI04C_Performances_tuning: For each model, selects the best hyperparameter combination (Expected time 10 sec).

Command line:
> python scripts/MI04C_Performances_tuning.py Age instances

Inputs:
> data/MI04B_Performances_merge/PERFORMANCES_withoutEnsembles_ranked_instances_Age_*
> data/MI03C_Predictions_merge/PREDICTIONS_withoutEnsembles_instances_Age_*

Outputs:
> data/MI04C_Performances_tuning/PERFORMANCES_tuned_*

<br/>
<br/>

- MI05A_Ensembles_predictions: Hierarchically builds ensemble models (Expected time 5 sec).

Command line:
> python scripts/MI05A_Ensembles_predictions.py Age instances

Inputs:
> data/MI04C_Performances_tuning/PERFORMANCES_tuned_*

Outputs:
> data/MI05A_Ensembles_predictions/Predictions_instances_Age_*
> data/MI05A_Ensembles_predictions/PREDICTIONS_withEnsembles_instances_Age_*

<br/>
<br/>

- MI05B_Performances_generate: Computes the performances for the ensemble models (Expected time 1 min 15 sec for each run).

Command line:
> python scripts/MI04A05B_Performances_generate.py Age "\*" "\*" "\*" "\*" "\*" "\*" "\*" "\*" "\*" "\*" "\*" val instances\
> The second argument has to be changed to *"\*instances23"* too.\
> The second last argument has to be changed to *test* too.\

Inputs:
> data/MI05A_Ensembles_predictions/Predictions_instances_Age_*

Outputs:
> data/MI05B_Performances_generate/Predictions_instances_Age_*

<br/>
<br/>

- MI05C_Performances_merge: Adds the ensemble models to the unified dataframe (Expected time 10 sec for each run).

Command line:
> python scripts/MI04B05C_Performances_merge.py Age val instances True\
> The second argument has to be changed to *test* too.

Inputs:
> data/MI04A_Performances_generate/Predictions_instances_Age_Abdomen_* \
> data/MI05B_Performances_generate/Predictions_instances_Age_*

Outputs:
> data/MI05C_Performances_merge/PERFORMANCES_withoutEnsembles_*_instances_Age_*

<br/>
<br/>

- MI06A_Residuals_generate: Computes accelerated aging phenotypes (Residuals, corrected for residuals bias with respect to age) (Expected time 5 sec).

Command line:
> python scripts/MI06A_Residuals_generate.py Age test instances

Inputs:
> data/MI05A_Ensembles_predictions/PREDICTIONS_withEnsembles_instances_Age_*

Outputs:
> data/MI06A_Residuals_generate/RESIDUALS_instances_Age_test.csv

<br/>
<br/>

- MI06B_Residuals_correlations: Computes the phenotypic correlation between aging dimensions (Expected time 5 sec).

Command line:
> python scripts/MI06B_Residuals_correlations.py Age test instances

Inputs:
> data/MI06A_Residuals_generate/RESIDUALS_instances_Age_test.csv

Outputs:
> data/MI06B_Residuals_correlations/RESIDUALS_instances_Age_test.csv

<br/>
<br/>

- MI07A_Select_best: For each aging main dimension and selected subdimensions, select the best performing model (Expected time 5 sec).

Command line:
> python scripts/MI07A_Select_best.py Age instances

Inputs:
> data/MI05A_Ensembles_predictions/PREDICTIONS_withEnsembles_* \
> data/MI06A_Residuals_generate/RESIDUALS_* \
> data/MI05C_Performances_merge/PERFORMANCES_withEnsembles_ranked_* \
> data/MI06B_Residuals_correlations/ResidualsCorrelations_str_* \
> data/MI06B_Residuals_correlations/ResidualsCorrelations_samplesizes_*

Outputs:
> data/MI07A_Select_best/PERFORMANCES_bestmodels_* \
> data/MI07A_Select_best/PREDICTIONS_bestmodels_instances_Age_test.csv\
> data/MI07A_Select_best/RESIDUALS_bestmodels_instances_Age_test.csv\
> data/MI07A_Select_best/ResidualsCorrelations_bestmodels_*

<br/>
<br/>

## Structure at the begin of the tutorial
```
 📦Age_Sex_and_Medical_Images
 ┣ 📂.devcontainer
 ┃ ┣ 📜Dockerfile
 ┃ ┗ 📜devcontainer.json
 ┣ 📂data
 ┃ ┣ 📂Abdomen
 ┃ ┃ ┣ 📂Liver
 ┃ ┃ ┃ ┣ 📂Contrast
 ┃ ┃ ┃ ┃ ┣ 📜1006879_2.jpg
 ┃ ┃ ┃ ┃ ┣ 📜1008016_2.jpg
 ┃ ┃ ┃ ┃ ┗ 📜*.jpg
 ┃ ┃ ┃ ┗ 📂Raw
 ┃ ┃ ┃   ┣ 📜1006879_2.jpg
 ┃ ┃ ┃   ┣ 📜1008016_2.jpg
 ┃ ┃ ┃   ┗ 📜*.jpg
 ┃ ┃ ┗ 📂Pancreas
 ┃ ┃   ┣ 📂Contrast
 ┃ ┃   ┃ ┣ 📜1013920_2.jpg
 ┃ ┃   ┃ ┣ 📜1023499_2.jpg
 ┃ ┃   ┃ ┣ 📜*.jpg
 ┃ ┃   ┗ 📂Raw
 ┃ ┃     ┣ 📜1013920_2.jpg
 ┃ ┃     ┣ 📜1023499_2.jpg
 ┃ ┃     ┗ 📜*.jpg
 ┃ ┣ 📂FoldsAugmented
 ┃ ┃ ┣ 📜fake_data-features_Heart_20208_Augmented_Age_test_{0... 9}.csv
 ┃ ┃ ┣ 📜fake_data-features_Heart_20208_Augmented_Age_train_{0... 9}.csv
 ┃ ┃ ┗ 📜fake_data-features_Heart_20208_Augmented_Age_val_{0... 9}.csv
 ┃ ┣ 📂MI01A_Preprocessing_main
 ┃ ┣ 📂MI01B_Preprocessing_imagesIDs
 ┃ ┣ 📂MI01C_Preprocessing_folds
 ┃ ┣ 📂MI02_Training
 ┃ ┃ ┣ 📜trained_model-weights_Age_Abdomen_Pancreas_Contrast_InceptionResNetV2_1_1024_Adam_0.0001_0.1_0.5_1.0_{0... 9}.h5
 ┃ ┣ 📂MI03A_Predictions_generate
 ┃ ┃ ┣ 📜short_Predictions_instances_Age_Abdomen_Pancreas_Contrast_InceptionResNetV2_1_1024_Adam_0.0001_0.1_0.5_1.0_test_{0... 9}.csv
 ┃ ┃ ┣ 📜short_Predictions_instances_Age_Abdomen_Pancreas_Contrast_InceptionResNetV2_1_1024_Adam_0.0001_0.1_0.5_1.0_train_{0... 9}.csv
 ┃ ┃ ┗ 📜short_Predictions_instances_Age_Abdomen_Pancreas_Contrast_InceptionResNetV2_1_1024_Adam_0.0001_0.1_0.5_1.0_val_{0... 9}.csv
 ┃ ┣ 📂MI03B_Predictions_concatenate
 ┃ ┣ 📂MI03C_Predictions_merge
 ┃ ┣ 📂MI03D_Predictions_eids
 ┃ ┃ ┗ 📂Predictions_eids_concatenate
 ┃ ┣ 📂MI04A_Performances_generate
 ┃ ┣ 📂MI04B_Performances_merge
 ┃ ┣ 📂MI04C_Performances_tuning
 ┃ ┣ 📂MI05A_Ensembles_predictions
 ┃ ┣ 📂MI05B_Performances_generate
 ┃ ┣ 📂MI05C_Performances_merge
 ┃ ┣ 📂MI06A_Residuals_generate
 ┃ ┣ 📂MI06B_Residuals_correlations
 ┃ ┣ 📂MI07A_Select_best
 ┃ ┣ 📜fake_PA_visit_date.csv
 ┃ ┣ 📜fake_all_eids.csv
 ┃ ┣ 📜fake_short_ukb41230.csv
 ┃ ┗ 📜missing_samples.csv
 ┣ 📂scripts
 ┃ ┣ 📜MI01A_Preprocessing_main.py
 ┃ ┣ 📜MI01B_Preprocessing_imagesIDs.py
 ┃ ┣ 📜MI01C_Preprocessing_folds.py
 ┃ ┣ 📜MI02_Training.py
 ┃ ┣ 📜MI03A_Predictions_generate.py
 ┃ ┣ 📜MI03B_Predictions_concatenate.py
 ┃ ┣ 📜MI03C_Predictions_merge.py
 ┃ ┣ 📜MI03D_Predictions_eids.py
 ┃ ┣ 📜MI04A05B_Performances_generate.py
 ┃ ┣ 📜MI04B05C_Performances_merge.py
 ┃ ┣ 📜MI04C_Performances_tuning.py
 ┃ ┣ 📜MI05A_Ensembles_predictions.py
 ┃ ┣ 📜MI06A_Residuals_generate.py
 ┃ ┣ 📜MI06B_Residuals_correlations.py
 ┃ ┣ 📜MI07A_Select_best.py
 ┃ ┗ 📜MI_Classes.py
 ┣ 📜.gitignore
 ┣ 📜README.md
 ┗ 📜requirements.txt
```

<br/>
<br/>

## Structure of the data folder at the end of the tutorial
```
📂data
┣ 📂Abdomen
┃ ┣ 📂Liver
┃ ┃ ┣ 📂Contrast
┃ ┃ ┃ ┣ 📜1006879_2.jpg
┃ ┃ ┃ ┣ 📜1008016_2.jpg
┃ ┃ ┃ ┗ 📜*.jpg
┃ ┃ ┗ 📂Raw
┃ ┃   ┣ 📜1006879_2.jpg
┃ ┃   ┣ 📜1008016_2.jpg
┃ ┃   ┗ 📜*.jpg
┃ ┗ 📂Pancreas
┃   ┣ 📂Contrast
┃   ┃ ┣ 📜1013920_2.jpg
┃   ┃ ┣ 📜1023499_2.jpg
┃   ┃ ┣ 📜*.jpg
┃   ┗ 📂Raw
┃     ┣ 📜1013920_2.jpg
┃     ┣ 📜1023499_2.jpg
┃     ┗ 📜*.jpg
┣ 📂FoldsAugmented
┃ ┣ 📜fake_data-features_Heart_20208_Augmented_Age_test_{0... 9}.csv
┃ ┣ 📜fake_data-features_Heart_20208_Augmented_Age_train_{0... 9}.csv
┃ ┗ 📜fake_data-features_Heart_20208_Augmented_Age_val_{0... 9}.csv
┣ 📂MI01A_Preprocessing_main
┃ ┣ 📜data-features_eids.csv
┃ ┗ 📜data-features_instances.csv
┣ 📂MI01B_Preprocessing_imagesIDs
┃ ┗ 📜instances23_eids_{0... 9}.csv
┣ 📂MI01C_Preprocessing_folds
┃ ┣ 📜data-features_Abdomen_Liver_Contrast_Age_test_{0... 9}.csv
┃ ┣ 📜data-features_Abdomen_Liver_Contrast_Age_train_{0... 9}.csv
┃ ┣ 📜data-features_Abdomen_Liver_Contrast_Age_val_{0... 9}.csv
┃ ┣ 📜data-features_Abdomen_Liver_Raw_Age_test_{0... 9}.csv
┃ ┣ 📜data-features_Abdomen_Liver_Raw_Age_train_{0... 9}.csv
┃ ┣ 📜data-features_Abdomen_Liver_Raw_Age_val_{0... 9}.csv
┃ ┣ 📜data-features_Abdomen_Pancreas_Contrast_Age_test_{0... 9}.csv
┃ ┣ 📜data-features_Abdomen_Pancreas_Contrast_Age_train_{0... 9}.csv
┃ ┣ 📜data-features_Abdomen_Pancreas_Contrast_Age_val_{0... 9}.csv
┃ ┣ 📜data-features_Abdomen_Pancreas_Raw_Age_test_{0... 9}.csv
┃ ┣ 📜data-features_Abdomen_Pancreas_Raw_Age_train_{0... 9}.csv
┃ ┗ 📜data-features_Abdomen_Pancreas_Raw_Age_val_{0... 9}.csv
┣ 📂MI02_Training
┃ ┣ 📜trained_model-weights_Age_Abdomen_Pancreas_Contrast_InceptionResNetV2_1_1024_Adam_0.0001_0.1_0.5_1.0_{0... 9}.h5
┣ 📂MI03A_Predictions_generate
┃ ┣ 📜short_Predictions_instances_Age_Abdomen_Pancreas_Contrast_InceptionResNetV2_1_1024_Adam_0.0001_0.1_0.5_1.0_test_{0... 9}.csv
┃ ┣ 📜short_Predictions_instances_Age_Abdomen_Pancreas_Contrast_InceptionResNetV2_1_1024_Adam_0.0001_0.1_0.5_1.0_train_{0... 9}.csv
┃ ┗ 📜short_Predictions_instances_Age_Abdomen_Pancreas_Contrast_InceptionResNetV2_1_1024_Adam_0.0001_0.1_0.5_1.0_val_{0... 9}.csv
┣ 📂MI03B_Predictions_concatenate
┃ ┣ 📜Predictions_instances_Age_Abdomen_Pancreas_Contrast_InceptionResNetV2_1_1024_Adam_0.0001_0.1_0.5_1.0_test.csv
┃ ┣ 📜Predictions_instances_Age_Abdomen_Pancreas_Contrast_InceptionResNetV2_1_1024_Adam_0.0001_0.1_0.5_1.0_train.csv
┃ ┗ 📜Predictions_instances_Age_Abdomen_Pancreas_Contrast_InceptionResNetV2_1_1024_Adam_0.0001_0.1_0.5_1.0_val.csv
┣ 📂MI03C_Predictions_merge
┃ ┣ 📜PREDICTIONS_withoutEnsembles_instances_Age_test.csv
┃ ┗ 📜PREDICTIONS_withoutEnsembles_instances_Age_val.csv
┣ 📂MI03D_Predictions_eids
┃ ┣ 📂Predictions_eids_concatenate
┃ ┃ ┣ 📜Predictions_instances_Age_Abdomen_Pancreas_Contrast_InceptionResNetV2_1_1024_Adam_0.0001_0.1_0.5_1.0_test.csv
┃ ┃ ┗ 📜Predictions_instances_Age_Abdomen_Pancreas_Contrast_InceptionResNetV2_1_1024_Adam_0.0001_0.1_0.5_1.0_val.csv
┃ ┣ 📜PREDICTIONS_withoutEnsembles_eids_Age_test.csv
┃ ┗ 📜PREDICTIONS_withoutEnsembles_eids_Age_val.csv
┣ 📂MI04A_Performances_generate
┃ ┣ 📜Performances_instances_Age_Abdomen_Pancreas_Contrast_InceptionResNetV2_1_1024_Adam_0.0001_0.1_0.5_1.0_test.csv
┃ ┣ 📜Performances_instances_Age_Abdomen_Pancreas_Contrast_InceptionResNetV2_1_1024_Adam_0.0001_0.1_0.5_1.0_test_sd.csv
┃ ┣ 📜Performances_instances_Age_Abdomen_Pancreas_Contrast_InceptionResNetV2_1_1024_Adam_0.0001_0.1_0.5_1.0_test_str.csv
┃ ┣ 📜Performances_instances_Age_Abdomen_Pancreas_Contrast_InceptionResNetV2_1_1024_Adam_0.0001_0.1_0.5_1.0_val.csv
┃ ┣ 📜Performances_instances_Age_Abdomen_Pancreas_Contrast_InceptionResNetV2_1_1024_Adam_0.0001_0.1_0.5_1.0_val_sd.csv
┃ ┗ 📜Performances_instances_Age_Abdomen_Pancreas_Contrast_InceptionResNetV2_1_1024_Adam_0.0001_0.1_0.5_1.0_val_str.csv
┣ 📂MI04B_Performances_merge
┃ ┣ 📜PERFORMANCES_withoutEnsembles_alphabetical_instances_Age_test.csv
┃ ┣ 📜PERFORMANCES_withoutEnsembles_alphabetical_instances_Age_val.csv
┃ ┣ 📜PERFORMANCES_withoutEnsembles_ranked_instances_Age_test.csv
┃ ┗ 📜PERFORMANCES_withoutEnsembles_ranked_instances_Age_val.csv
┣ 📂MI04C_Performances_tuning
┃ ┣ 📜PERFORMANCES_tuned_alphabetical_instances_Age_test.csv
┃ ┣ 📜PERFORMANCES_tuned_alphabetical_instances_Age_val.csv
┃ ┣ 📜PERFORMANCES_tuned_ranked_instances_Age_test.csv
┃ ┣ 📜PERFORMANCES_tuned_ranked_instances_Age_val.csv
┃ ┣ 📜PREDICTIONS_tuned_instances_Age_test.csv
┃ ┗ 📜PREDICTIONS_tuned_instances_Age_val.csv
┣ 📂MI05A_Ensembles_predictions
┃ ┣ 📜PREDICTIONS_withEnsembles_instances_Age_test.csv
┃ ┣ 📜PREDICTIONS_withEnsembles_instances_Age_val.csv
┃ ┣ 📜Predictions_instances_Age_*_*_*_*_*_*_*_*_*_*_*_test.csv
┃ ┣ 📜Predictions_instances_Age_*_*_*_*_*_*_*_*_*_*_*_val.csv
┃ ┣ 📜Predictions_instances_Age_*instances01_*_*_*_*_*_*_*_*_*_*_test.csv
┃ ┣ 📜Predictions_instances_Age_*instances01_*_*_*_*_*_*_*_*_*_*_val.csv
┃ ┣ 📜Predictions_instances_Age_*instances1.5x_*_*_*_*_*_*_*_*_*_*_test.csv
┃ ┣ 📜Predictions_instances_Age_*instances1.5x_*_*_*_*_*_*_*_*_*_*_val.csv
┃ ┣ 📜Predictions_instances_Age_*instances23_*_*_*_*_*_*_*_*_*_*_test.csv
┃ ┗ 📜Predictions_instances_Age_*instances23_*_*_*_*_*_*_*_*_*_*_val.csv
┣ 📂MI05B_Performances_generate
┃ ┣ 📜Performances_instances_Age_*_*_*_*_*_*_*_*_*_*_*_test.csv
┃ ┣ 📜Performances_instances_Age_*_*_*_*_*_*_*_*_*_*_*_test_sd.csv
┃ ┣ 📜Performances_instances_Age_*_*_*_*_*_*_*_*_*_*_*_test_str.csv
┃ ┣ 📜Performances_instances_Age_*_*_*_*_*_*_*_*_*_*_*_val.csv
┃ ┣ 📜Performances_instances_Age_*_*_*_*_*_*_*_*_*_*_*_val_sd.csv
┃ ┣ 📜Performances_instances_Age_*_*_*_*_*_*_*_*_*_*_*_val_str.csv
┃ ┣ 📜Performances_instances_Age_*instances23_*_*_*_*_*_*_*_*_*_*_test.csv
┃ ┣ 📜Performances_instances_Age_*instances23_*_*_*_*_*_*_*_*_*_*_test_sd.csv
┃ ┣ 📜Performances_instances_Age_*instances23_*_*_*_*_*_*_*_*_*_*_test_str.csv
┃ ┣ 📜Performances_instances_Age_*instances23_*_*_*_*_*_*_*_*_*_*_val.csv
┃ ┣ 📜Performances_instances_Age_*instances23_*_*_*_*_*_*_*_*_*_*_val_sd.csv
┃ ┗ 📜Performances_instances_Age_*instances23_*_*_*_*_*_*_*_*_*_*_val_str.csv
┣ 📂MI05C_Performances_merge
┃ ┣ 📜PERFORMANCES_withEnsembles_alphabetical_instances_Age_test.csv
┃ ┣ 📜PERFORMANCES_withEnsembles_alphabetical_instances_Age_val.csv
┃ ┣ 📜PERFORMANCES_withEnsembles_ranked_instances_Age_test.csv
┃ ┗ 📜PERFORMANCES_withEnsembles_ranked_instances_Age_val.csv
┣ 📂MI06A_Residuals_generate
┃ ┗ 📜RESIDUALS_instances_Age_test.csv
┣ 📂MI06B_Residuals_correlations
┃ ┣ 📜ResidualsCorrelations_instances_Age_test.csv
┃ ┣ 📜ResidualsCorrelations_samplesizes_instances_Age_test.csv
┃ ┣ 📜ResidualsCorrelations_sd_instances_Age_test.csv
┃ ┗ 📜ResidualsCorrelations_str_instances_Age_test.csv
┣ 📂MI07A_Select_best
┃ ┣ 📜PERFORMANCES_bestmodels_alphabetical_instances_Age_test.csv
┃ ┣ 📜PERFORMANCES_bestmodels_ranked_instances_Age_test.csv
┃ ┣ 📜PREDICTIONS_bestmodels_instances_Age_test.csv
┃ ┣ 📜RESIDUALS_bestmodels_instances_Age_test.csv
┃ ┣ 📜ResidualsCorrelations_bestmodels_instances_Age_test.csv
┃ ┣ 📜ResidualsCorrelations_bestmodels_sd_instances_Age_test.csv
┃ ┗ 📜ResidualsCorrelations_bestmodels_str_instances_Age_test.csv
┣ 📜fake_PA_visit_date.csv
┣ 📜fake_all_eids.csv
┣ 📜fake_short_ukb41230.csv
┗ 📜missing_samples.csv
```