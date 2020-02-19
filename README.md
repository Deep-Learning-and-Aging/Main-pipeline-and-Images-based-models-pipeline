Predicting Age and Sex using different kinds of medical images from UKB, and deep learning.

There are four directories.

scripts: where the python script can be written. This is the core of the pipeline.

bash: Where the bash scripts to submit the jobs using slurm are written. These are the wrapper that allow the pipeline to be run in parallel. Usually one script is named parallel and is going to call the script with the same name without "\_parallel"  to submit the a job that will be calling the python scripts with different arguments.

eo: Where the error and output files are stored, to keep track of successful and failed jobs, and to help with debugging.

data: Where the images used to predict the phenotypes are stored. They must be stored in a 4 levels/folders arborescence. The first level is the organ, the second level is the ukb field_id, the third level is the view (for example for Heart_20204 three different axis views are available), and the fourth level is the transformation of the images (they can be raw of preprocessed, for example to enhance the contrast).

The following are the steps that should be taken to run the pipeline after that the data has been placed in the folder as described before under "data:".

Step 00: MI00_Images_formatting_parallel.sh: Preprocesses the images for the pipeline. Currently this only preprocesses the hart images to merge the 3 axis as a single image.

Step 01: MI01_Preprocessing_parallel.sh: For each combination of organ, image_field and target, preprocesses the data. Mostly, splits the data between the different cross validation folds. It is possible to share IDs for different targets or image_fields. Currently, the same split is shared for the targets "Age" and "Sex", so the pipeline is only run for "Age".

Step 02: MI02_Training_parallel.sh: This is the main and the most time consuming step. For each combination of target, predictors, and hyperparameters, train a model. If the model has already be trained to convergence, no job will be resubmitted. This way it is possible to keep rerunning step 2 until all models have converged. Sometimes increasing the time of the job is helpful, so that the minimum number of epochs without improvement to conclude about convergence can be reached.

Step 03A: MI03A_Predictions_generate_parallel.sh: Generates the prediction for every model for which all the outer cross validations folds were generated. There is an option to only run jobs for which predictions do not exist already. This allows Step 02 and Step 03 to be run partly in parallel.

Step 03B: MI03B_Predictions_merge_parallel.sh: Merges all the predictions generated into a large dataset to centralize them. Predictions from side pipelines (e.g Age=f(Biomarkers)) can also be integrated here.

Step 04: MI04A_Performances_generate_parallel.sh: 
