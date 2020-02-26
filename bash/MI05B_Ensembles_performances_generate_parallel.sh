#!/bin/bash
regenerate_performances=true
memory=8G
n_cpu_cores=1
n_gpus=1

#generate file with list of ensemble models (avoids the trouble with parsing files with * character)
file_list_ensemble_models="../data/list_ensemble_models.txt"
ls ../data/Predictions_*_\*_* > $file_list_ensemble_models
ls ../data/Predictions_*_\?_* >> $file_list_ensemble_models
ls ../data/Predictions_*_\,_* >> $file_list_ensemble_models

#parse the file line by line to submit a job for each ensemble model
while IFS= read -r model
do
	IFS='_' read -ra PARAMETERS <<< ${model%".csv"}
	target="${PARAMETERS[1]}"
	image_type="${PARAMETERS[2]}_${PARAMETERS[3]}_${PARAMETERS[4]}"
	transformation="${PARAMETERS[5]}"
	architecture="${PARAMETERS[6]}"
	optimizer="${PARAMETERS[7]}"
	learning_rate="${PARAMETERS[8]}"
	weight_decay="${PARAMETERS[9]}"
	dropout_rate="${PARAMETERS[10]}"
	fold="${PARAMETERS[11]}"
	if [ "${PARAMETERS[2]}" == "PhysicalActivity" ]; then
		id_set="A"
	else
		id_set="B"
	fi
	version="${target}_${image_type}_${transformation}_${architecture}_${optimizer}_${learning_rate}_${weight_decay}_${dropout_rate}_${fold}_${id_set}"
	name=MI05B-"$version"
	job_name="$name.job"
	out_file="../eo/$name.out"
	err_file="../eo/$name.err"
	time=90
	time=10 #debug mode
	#allocate more time for the training fold because of the larger sample size
	if [ $fold = "train" ]; then
		time=$(( 8*$time ))
	fi
	#check if the predictions have already been generated. If not, do not run the model.
	if ! test -f "../data/Predictions_${version}.csv"; then
		echo The predictions at "../data/Predictions_${version}.csv" cannot be found. The job cannot be run.
		break
	fi
	#if regenerate_performances option is on or if the performances have not yet been generated, run the job
	if ! test -f "../data/Performances_${version}.csv" || $regenerate_performances; then
		echo Submitting job for "$version"
		#sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cpu_cores -t $time MI04A05B_Performances_generate.sh "${target}" "${image_type}" "${transformation}" "${architecture}" "${optimizer}" "${learning_rate}" "${weight_decay}" "${dropout_rate}" "${fold}" "${id_set}"
	#else
	#	echo Performance for $version have already been generated.
	fi
done < "$file_list_ensemble_models"

