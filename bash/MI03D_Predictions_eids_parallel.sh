#!/bin/bash
#targets=( "Age" "Sex" )
targets=( "Age" )
folds=( "train" "val" "test" )
#folds=( "train" )
#Ensure that the ensemble_model parameter was specified
if [[ ! ($1 == "True" || $1 == "False") ]]; then
    echo ERROR. Usage: ./MI03C05B_Predictions_eids_parallel.sh ensemble_models    ensemble_models must be either False to generate performances for simple models \(03C\), or True to generate performances for ensemble models \(05B\)
	exit
fi
ensemble_models=$1
time=20
memory=32G
n_cpu_cores=1
declare -a IDs=()
for fold in "${folds[@]}"; do
	for target in "${targets[@]}"; do
		version=MI03D_${target}_${fold}_${ensemble_models}
		job_name="$version.job"
		out_file="../eo/$version.out"
		err_file="../eo/$version.err"
		ID=$(sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cpu_cores -t $time MI03D_Predictions_eids.sh $target $fold $ensemble_models)
		IDs+=($ID)
	done
done
# Produce the list of job dependencies fr the next step
printf -v IDs_list '%s:' "${IDs[@]}"
dependencies="${IDs_list%:}"
echo $dependencies

