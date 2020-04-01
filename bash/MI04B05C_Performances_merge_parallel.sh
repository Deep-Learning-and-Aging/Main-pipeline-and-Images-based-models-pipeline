#!/bin/bash
#define parameters
#targets=( "Age" "Sex" )
targets=( "Age" )
folds=( "train" "val" "test" )
#Ensure that the ensemble_model parameter was specified
if [[ ! ($1 == "True" || $1 == "False") ]]; then
    echo ERROR. Usage: ./MI04B05C_Performance_merge_parallel.sh ensemble_models    ensemble_models must be either False to generate performances for simple models \(04B\), or True to generate performances for ensemble models \(05C\)
	exit
fi
ensemble_models=$1
memory=8G
n_cpu_cores=1
time=5

#loop through the jobs to submit
for target in "${targets[@]}"; do
	for fold in "${folds[@]}"; do
		version=MI04B05C_${target}_${fold}
		job_name="$version.job"
		out_file="../eo/$version.out"
		err_file="../eo/$version.err"
		sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cpu_cores -t $time MI04B05C_Performances_merge.sh $target $fold $ensemble_models
	done
done

