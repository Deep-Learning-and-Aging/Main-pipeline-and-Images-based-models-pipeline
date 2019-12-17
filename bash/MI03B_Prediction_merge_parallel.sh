#!/bin/bash
targets=( "Age" "Sex" )
folds=( "train" "val" "test" )
memory=8G
n_cpu_cores=1
time=60
for target in "${targets[@]}"; do
	for fold in "${folds[@]}"; do
		version=MI03B_${target}_${fold}
		job_name="$version.job"
		out_file="../eo/$version.out"
		err_file="../eo/$version.err"
		sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cpu_cores -t $time MI03B_Prediction_merge.sh $target $fold
	done
done
