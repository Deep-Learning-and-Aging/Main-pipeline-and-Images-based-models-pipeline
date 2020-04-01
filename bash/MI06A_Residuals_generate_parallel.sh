#!/bin/bash
#define parameters
#targets=( "Age" "Sex" )
targets=( "Age" )
folds=( "train" "val" "test" )
memory=8G
n_cpu_cores=1
time=15

#loop through the jobs to submit
for target in "${targets[@]}"; do
	for fold in "${folds[@]}"; do
		version=MI06A_${target}_${fold}
		job_name="$version.job"
		out_file="../eo/$version.out"
		err_file="../eo/$version.err"
		sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cpu_cores -t $time MI06A_Residuals_generate.sh $target $fold
	done
done

