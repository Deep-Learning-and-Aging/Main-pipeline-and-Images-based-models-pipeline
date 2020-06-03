#!/bin/bash
#define parameters
#targets=( "Age" "Sex" )
targets=( "Age" )
folds=( "train" "val" "test" )
pred_types=( "instances" "eids" )
memory=8G
n_cpu_cores=1
time=15

#loop through the jobs to submit
for target in "${targets[@]}"; do
	for fold in "${folds[@]}"; do
		for pred_type in "${pred_types[@]}"; do
			version=MI06B_${target}_${fold}_${pred_type}
			job_name="$version.job"
			out_file="../eo/$version.out"
			err_file="../eo/$version.err"
			sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cpu_cores -t $time MI06B_Residuals_correlations.sh $target $fold $pred_type
		done
	done
done

