#!/bin/bash
#define parameters
#targets=( "Age" "Sex" )
targets=( "Age" )
folds=( "train" "val" "test" )
#id_sets=( "A" "B" )
id_sets=( "B" )
memory=8G
n_cpu_cores=1
time=15

#loop through the jobs to submit
for target in "${targets[@]}"; do
	for fold in "${folds[@]}"; do
		for id_set in "${id_sets[@]}"; do
			version=MI06B_${target}_${fold}_${id_set}
			job_name="$version.job"
			out_file="../eo/$version.out"
			err_file="../eo/$version.err"
			sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cpu_cores -t $time MI06B_Residuals_correlations.sh $target $fold $id_set
		done
	done
done

