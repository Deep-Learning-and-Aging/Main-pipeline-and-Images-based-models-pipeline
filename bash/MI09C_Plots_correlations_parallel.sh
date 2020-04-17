#!/bin/bash
#targets=( "Age" "Sex" )
targets=( "Age" )
#folds=( "train" "val" "test" )
folds=( "test" )
memory=2G
n_cpu_cores=1
time=5

#loop through the jobs to submit
for target in "${targets[@]}"; do
	for fold in "${folds[@]}"; do
		version=MI09C_${target}_${fold}
		job_name="$version.job"
		out_file="../eo/$version.out"
		err_file="../eo/$version.err"
		sbatch --x11 --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cpu_cores -t $time MI09C_Plots_correlations.sh $target $fold
	done
done

