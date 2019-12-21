#!/bin/bash
targets=( "Age" "Sex" )
folds=( "train" "val" "test" )
memory=8G
n_cpu_cores=1
time=5
for target in "${targets[@]}"; do
	for fold in "${folds[@]}"; do
		if [ $fold = "train" ]; then
			time=$(( 8*$time ))
		fi
		version=MI04_${target}_${fold}
		job_name="$version.job"
		out_file="../eo/$version.out"
		err_file="../eo/$version.err"
		sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cpu_cores -t $time MI04_Performance.sh $target $fold
	done
done
