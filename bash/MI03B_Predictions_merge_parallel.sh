#!/bin/bash
#targets=( "Age" "Sex" )
targets=( "Age" )
folds=( "train" "val" "test" )
#folds=("train" )
n_cpu_cores=1
for fold in "${folds[@]}"; do
	if [ $fold == "train" ]; then
		time=15
		memory=8G
	else
		time=5
		memory=2G
	fi
	for target in "${targets[@]}"; do
		version=MI03B_${target}_${fold}
		job_name="$version.job"
		out_file="../eo/$version.out"
		err_file="../eo/$version.err"
		sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cpu_cores -t $time MI03B_Predictions_merge.sh $target $fold
	done
done

