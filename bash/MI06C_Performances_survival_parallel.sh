#!/bin/bash
#define parameters
#targets=( "Age" "Sex" )
targets=( "Age" )
folds=( "test" )
pred_types=( "instances" "eids" )

#loop through the jobs to submit
declare -a IDs=()
for target in "${targets[@]}"; do
	for fold in "${folds[@]}"; do
		if [ $fold == "train" ]; then
			memory=64G
			time=120
		else
			memory=16G
			time=300
		fi
		for pred_type in "${pred_types[@]}"; do
			version=MI06C_${target}_${fold}_${pred_type}
			job_name="$version.job"
			out_file="../eo/$version.out"
			err_file="../eo/$version.err"
			ID=$(sbatch --dependency=$1 --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -t $time MI06C_Performances_survival.sh $target $fold $pred_type)
			IDs+=($ID)
		done
	done
done
# Produce the list of job dependencies fr the next step
printf -v IDs_list '%s:' "${IDs[@]}"
dependencies="${IDs_list%:}"
echo $dependencies

