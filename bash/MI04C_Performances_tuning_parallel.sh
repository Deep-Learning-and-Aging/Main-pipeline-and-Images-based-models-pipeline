#!/bin/bash
targets=( "Age" "Sex" )
targets=( "Age" )
pred_types=( "instances" "eids" )
memory=8G
n_cpu_cores=1
time=60
for target in "${targets[@]}"; do
	for pred_type in "${pred_types[@]}"; do
			version=MI04C_${target}_${pred_type}
			job_name="$version.job"
			out_file="../eo/$version.out"
			err_file="../eo/$version.err"
			sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cpu_cores -t $time MI04C_Performances_tuning.sh $target $pred_type
	done
done

