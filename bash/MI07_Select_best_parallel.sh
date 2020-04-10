#!/bin/bash
#targets=( "Age" "Sex" )
targets=( "Age" )
memory=8G
n_cpu_cores=1
time=15
memory=8G
#loop through the jobs to submit
for target in "${targets[@]}"; do
	version=MI07_${target}
	job_name="$version.job"
	out_file="../eo/$version.out"
	err_file="../eo/$version.err"
	sbatch --x11 --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cpu_cores -t $time MI07_Select_best.sh $plots_type $target
done

