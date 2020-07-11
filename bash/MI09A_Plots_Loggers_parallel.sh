#!/bin/bash
#targets=( "Age" "Sex" )
targets=( "Age" )
memory=8G
n_cpu_cores=1
time=15
memory=2G

#loop through the jobs to submit
declare -a IDs=()
for target in "${targets[@]}"; do
	version=MI09A_${target}
	job_name="$version.job"
	out_file="../eo/$version.out"
	err_file="../eo/$version.err"
	ID=$(sbatch --x11 --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cpu_cores -t $time MI09A_Plots_loggers.sh $target)
	IDs+=($ID)
done
# Produce the list of job dependencies fr the next step
printf -v IDs_list '%s:' "${IDs[@]}"
dependencies="${IDs_list%:}"
echo $dependencies

