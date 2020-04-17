#!/bin/bash
#Ensure that the plots_type parameter was specified
if [[ ! ($1 == "loggers" || $1 == "scatterplots") ]]; then
    echo ERROR. Usage: ./MI09_Plots_parallel.sh plots_type    plots_type must be "loggers" or "scatterplots".
	exit
fi
plots_type=$1
#targets=( "Age" "Sex" )
targets=( "Age" )
memory=8G
n_cpu_cores=1
if [[ $1 == "loggers" ]] ; then
    time=15
	memory=2G
else
	time=30
	memory=8G
fi

#loop through the jobs to submit
for target in "${targets[@]}"; do
	version=MI09AB_${plots_type}_${target}
	job_name="$version.job"
	out_file="../eo/$version.out"
	err_file="../eo/$version.err"
	sbatch --x11 --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cpu_cores -t $time MI09AB_Plots.sh $plots_type $target
done

