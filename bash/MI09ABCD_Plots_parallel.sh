#!/bin/bash
#Ensure that the plots_type parameter was specified
if [[ ! ($1 == "loggers" || $1 == "scatterplots" || $1 == "saliencymaps" || $1 == "correlations") ]]; then
    echo ERROR. Usage: ./MI09_Plots_parallel.sh plots_type    plots_type must be "loggers", "scatterpots", "saliencymaps" or "correlations".
	exit
fi
plots_type=$1
targets=( "Age" "Sex" )
targets=( "Age" )
id_sets=( "A" "B" )
id_sets=( "B" )
memory=8G
n_cpu_cores=1
time=200
#loop through the jobs to submit
for target in "${targets[@]}"; do
	for id_set in "${id_sets[@]}"; do
		version=MI09_${plots_type}_${target}_${id_set}
		job_name="$version.job"
		out_file="../eo/$version.out"
		err_file="../eo/$version.err"
		sbatch --x11 --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cpu_cores -t $time MI09ABCD_Plot.sh $plots_type $target $id_set
	done
done

