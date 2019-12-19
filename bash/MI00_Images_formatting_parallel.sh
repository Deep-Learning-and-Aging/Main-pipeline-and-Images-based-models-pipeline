#!/bin/bash
transformations=( "raw" "contrast" )
memory=8G
n_cpu_cores=1
time=150
for transformation in "${transformations[@]}"; do
	version=MI00_${transformation}
	job_name="$version.job"
	out_file="../eo/$version.out"
	err_file="../eo/$version.err"
	sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cpu_cores -t $time MI00_Images_formatting.sh $transformation
done
