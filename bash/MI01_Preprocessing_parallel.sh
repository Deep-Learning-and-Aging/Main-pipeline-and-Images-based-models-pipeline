#!/bin/bash
targets=( "Age" )
image_fields=( "PhysicalActivity_90001" "Liver_20204" "Heart_20208" )
image_fields=( "PhysicalActivity_90001" )
memory=8G
n_cpu_cores=1
time=60
for target in "${targets[@]}"; do
	for image_field in "${image_fields[@]}"; do
		version=MI01-$target-$image_field
		job_name="$version.job"
		out_file="../eo/$version.out"
		err_file="../eo/$version.err"
		sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cpu_cores -t $time MI01_Preprocessing.sh $target $image_field
	done
done
