#!/bin/bash
image_fields=( "PhysicalActivity_90001" "Liver_20204" "Heart_20208" )
image_fields=( "Heart_20208" )
targets=( "Age" )
memory=8G
n_cpu_cores=1
time=60
for image_field in "${image_fields[@]}"
do
for target in "${targets[@]}"
do
version=MI01-$image_field-$target
job_name="$version.job"
out_file="../eo/$version.out"
err_file="../eo/$version.err"
sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cpu_cores -t $time MI01_Preprocessing.sh $image_field $target
done
done
