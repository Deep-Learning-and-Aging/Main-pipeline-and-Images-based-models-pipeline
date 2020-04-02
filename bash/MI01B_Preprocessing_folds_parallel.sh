#!/bin/bash
targets=( "Age" )
#DO NOT RERUN. KEPT SAME INDEX AS WRONGAGE TO ALLOW TRANSFER LEARNING image_fields=( "Liver_20204" "Heart_20208" )
#image_fields=( "ECG_6025" "ECG_20205" "ArterialStiffness_4205" "Brain_20227" "PhysicalActivity_90001" )
image_fields=( "Brain_20227" )
memory=8G
n_cpu_cores=1
time=5
for target in "${targets[@]}"; do
	for image_field in "${image_fields[@]}"; do
		version=MI01B_${target}_${image_field}
		job_name="$version.job"
		out_file="../eo/$version.out"
		err_file="../eo/$version.err"
		sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cpu_cores -t $time MI01B_Preprocessing_folds.sh $target $image_field
	done
done
