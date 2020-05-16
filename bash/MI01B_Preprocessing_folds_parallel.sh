#!/bin/bash
targets=( "Age" )
#image_fields=( "ECG_6025" "ECG_20205" "ArterialStiffness_4205" "Brain_20227" "EyeFundus_210156" "Liver_20204" "Heart_20208" ) #"PhysicalActivity_90001" )
image_fields=( "Liver_20204" "Heart_20208" "EyeFundus_210156" )
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
