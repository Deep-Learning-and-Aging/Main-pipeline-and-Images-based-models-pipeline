#!/bin/bash
targets=( "Age" )
organs=( "Brain" "Carotids" "Eyes" "Heart" "Liver" "Pancreas" "FullBody" "Spine" "Hips" "Knees" ) #"PhysicalActivity" )
organs=( "Brain" "Carotids" "Eyes" "Liver" "Pancreas" "FullBody" "Spine" "Hips" "Knees" )
organs=( "Heart" )
n_cpu_cores=1
time=5
memory=8G
dependencies=
for target in "${targets[@]}"; do
	for organ in "${organs[@]}"; do
		version=MI01B_${target}_${organ}
		job_name="$version.job"
		out_file="../eo/$version.out"
		err_file="../eo/$version.err"
		sbatch --dependency=$1 --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cpu_cores -t $time MI01B_Preprocessing_folds.sh $target $organ
	done
done
