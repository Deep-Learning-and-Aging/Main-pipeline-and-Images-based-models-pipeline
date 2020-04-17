#!/bin/bash
#targets=( "Age" "Sex" )
targets=( "Age" )
organs_ids_views=( "Liver_20204_main" "Heart_20208_2chambers" "Heart_20208_3chambers" "Heart_20208_4chambers")
#organs_ids_views=( "Liver_20204_main" )
#transformations=( "raw" "contrast" )
transformations=( "raw" )
folds=( "train" "val" "test" )
folds=( "test" )
memory=8G
n_cpu_cores=1
time=30

#loop through the jobs to submit
for target in "${targets[@]}"; do
	for organ_id_view in "${organs_ids_views[@]}"; do
		for transformation in "${transformations[@]}"; do
			for fold in "${folds[@]}"; do
				version=MI09D_${target}_${organ_id_view}_${transformation}_${fold}
				job_name="$version.job"
				out_file="../eo/$version.out"
				err_file="../eo/$version.err"
				sbatch --x11 --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cpu_cores -t $time MI09D_Plots_attentionmaps.sh $target $organ_id_view $transformation $fold
			done
		done
	done
done

