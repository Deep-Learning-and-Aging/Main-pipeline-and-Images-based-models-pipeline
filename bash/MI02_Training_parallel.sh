#!/bin/bash
#targets=( "Age" "Sex" )
targets=( "Age" )
#targets=( "Sex" )
organs=( "Brain" "Eyes" "Carotids" "Heart" "Liver" "Pancreas" "FullBody" "Spine" "Hips" "Knees" )
organs=( "Heart" "FullBody" )
#organs=( "Brain" "Carotids" "Spine" "Hips" )
#architectures=( "VGG16" "VGG19" "DenseNet121" "DenseNet169" "DenseNet201" "Xception" "InceptionV3" "InceptionResNetV2" "EfficientNetB7" )
architectures=( "InceptionResNetV2" "InceptionV3" )
architectures=( "InceptionV3" )
#n_fc_layersS=( "0" "1" "2" "3" "4" "5" )
n_fc_layersS=( "0" "1" )
#n_fc_nodesS=( "16" "64" "128" "256" "512" "1024" )
n_fc_nodesS=( "1024" )
#optimizers=( "Adam" "RMSprop" "Adadelta" )
optimizers=( "Adam" )
#learning_rates=( "0.01" "0.001" "0.0001" "0.00001" "0.000001" )
learning_rates=( "0.0001" )
weight_decays=( "1.0" "10.0" )
weight_decays=( "0.0" )
dropout_rates=( "0.5" "0.95" )
data_augmentation_factors=( "1.0" )
#outer_folds=( "0" "1" "2" "3" "4" "5" "6" "7" "8" "9" )
outer_folds=( "0" )
memory=8G
n_cpu_cores=1
n_gpus=1
time=300
#time=250
for target in "${targets[@]}"; do
	for organ in "${organs[@]}"; do
		if [ $organ == "Brain" ]; then
			views=( "sagittal" "coronal" "transverse" )
			views=( "sagittal" )
		elif [ $organ == "Eyes" ]; then
			views=( "fundus" "OCT" )
			views=( "fundus" )
        elif [ $organ == "Carotids" ]; then
			views=( "longaxis" "shortaxis" "CIMT120" "CIMT150" "mixed" )
			views=( "longaxis" )
		elif [ $organ == "Heart" ]; then
			views=( "2chambers" "3chambers" "4chambers" )
			views=( "4chambers" )
		elif [ $organ == "FullBody" ]; then
			views=( "figure" "skeleton" "flesh" "mixed" )
			views=( "mixed" )
		elif [ $organ == "Spine" ]; then
			views=( "sagittal" "coronal" )
			views=( "coronal" )
		else
			views=( "main" )
		fi
		if [ $organ == "Heart" ] || [ $organ == "Liver" ] || [ $organ == "Pancreas" ]; then
			transformations=( "raw" "contrast" )
            transformations=( "contrast" )
		else
			transformations=( "raw" )
		fi
		for view in "${views[@]}"; do
			for transformation in "${transformations[@]}"; do
				for architecture in "${architectures[@]}"; do
					for n_fc_layers in "${n_fc_layersS[@]}"; do
						if [ $n_fc_layers == "0" ]; then
							n_fc_nodesS_amended=( "0" )
						else
							n_fc_nodesS_amended=( "${n_fc_nodesS[@]}" )
						fi
						for n_fc_nodes in "${n_fc_nodesS_amended[@]}"; do
							for optimizer in "${optimizers[@]}"; do
								for learning_rate in "${learning_rates[@]}"; do
									for weight_decay in "${weight_decays[@]}"; do
										for dropout_rate in "${dropout_rates[@]}"; do
											for outer_fold in "${outer_folds[@]}"; do
												for data_augmentation_factor in "${data_augmentation_factors[@]}"; do
													version=MI02_${target}_${organ}_${view}_${transformation}_${architecture}_${n_fc_layers}_${n_fc_nodes}_${optimizer}_${learning_rate}_${weight_decay}_${dropout_rate}_${data_augmentation_factor}_${outer_fold}
													job_name="$version.job"
													out_file="../eo/$version.out"
													err_file="../eo/$version.err"
													if ! test -f "$out_file" || ! grep -q "Done." "$out_file" || grep -q " improved from " "$out_file"; then
														similar_models=MI02_${target}_${organ}_${view}_${transformation}_${architecture}_${n_fc_layers}_${n_fc_nodes}_${optimizer}_${learning_rate}_${weight_decay}_${dropout_rate}_*_${outer_fold}	
														#if [ $(sacct -u al311 --format=JobID,JobName%100,MaxRSS,NNodes,Elapsed,State | grep $similar_models | egrep 'PENDING|RUNNING' | wc -l) -eq 0 ]; then
															echo SUBMITTING: $version
															sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cpu_cores --gres=gpu:$n_gpus -t $time MI02_Training.sh $target $organ $view $transformation $architecture $n_fc_layers $n_fc_nodes $optimizer $learning_rate $weight_decay $dropout_rate $data_augmentation_factor $outer_fold $time
														#else
														#	echo "Pending/Running: $version (or similar model)"
														#fi
													else
														echo "Already converged: $version"
													fi
												done
											done
										done
									done
								done
							done
						done
					done
				done
			done
		done
	done
done

