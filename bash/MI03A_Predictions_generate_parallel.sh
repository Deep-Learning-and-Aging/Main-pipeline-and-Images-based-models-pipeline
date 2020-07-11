#!/bin/bash
regenerate_predictions=false
#targets=( "Age" "Sex" )
targets=( "Age" )
organs=( "Brain" "Eyes" "Carotids" "Heart" "Abdomen" "Spine" "Hips" "Knees" "FullBody" )
organs=( "Liver" )
architectures=( "VGG16" "VGG19" "DenseNet121" "DenseNet169" "DenseNet201" "Xception" "InceptionV3" "InceptionResNetV2" "EfficientNetB7" )
architectures=( "DenseNet201" "ResNext101" "InceptionResNetV2" "EfficientNetB7" )
architectures=( "InceptionV3" )
#optimizers=( "Adam" "RMSprop" "Adadelta" )
optimizers=( "Adam" )
learning_rates=( "0.0001" )
weight_decays=( "0.1" )
dropout_rates=( "0.5" )
data_augmentation_factors=( "1.0")
folds=( "train" "val" "test" )
#folds=( "val" "test" )
outer_folds=( "0" "1" "2" "3" "4" "5" "6" "7" "8" "9" )
outer_folds=( "0" )
memory=8G
n_cpu_cores=1
n_gpus=1
declare -a IDs=()
for target in "${targets[@]}"; do
	for organ in "${organs[@]}"; do
		if [ $organ == "Brain" ]; then
			views=( "Sagittal" "Coronal" "Transverse" )
		elif [ $organ == "Eyes" ]; then
			views=( "Fundus" "OCT" )
		elif [ $organ == "Carotids" ]; then
			views=( "Shortaxis" "Longaxis" "CIMT120" "CIMT150" "Mixed" )
		elif [ $organ == "Heart" ]; then
			views=( "MRI" )
		elif [ $organ == "Abdomen" ]; then
			views=( "Liver" "Pancreas" )
		elif [ $organ == "Spine" ]; then
			views=( "Sagittal" "Coronal" )
		elif [ $organ == "FullBody" ]; then
			views=( "Figure" "Skeleton" "Flesh" "Mixed" )
		else
			views=( "MRI" )
		fi
		if [ $organ == "Brain" ]; then
			transformations=( "Raw" "Reference" )
		elif [ $organ == "Heart" ]; then
			transformations=( "2chambersRaw" "2chambersContrast" "3chambersRaw" "3chambersContrast" "4chambersRaw" "4chambersContrast" )
		elif [ $organ == "Abdomen" ]; then
			transformations=( "Raw" "Contrast" )
		else
			transformations=( "Raw" )
		fi
		for view in "${views[@]}"; do
			for transformation in "${transformations[@]}"; do
				for architecture in "${architectures[@]}"; do
					for optimizer in "${optimizers[@]}"; do
						for n_fc_layers
						for learning_rate in "${learning_rates[@]}"; do
							for weight_decay in "${weight_decays[@]}"; do
								for dropout_rate in "${dropout_rates[@]}"; do
									for outer_fold in "${outer_folds[@]}"; do
										version=${target}_${organ}_${view}_${transformation}_${architecture}_${optimizer}_${learning_rate}_${weight_decay}_${dropout_rate}_${data_augmentation_factor}_${outer_fold}
										name=MI03A_$version
										job_name="$name.job"
										out_file="../eo/$name.out"
										err_file="../eo/$name.err"
										# time as a function of the dataset
										if [ $organ == "Carotids" ]; then
											time=40 # 9k samples
											time=10
										elif [ $organ == "Brain" ] || [ $organ == "Liver" ] || [ $organ == "Pancreas" ] || [ $organ == "Heart" ] || [ $organ == "FullBody" ] || [ $organ == "Spine" ] || [ $organ == "Hips" ] || [ $organ == "Knees" ]; then
											time=300 #45k samples
											time=90
										elif [ $organ == "Eyes" ]; then
											time=600 #90k samples
											time=170
										fi
										# double the time for datasets for which each image is available for both the left and the right side
										if [ $organ == "Carotids" ] || [ $organ == "Eyes" ] || [ $organs == "Hips" ] || [ $organs == "Knees" ]; then
											time=$(( 2*$time ))
										fi
										# time multiplicator as a function of architecture
										if [ $architecture == "InceptionResNetV2" ]; then
											time=$(( 2*$time ))
										fi
										#check if all weights have already been generated. If not, do not run the model.
										path_weights="../data/model-weights_${version}.h5"
										if ! test -f $path_weights; then
											echo The weights at $path_weights cannot be found. The job cannot be run.
											continue
										fi
										#if regenerate_predictions option is on or if one of the predictions is missing, run the job
										to_run=false
										for fold in "${folds[@]}"; do
											path_predictions="../data/Predictions_instances_${target}_${organ}_${view}_${transformation}_${architecture}_${optimizer}_${learning_rate}_${weight_decay}_${dropout_rate}_${data_augmentation_factor}_${fold}_${outer_fold}.csv"
											if ! test -f $path_predictions; then
												to_run=true
											fi
										done
										if $regenerate_predictions; then
											to_run=true
										fi
										if $to_run; then
											echo Submitting job for $version
											ID=$(sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cpu_cores --gres=gpu:$n_gpus -t $time MI03A_Predictions_generate.sh $target $organ $view $transformation $architecture $optimizer $learning_rate $weight_decay $dropout_rate $data_augmentation_factor $outer_fold)
											IDs+=($ID)
										#else
										#	echo Predictions for $version have already been generated.
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
# Produce the list of job dependencies fr the next step
printf -v IDs_list '%s:' "${IDs[@]}"
dependencies="${IDs_list%:}"
echo $dependencies

