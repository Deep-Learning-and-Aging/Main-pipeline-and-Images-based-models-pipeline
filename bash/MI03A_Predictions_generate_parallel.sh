#!/bin/bash
regenerate_predictions=false
#targets=( "Age" "Sex" )
targets=( "Age" )
organs_fields=( "Brain_20227" "Carotid_202223" "EyeFundus_210156" "EyeOCT_210178" "Heart_20208" "Liver_20204" "Pancreas_20259" "FullBody_201580" "Spine_201581" "Hip_201582" "Knee_201583" )
#organ_fields=( "EyeFundus_210156" "Liver_20204" "Brain_20227" )
#organs_fields=( "Liver_20204" "Heart_20208" )
organs_fields=( "EyeFundus_210156" )
architectures=( "VGG16" "VGG19" "DenseNet121" "DenseNet169" "DenseNet201" "Xception" "InceptionV3" "InceptionResNetV2" "EfficientNetB7" )
architectures=( "DenseNet201" "ResNext101" "InceptionResNetV2" "EfficientNetB7" )
architectures=( "VGG16" "VGG19" "DenseNet121" "DenseNet169" "ResNet152V2" "Xception" "InceptionV3" )
architectures=( "InceptionResNetV2" )
#optimizers=( "Adam" "RMSprop" "Adadelta" )
optimizers=( "Adam" )
learning_rates=( "0.000001" )
weight_decays=( "0.0" )
dropout_rates=( "0.0" )
#weight_decays=( "0.0" "0.0001" "0.001" )
#dropout_rates=( "0.0" "0.1" "0.2" )
folds=( "train" "val" "test" )
#folds=( "val" "test" )
outer_folds=( "0" "1" "2" "3" "4" "5" "6" "7" "8" "9" )
memory=8G
n_cpu_cores=1
n_gpus=1
for target in "${targets[@]}"; do
	for organ_field in "${organs_fields[@]}"; do
		if [ $organ_field == "Heart_20208" ]; then
			views=( "2chambers" "3chambers" "4chambers" )
		elif [ $organ_field == "Brain_20227" ]; then
			views=( "sagittal" "coronal" "transverse" )
        elif [ $organ_field == "Carotid_202223" ]; then
			views=( "longaxis" "shortaxis" "CIMT120" "CIMT150" "mixed" )
		elif [ $organ_field == "FullBody_201580" ]; then
			views=( "figure" "skeleton" "flesh" "mixed" )
		elif [ $organ_field == "Spine_201581" ]; then
			views=( "sagittal" "coronal" )
		else
			views=( "main" )
		fi
		if [ $organ_field == "Liver_20204" ] || [ $organ_field == "Heart_20208" ]; then
			transformations=( "raw" "contrast" )
		else
			transformations=( "raw" )
		fi
		for view in "${views[@]}"; do
			for transformation in "${transformations[@]}"; do
				for architecture in "${architectures[@]}"; do
					for optimizer in "${optimizers[@]}"; do
						for learning_rate in "${learning_rates[@]}"; do
							for weight_decay in "${weight_decays[@]}"; do
								for dropout_rate in "${dropout_rates[@]}"; do
									version=${target}_${organ_field}_${view}_${transformation}_${architecture}_${optimizer}_${learning_rate}_${weight_decay}_${dropout_rate}
									echo $version
									name=MI03A_$version
									job_name="$name.job"
									out_file="../eo/$name.out"
									err_file="../eo/$name.err"
									# time as a function of the dataset
									if [ $organ_field == "Carotid_20223" ]; then
										time=40 # 9k samples
									elif [ $organ_field == "Brain_20227" ] || [ $organ_field == "Liver_20204" ] || [ $organ_field == "Pancreas_20259" ] || [ $organ_field == "Heart_20208" ] || [ $organ_field == "FullBody_201580" ] || [ $organ_field == "Spine_201581" ] || [ $organ_field == "Hip_201582" ] || [ $organ_field == "Knee_201583" ]; then
										time=200 #45k samples
									elif [ $organ_field == "EyeFundus_210156" ] || [ $organ_field == "EyeOCT_210178" ]; then
										time=400 #90k samples
									fi
									# double the time for datasets for which each image is available for both the left and the right side
									if [ $organ_field == "Carotid_20223" ] || [ $organ_field == "EyeFundus_210156" ] || [ $organ_field == "EyeOCT_210178" ] || [ $organ_field == "Hip_201582" ] || [ $organ_field == "Knee_201583" ]; then
										time=$(( 2*$time ))
									fi
									# time multiplicator as a function of architecture
									if [ $architecture == "InceptionResNetV2" ]; then
										time=$(( 2*$time ))
									fi
									#check if all weights have already been generated. If not, do not run the model.
									missing_weights=false
									for outer_fold in "${outer_folds[@]}"; do
										path_weights="../data/model-weights_${version}_${outer_fold}.h5"
										if ! test -f $path_weights; then
											missing_weights=true
											echo The weights at $path_weights cannot be found. The job cannot be run.
											#some weights are missing despite having an associated .out file with "THE MODEL CONVERGED!"
											#delete these files to allow the model to be run during phase MI02.
											#rm "../eo/MI02_${version}_${outer_fold}.out"
											#rm "../eo/MI02_${version}_${outer_fold}.err"
											#break
										fi
									done
									if $missing_weights; then
										continue
									fi
									echo HERE
									#if regenerate_predictions option is on or if one of the predictions is missing, run the job
									to_run=false
									for fold in "${folds[@]}"; do
										path_predictions="../data/Predictions_${version}_${fold}.csv"
										echo $path_predictions
										if ! test -f $path_predictions; then
											echo $path_predictions
											to_run=true
										fi
									done
									echo $to_run
									if $regenerate_predictions; then
										to_run=true
									fi
									if $to_run; then
										echo Submitting job for $version
										#sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cpu_cores --gres=gpu:$n_gpus -t $time MI03A_Predictions_generate.sh $target $organ_id $view $transformation $architecture $optimizer $learning_rate $weight_decay $dropout_rate
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

