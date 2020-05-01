#!/bin/bash
regenerate_predictions=false
#targets=( "Age" "Sex" )
targets=( "Age" )
#organ_fields=( "EyeFundus_210156" "Liver_20204" "Brain_20227" )
organs_fields=( "Liver_20204" "Heart_20208" )
organs_fields=( "Liver_20204" )
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
									version=${target}_${organ_id}_${view}_${transformation}_${architecture}_${optimizer}_${learning_rate}_${weight_decay}_${dropout_rate}
									name=MI03A_$version
									job_name="$name.job"
									out_file="../eo/$name.out"
									err_file="../eo/$name.err"
									if [ $architecture == "InceptionResNetV2" ]; then
										time=250
									else
										time=150
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
									#if regenerate_predictions option is on or if one of the predictions is missing, run the job
									to_run=false
									for fold in "${folds[@]}"; do
										path_predictions="../data/Predictions_${version}_${fold}.csv"
										if ! test -f $path_predictions; then
											to_run=true
										fi
									done
									if $regenerate_predictions; then
										to_run=true
									fi
									if $to_run; then
										echo Submitting job for $version
										sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cpu_cores --gres=gpu:$n_gpus -t $time MI03A_Predictions_generate.sh $target $organ_id $view $transformation $architecture $optimizer $learning_rate $weight_decay $dropout_rate
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

