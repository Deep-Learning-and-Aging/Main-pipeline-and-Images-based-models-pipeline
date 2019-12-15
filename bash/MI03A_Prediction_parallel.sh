#!/bin/bash
#set -u
regenerate_predictions=false
targets=( "Age" "Sex" )
targets=( "Age" )
image_types=( "PhysicalActivity_90001_main" "Liver_20204_main" "Heart_20208_2chambers" "Heart_20208_3chambers" "Heart_20208_4chambers" )
image_types=( "PhysicalActivity_90001_main" )
preprocessings=( "raw" "contrast" )
preprocessings=( "raw" )
architectures=( "VGG16" "VGG19" "MobileNet" "MobileNetV2" "DenseNet121" "DenseNet169" "DenseNet201" "NASNetMobile" "NASNetLarge" "Xception" "InceptionV3" "InceptionResNetV2" )
architectures=( "VGG16" )
optimizers=( "Adam" "RMSprop" "Adadelta" )
optimizers=( "Adam" )
learning_rates=( "0.0001" )
lambdas=( "0.0" )
dropout_rates=( "0.1" "0.3" "0.5" "0.8" )
dropout_rates=( "0.0" )
folds=( "train" "val" "test" )
folds=( "val" "test" )
outer_folds=( "0" "1" "2" "3" "4" "5" "6" "7" "8" "9" )
#outer_folds=( "1" )
memory=8G
n_cpu_cores=1
n_gpus=1
time=100
for target in "${targets[@]}"; do
	for image_type in "${image_types[@]}"; do
		for preprocessing in "${preprocessings[@]}"; do
			for architecture in "${architectures[@]}"; do
				for optimizer in "${optimizers[@]}"; do
					for learning_rate in "${learning_rates[@]}"; do
						for lambda in "${lambdas[@]}"; do
							for dropout_rate in "${dropout_rates[@]}"; do
								version=${target}_${image_type}_${preprocessing}_${architecture}_${optimizer}_${learning_rate}_${lambda}_${dropout_rate}
								name=M03A-$version
								job_name="$name.job"
								out_file="../eo/$name.out"
								err_file="../eo/$name.err"
								#check if all weights have already been generated. If not, do not run the model.
								missing_weights=false
								for outer_fold in "${outer_folds[@]}"; do
									path_weights="../data/model-weights_${version}_${outer_fold}.h5"
									if ! test -f $path_weights; then
										missing_weights=true
										echo The weights at $path_weights cannot be found. The job cannot be run.
										break
									fi
								done
								if $missing_weights; then
									break
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
									#sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cpu_cores --gres=gpu:$n_gpus -t $time MI03A_Prediction.sh $target $image_type $preprocessing $architecture $optimizer $learning_rate $lambda $dropout_rate
								else
									echo Predictions for $version have already been generated.
								fi
							done
						done
					done
				done
			done
		done
	done
done

