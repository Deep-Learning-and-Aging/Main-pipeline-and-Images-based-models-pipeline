#!/bin/bash
regenerate_performances=false
#targets=( "Age" "Sex" )
targets=( "Age" )
#image_types=( "PhysicalActivity_90001_main" "Liver_20204_main" "Heart_20208_2chambers" "Heart_20208_3chambers" "Heart_20208_4chambers" )
image_types=( "Liver_20204_main" "Heart_20208_2chambers" "Heart_20208_3chambers" "Heart_20208_4chambers" )
#image_types=( "Heart_20208_3chambers" )
#image_types=( "Heart_20208_2chambers" "Heart_20208_3chambers" "Heart_20208_4chambers" )
transformations_images=( "raw" "contrast" )
#transformations_images=( "raw" )
transformations_PA=( "raw" )
#architectures=( "VGG16" "VGG19" "MobileNet" "MobileNetV2" "DenseNet121" "DenseNet169" "DenseNet201" "NASNetMobile" "NASNetLarge" "Xception" "InceptionV3" "InceptionResNetV2" )
architectures=( "VGG16" "VGG19" "MobileNet" "MobileNetV2" "DenseNet121" "DenseNet169" "DenseNet201" "NASNetMobile" "Xception" "InceptionV3" "InceptionResNetV2" )
#architectures=( "MobileNet" "MobileNetV2" "DenseNet121" "DenseNet169" "DenseNet201" "NASNetMobile" "InceptionV3" "InceptionResNetV2" )
#architectures=( "Xception" )
#optimizers=( "Adam" "RMSprop" "Adadelta" )
optimizers=( "Adam" )
learning_rates=( "0.000001" )
weight_decays=( "0.0" )
#dropout_rates=( "0.1" "0.3" "0.5" "0.8" )
dropout_rates=( "0.0" )
#weight_decays=( "0.0" "0.0001" "0.001" "0.01" "0.1" )
#dropout_rates=( "0.0" "0.1" "0.3" "0.5" "0.8" "0.95")
folds=( "train" "val" "test" )
#folds=( "val" "test" )
memory=8G
n_cpu_cores=1
n_gpus=1
for target in "${targets[@]}"; do
	for image_type in "${image_types[@]}"; do
		if [ $image_type == "PhysicalActivity_90001_main" ]; then
			transformations=("${transformations_PA[@]}")
		else
			transformations=("${transformations_images[@]}")
		fi
		for transformation in "${transformations[@]}"; do
			for architecture in "${architectures[@]}"; do
				for optimizer in "${optimizers[@]}"; do
					for learning_rate in "${learning_rates[@]}"; do
						for weight_decay in "${weight_decays[@]}"; do
							for dropout_rate in "${dropout_rates[@]}"; do
								for fold in "${folds[@]}"; do
									version=${target}_${image_type}_${transformation}_${architecture}_${optimizer}_${learning_rate}_${weight_decay}_${dropout_rate}_${fold}
									name=MI04A-$version
									job_name="$name.job"
									out_file="../eo/$name.out"
									err_file="../eo/$name.err"
									time=90
									time=10 #debug mode
									#allocate more time for the training fold because of the larger sample size
									if [ $fold = "train" ]; then
										time=$(( 8*$time ))
									fi
									#check if the predictions have already been generated. If not, do not run the model.
									if ! test -f "../data/Predictions_${version}.csv"; then
										echo The predictions at "../data/Predictions_${version}.csv" cannot be found. The job cannot be run.
										break
									fi
									#if regenerate_performances option is on or if the performances have not yet been generated, run the job
									if ! test -f "../data/Performances_${version}.csv" || $regenerate_performances; then
										echo Submitting job for $version
										sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cpu_cores -t $time MI04A05B_Performances_generate.sh $target $image_type $transformation $architecture $optimizer $learning_rate $weight_decay $dropout_rate $fold
									#else
									#	echo Performance for $version have already been generated.
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

