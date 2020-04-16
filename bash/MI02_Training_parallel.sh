#!/bin/bash
#targets=( "Age" "Sex" )
targets=( "Age" )
#image_types=( "PhysicalActivity_90001_main" "Liver_20204_main" "Heart_20208_2chambers" "Heart_20208_3chambers" "Heart_20208_4chambers" "Brain_20227_sagittal" "Brain_20227_coronal" "Brain_20227_transverse" "EyeFundus_210156_right" "EyeFundus_210156_left" )
image_types=(  "Liver_20204_main" "Heart_20208_2chambers" "Heart_20208_3chambers" "Heart_20208_4chambers" )
#image_types=(  "Liver_20204_main" "Heart_20208_2chambers" "Heart_20208_3chambers" )
#image_types=(  "Brain_20227_sagittal" "Brain_20227_coronal" "Brain_20227_transverse" )
#image_types=(  "Liver_20204_main" )
image_types=(  "Heart_20208_3chambers" )
#image_types=( "EyeFundus_210156_right" "EyeFundus_210156_left" )
transformations_images=( "raw" "contrast" )
transformations_images=( "raw" )
#transformations_images=( "raw" "reference" )
#transformations_PA=( "raw" )
#architectures=( "VGG16" "VGG19" "MobileNet" "MobileNetV2" "DenseNet121" "DenseNet169" "DenseNet201" "NASNetMobile" "NASNetLarge" "Xception" "InceptionV3" "InceptionResNetV2" )
architectures=( "VGG16" "VGG19" "MobileNet" "MobileNetV2" "DenseNet121" "DenseNet169" "DenseNet201" "NASNetMobile" "Xception" "InceptionV3" "InceptionResNetV2" )
architectures=( "EfficientNetB7" )
#architectures=( "MobileNet" "MobileNetV2" "DenseNet121" "DenseNet169" "DenseNet201" "NASNetMobile" "Xception" "InceptionV3" "InceptionResNetV2" )
#architectures=( "VGG16" "VGG19" "NASNetLarge" )
#architectures=( "InceptionResNetV2" "DenseNet201" "Xception" "InceptionV3" )
#architectures=( "MobileNet" "MobileNetV2" "DenseNet121" "DenseNet169" )
#architectures=( "NASNetMobile" )
#architectures=( "InceptionResNetV2" )
#architectures=( "VGG16" )
#optimizers=( "Adam" "RMSprop" "Adadelta" )
optimizers=( "Adam" )
learning_rates=( "0.000001" )
#learning_rates=( "0.00001" "0.000001" "0.0000001" )
weight_decays=( "0.0" )
#weight_decays=( "0.0001" )
#weight_decays=( "0.0" "0.00001" "0.0001")
dropout_rates=( "0.0" )
#dropout_rates=( "0.2" )
#dropout_rates=( "0.0" "0.1" "0.2" )
outer_folds=( "0" "1" "2" "3" "4" "5" "6" "7" "8" "9" )
#outer_folds=( "1" "2" "3" "4" "5" "6" "7" "8" "9" )
#outer_folds=( "0" )
memory=8G
n_cpu_cores=1
n_gpus=1
time=600
#time=900
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
								for outer_fold in "${outer_folds[@]}"; do
								version=MI02_${target}_${image_type}_${transformation}_${architecture}_${optimizer}_${learning_rate}_${weight_decay}_${dropout_rate}_${outer_fold}
								job_name="$version.job"
								out_file="../eo/$version.out"
								err_file="../eo/$version.err"
								#if trying to compare the effect of a parameter (eg learning rate, optimizer...) it might be better to request K80 for all jobs
								if ! grep -q 'THE MODEL CONVERGED!' $out_file; then
									echo Submitting model: $version
									sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cpu_cores --gres=gpu:$n_gpus -t $time MI02_Training.sh $target $image_type $transformation $architecture $optimizer $learning_rate $weight_decay $dropout_rate $outer_fold
								#else
									#echo The model $version already converged.
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

