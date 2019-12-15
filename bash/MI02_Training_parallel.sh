#!/bin/bash
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
dropout_rates=( "0.0" )
#dropout_rates=( "0.1" "0.3" "0.5" "0.8" )
outer_folds=( "0" "1" "2" "3" "4" "5" "6" "7" "8" "9" )
outer_folds=( "1" )
memory=8G
n_cpu_cores=1
n_gpus=1
time=700
time=10
for target in "${targets[@]}"; do
	for image_type in "${image_types[@]}"; do
		for preprocessing in "${preprocessings[@]}"; do
			for architecture in "${architectures[@]}"; do
				for optimizer in "${optimizers[@]}"; do
					for learning_rate in "${learning_rates[@]}"; do
						for lambda in "${lambdas[@]}"; do
							for dropout_rate in "${dropout_rates[@]}"; do
								for outer_fold in "${outer_folds[@]}"; do
								version=M02-$target-$image_type-$preprocessing-$architecture-$optimizer-$learning_rate-$lambda-$dropout_rate-$outer_fold
								job_name="$version.job"
								out_file="../eo/$version.out"
								err_file="../eo/$version.err"
								#if trying to compare the effect of a parameter (eg learning rate, optimizer...) it might be better to request K80 for all jobs
								if ! grep -q 'THE MODEL CONVERGED!' $out_file; then
									echo Submitting model: $version
									sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cpu_cores --gres=gpu:$n_gpus -t $time MI02_Training.sh $target $image_type $preprocessing $architecture $optimizer $learning_rate $lambda $dropout_rate $outer_fold
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

