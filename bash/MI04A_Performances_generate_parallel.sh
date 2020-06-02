#!/bin/bash
regenerate_performances=false
#targets=( "Age" "Sex" )
targets=( "Age" )
organs=( "Brain" "Eyes" "Carotids" "Heart" "Liver" "Pancreas" "FullBody" "Spine" "Hips" "Knees" )
#organs=( "Liver" )
#architectures=( "VGG16" "VGG19" "MobileNet" "MobileNetV2" "DenseNet121" "DenseNet169" "DenseNet201" "NASNetMobile" "NASNetLarge" "Xception" "InceptionV3" "InceptionResNetV2" )
architectures=( "VGG16" "VGG19" "MobileNet" "MobileNetV2" "DenseNet121" "DenseNet169" "DenseNet201" "NASNetMobile" "Xception" "InceptionV3" "InceptionResNetV2" )
#architectures=( "MobileNet" "MobileNetV2" "DenseNet121" "DenseNet169" "DenseNet201" "NASNetMobile" "InceptionV3" "InceptionResNetV2" )
architectures=( "InceptionV3" )
#optimizers=( "Adam" "RMSprop" "Adadelta" )
optimizers=( "Adam" )
learning_rates=( "0.000001" )
weight_decays=( "0.0" )
#dropout_rates=( "0.1" "0.3" "0.5" "0.8" )
dropout_rates=( "0.2" )
#weight_decays=( "0.0" "0.0001" "0.001" "0.01" "0.1" )
#dropout_rates=( "0.0" "0.1" "0.3" "0.5" "0.8" "0.95")
folds=( "train" "val" "test" )
#folds=( "val" "test" )
memory=8G
n_cpu_cores=1
n_gpus=1
for target in "${targets[@]}"; do
	for organ in "${organs[@]}"; do
		if [ $organ == "Brain" ]; then
			views=( "sagittal" "coronal" "transverse" )
			#views=( "sagittal" )
		elif [ $organ == "Eyes" ]; then
			views=( "fundus" "OCT" )
        elif [ $organ == "Carotids" ]; then
			views=( "longaxis" "shortaxis" "CIMT120" "CIMT150" "mixed" )
			#views=( "longaxis" )
		elif [ $organ == "Heart" ]; then
			views=( "2chambers" "3chambers" "4chambers" )
			#views=( "4chambers" )		
		elif [ $organ == "FullBody" ]; then
			views=( "figure" "skeleton" "flesh" "mixed" )
			#views=( "mixed" )
		elif [ $organ == "Spine" ]; then
			views=( "sagittal" "coronal" )
			#views=( "coronal" )
		else
			views=( "main" )
		fi
		if [ $organ == "Heart" ] || [ $organ == "Liver" ] || [ $organ == "Pancreas" ]; then
			transformations=( "raw" "contrast" )
            #transformations=( "raw" )
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
									for fold in "${folds[@]}"; do
										version=${target}_${organ}_${view}_${transformation}_${architecture}_${optimizer}_${learning_rate}_${weight_decay}_${dropout_rate}_${fold}
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
											sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cpu_cores -t $time MI04A05C_Performances_generate.sh $target $organ $view $transformation $architecture $optimizer $learning_rate $weight_decay $dropout_rate $fold
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
done
