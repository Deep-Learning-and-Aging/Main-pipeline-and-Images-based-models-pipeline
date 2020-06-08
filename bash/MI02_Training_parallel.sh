#!/bin/bash
#targets=( "Age" "Sex" )
targets=( "Age" )
#targets=( "Sex" )
organs=( "Brain" "Eyes" "Carotids" "Heart" "Liver" "Pancreas" "FullBody" "Spine" "Hips" "Knees" )
#organs=( "Eyes" )
#organs=( "Brain" "Eyes" "Carotids" "Heart" "Pancreas" "FullBody" "Spine" "Hips" "Knees" )
#organs=( "Heart" "Liver" ) # "Eyes" )
#architectures=( "VGG16" "VGG19" "DenseNet121" "DenseNet169" "DenseNet201" "Xception" "InceptionV3" "InceptionResNetV2" "EfficientNetB7" )
#architectures=( "DenseNet121" "DenseNet169" "DenseNet201" "Xception" "InceptionV3" "InceptionResNetV2" "EfficientNetB7" )
#architectures=( "InceptionResNetV2" )
#architectures=( "EfficientNetB7" "DenseNet201" )
#architectures=( "ResNext101" "Xception" "VGG19" )
#architectures=( "VGG16" "DenseNet121" "DenseNet169" "ResNet152V2" "InceptionV3" )
architectures=( "InceptionResNetV2" "InceptionV3" )
#architectures=( "InceptionV3" )
#architectures=( "InceptionResNetV2" )
#optimizers=( "Adam" "RMSprop" "Adadelta" )
optimizers=( "Adam" )
learning_rates=( "0.000001" )
#learning_rates=( "0.0000001" "0.00000001" "0.0000000001" )
#learning_rates=( "0.000001" "0.0000001" "0.00000001" "0.000000001" "0.0000000001")
weight_decays=( "0.0" )
#weight_decays=( "0.05" "0.1" "0.5" )
dropout_rates=( "0.15" "0.2" "0.25" )
#dropout_rates=( "0.2" )
#dropout_rates=( "0.1" "0.2" "0.3" "0.4" )
#outer_folds=( "0" "1" "2" "3" "4" "5" "6" "7" "8" "9" )
#outer_folds=( "2" "3" "4" "5" "6" "7" "8" "9" )
outer_folds=( "0" )
memory=8G
n_cpu_cores=1
n_gpus=1
time=600
#time=250
#time=10
for target in "${targets[@]}"; do
	for organ in "${organs[@]}"; do
		if [ $organ == "Brain" ]; then
			views=( "sagittal" "coronal" "transverse" )
			#views=( "sagittal" )
		elif [ $organ == "Eyes" ]; then
			views=( "fundus" "OCT" )
			#views=( "fundus" )
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
									for outer_fold in "${outer_folds[@]}"; do
										version=MI02_${target}_${organ}_${view}_${transformation}_${architecture}_${optimizer}_${learning_rate}_${weight_decay}_${dropout_rate}_${outer_fold}
										job_name="$version.job"
										out_file="../eo/$version.out"
										err_file="../eo/$version.err"
										#if trying to compare the effect of a parameter (eg learning rate, optimizer...) it might be better to request K80 for all jobs
										if ! grep -q 'Done.' $out_file; then
											echo Submitting model: $version
											sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cpu_cores --gres=gpu:$n_gpus -t $time MI02_Training.sh $target $organ $view $transformation $architecture $optimizer $learning_rate $weight_decay $dropout_rate $outer_fold
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
done

