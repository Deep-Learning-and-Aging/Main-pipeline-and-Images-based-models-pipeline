#!/bin/bash
regenerate_performances=false
#targets=( "Age" "Sex" )
targets=( "Age" )
# All organs
#organs=( "Brain" "Eyes" "Hearing" "Carotids" "Vascular" "BloodPressure" "Heart" "ECG" "Lungs" "Liver" "Pancreas" "FullBody" "Spine" "Hips" "Knees" "Heel" "Hand" "Anthropometry" "ImmuneSystem" "Blood" "Urine" )
# Biomarkers
organs=( "Brain" "Eyes" "Hearing" "Carotids" "Vascular" "BloodPressure" "Heart" "ECG" "Lungs" "Heel" "Hand" "Anthropometry" "ImmuneSystem" "Blood" "Urine" )
# Images
#organs=( "Brain" "Eyes" "Carotids" "Heart" "Liver" "Pancreas" "FullBody" "Spine" "Hips" "Knees" )
#organs=( "Carotids" )
n_fc_layersS=( "0" )
n_fc_nodesS=( "0" )
#optimizers=( "Adam" "RMSprop" "Adadelta" )
optimizers=( "Adam" )
optimizers=( "0" )
learning_rates=( "0.000001" )
learning_rates=( "0" )
weight_decays=( "0.0" )
weight_decays=( "0" )
dropout_rates=( "0" )
data_augmentation_factors=( "0" )
folds=( "train" "val" "test" )
#folds=( "val" "test" )
#folds=( "train" )
pred_types=( "instances" "eids" )
pred_types=( "instances" )
memory=2G
n_cpu_cores=1
n_gpus=1
declare -a IDs=()
for target in "${targets[@]}"; do
	for organ in "${organs[@]}"; do
		if [ $organ == "Brain" ]; then
			views=( "GreyMatterVolumes" "dMRIWeightedMeans" "SubcorticalVolumes" "AllBiomarkers" "sagittal" "coronal" "transverse" )
			#views=( "sagittal" )
			views=( "GreyMatterVolumes" "dMRIWeightedMeans" "SubcorticalVolumes" "AllBiomarkers" )
		elif [ $organ == "Eyes" ]; then
			views=( "Autorefraction" "Acuity" "IntraocularPressure" "AllBiomarkers" "fundus" "OCT" )
			views=( "Autorefraction" "Acuity" "IntraocularPressure" "AllBiomarkers" )
		elif [ $organ == "Hearing" ]; then
			views=( "HearingTest" )
        elif [ $organ == "Carotids" ]; then
			views=( "BiomarkersUltrasound" "shortaxis" "longaxis" "CIMT120" "CIMT150" "mixed" )
			views=( "BiomarkersUltrasound" )
		elif [ $organ == "Vascular" ]; then
			views=( "BiomarkersArterialStiffness" )
		elif [ $organ == "BloodPressure" ]; then
			views=( "Biomarkers" )
		elif [ $organ == "Heart" ]; then
			views=( "Size" "PWA" "AllBiomarkers" "2chambers" "3chambers" "4chambers" )
			#views=( "4chambers" )
			views=( "Size" "PWA" "AllBiomarkers" )
		elif [ $organ == "ECG" ]; then
			views=( "BiomarkersAtRest" )
		elif [ $organ == "Lungs" ]; then
			views=( "Spirometry" )
		elif [ $organ == "Liver" ]; then
			views=( "MRI" )
		elif [ $organ == "Pancreas" ]; then
			views=( "MRI" )
		elif [ $organ == "FullBody" ]; then
			views=( "figure" "skeleton" "flesh" "mixed" )
			#views=( "mixed" )
		elif [ $organ == "Spine" ]; then
			views=( "sagittal" "coronal" )
			#views=( "coronal" )
		elif [ $organ == "Hips" ]; then
			views=( "MRI" )
		elif [ $organ == "Knees" ]; then
			views=( "MRI" )
		elif [ $organ == "Heel" ]; then
			views=( "BoneDensitometry" )
		elif [ $organ == "Hand" ]; then
			views=( "GripStrenght" )
		elif [ $organ == "Anthropometry" ]; then
			views=( "Impedance" "BodySize" "AllBiomarkers" )
		elif [ $organ == "ImmuneSystem" ]; then
			views=( "BloodCount" )
		elif [ $organ == "Blood" ]; then
			views=( "Biochemistry" )
		elif [ $organ == "Urine" ]; then
			views=( "Biochemistry" )
		else
			echo "Organ $organ does not match any view!"
		fi
		if [ $organ == "Heart" ] || [ $organ == "Liver" ] || [ $organ == "Pancreas" ]; then
			transformations=( "raw" "contrast" )
            #transformations=( "raw" )
		else
			transformations=( "raw" )
		fi
		for view in "${views[@]}"; do
			organview="${organ}_${view}"
			if [ $organview == "Brain_sagittal" ] || [ $organview == "Brain_coronal" ] || [ $organview == "Brain_transverse" ] || [ $organview == "Eyes_fundus" ] || [ $organview == "Eyes_OCT" ] || [ $organview == "Carotids_shortaxis" ] || [ $organview == "Carotids_longaxis" ] || [ $organview == "Carotids_CIMT120" ] || [ $organview == "Carotids_CIMT150" ] || [ $organview == "Carotids_mixed" ] || [ $organview == "Heart_2chambers" ] || [ $organview == "Heart_3chambers" ] || [ $organview == "Heart_4chambers" ] || [ $organview == "Liver_MRI" ] || [ $organview == "Pancreas_MRI" ] || [ $organview == "FullBody_figure" ] || [ $organview == "FullBody_skeleton" ] || [ $organview == "FullBody_flesh" ] || [ $organview == "FullBody_mixed" ] || [ $organview == "Spine_sagittal" ] || [ $organview == "Spine_coronal" ] || [ $organview == "Hips_MRI" ] || [ $organview == "Knees_MRI" ] ; then
				architectures=( "VGG16" "VGG19" "MobileNet" "MobileNetV2" "DenseNet121" "DenseNet169" "DenseNet201" "NASNetMobile" "Xception" "InceptionV3"     "InceptionResNetV2" )
				architectures=( "InceptionV3" )
			else
				architectures=( "ElasticNet" "LightGbm" "NeuralNetwork" )
				#architectures=( "ElasticNet" )
			fi
			if [ $organview == "Heart_2chambers" ] || [ $organview == "Heart_3chambers" ] || [ $organview == "Heart_4chambers" ] || [ $organview == "Liver_MRI" ] || [ $organview == "Pancreas_MRI" ]; then
				transformations=( "raw" "contrast" )
  	          #transformations=( "raw" )
			else
				transformations=( "raw" )
			fi
			for transformation in "${transformations[@]}"; do
				for architecture in "${architectures[@]}"; do
					for n_fc_layers in "${n_fc_layersS[@]}"; do
						for n_fc_nodes in "${n_fc_nodesS[@]}"; do
							for optimizer in "${optimizers[@]}"; do
								for learning_rate in "${learning_rates[@]}"; do
									for weight_decay in "${weight_decays[@]}"; do
										for dropout_rate in "${dropout_rates[@]}"; do
											for data_augmentation_factor in "${data_augmentation_factors[@]}"; do
												for fold in "${folds[@]}"; do
													for pred_type in "${pred_types[@]}"; do
														version=${pred_type}_${target}_${organ}_${view}_${transformation}_${architecture}_${n_fc_layers}_${n_fc_nodes}_${optimizer}_${learning_rate}_${weight_decay}_${dropout_rate}_${data_augmentation_factor}_${fold}
														name=MI04A-$version
														job_name="$name.job"
														out_file="../eo/$name.out"
														err_file="../eo/$name.err"
														time=90
														time=20 #debug mode
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
															ID=$(sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cpu_cores -t $time MI04A05B_Performances_generate.sh $target $organ $view $transformation $architecture $n_fc_layers $n_fc_nodes $optimizer $learning_rate $weight_decay $dropout_rate $data_augmentation_factor $fold $pred_type)
															IDs+=($ID)
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
			done
		done
	done
done
# Produce the list of job dependencies fr the next step
printf -v IDs_list '%s:' "${IDs[@]}"
dependencies="${IDs_list%:}"
echo $dependencies

