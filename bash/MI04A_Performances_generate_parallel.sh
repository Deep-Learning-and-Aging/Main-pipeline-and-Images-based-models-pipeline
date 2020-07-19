#!/bin/bash
regenerate_performances=false
#targets=( "Age" "Sex" )
targets=( "Age" )
folds=( "train" "val" "test" )
pred_types=( "instances" "eids" )
memory=2G
n_cpu_cores=1
n_gpus=1
declare -a IDs=()
# For reference, order of the organs (by similarity): Brain, Eyes, Hearing, Lungs, BloodPressure, Artery,  Carotids, Heart,  Abdomen, Spine, Hips, Knees, FullBody, Anthropometry, Heel, Hand, PhysicalActivity, BloodCount, BloodBiochemistry, Urine
organs_groups=( "Biomarkers" "TimeSeries" "Images" "Videos" )
for organs_group in "${targets[@]}"; do
	if [ $organs_group == "Biomarkers" ]; then
		organs=( "Brain" "Eyes" "Hearing" "Lungs" "BloodPressure" "Artery" "Carotids" "Heart" "Abdomen" "Anthropometry" "Heel" "Hand" "PhysicalActivity" "BloodCount" "BloodChemistry" "UrineChemistry" )
		architectures=( "ElasticNet" "LightGBM" "NeuralNetwork" )
		n_fc_layer="0"
		n_fc_nodes="0"
		optimizer="0"
		learning_rate="0"
		weight_decay="0"
		dropout_rate="0"
		data_augmentation_factor="0"
	elif [ $organs_group == "TimeSeries" ]; then
		organs=( "Artery" "Heart" "PhysicalActivity" )
		architectures=( TODO )
		n_fc_layer=TODO
		n_fc_nodes=TODO
		optimizer=TODO
		learning_rate=TODO
		weight_decay=TODO
		dropout_rate=TODO
		data_augmentation_factor=TODO
	elif [ $organs_group == "Images" ]; then
		organs=( "Brain" "Eyes" "Carotids" "Heart" "Abdomen" "Spine" "Hips" "Knees" "FullBody" )
		#architectures=( "VGG16" "VGG19" "MobileNet" "MobileNetV2" "DenseNet121" "DenseNet169" "DenseNet201" "NASNetMobile" "Xception" "InceptionV3" "InceptionResNetV2" )
		architectures=( "InceptionV3" )
		n_fc_layer="1"
		n_fc_nodes="1024"
		optimizer="Adam"
		learning_rate="0.001"
		weight_decay="0.1"
		dropout_rate="0.5"
		data_augmentation_factor="1.0"
	elif [ $organs_group == "Videos" ]; then
		organs=( "Heart" )
		architectures=( "3DCNN" )
		n_fc_layer=TODO
		n_fc_nodes=TODO
		optimizer=TODO
		learning_rate=TODO
		weight_decay=TODO
		dropout_rate=TODO
		data_augmentation_factor=TODO
	else
		echo "organs_group must be either Biomarkers, TimeSeries, Images, Videos"
	fi
	for target in "${targets[@]}"; do
		for organ in "${organs[@]}"; do
			if [ $organ == "Brain" ]; then
				if [ $organs_group == "Biomarkers" ]; then
					views=( "GreyMatterVolumes" "dMRIWeightedMeans" "SubcorticalVolumes" "AllBiomarkers" )
				elif [ $organs_group == "Images" ]; then
					views=( "Sagittal" "Coronal" "Transverse" )
				fi
			elif [ $organ == "Eyes" ]; then
				if [ $organs_group == Biomarkers ]; then
					views=( "Autorefraction" "Acuity" "IntraocularPressure" "AllBiomarkers" )
				elif [ $organs_group == "Images" ]; then
					views=( "Fundus" "OCT" )
				fi
			elif [ $organ == "Hearing" ]; then
				if [ $organs_group == "Biomarkers" ]; then
					views=( "HearingTest" )
				fi
			elif [ $organ == "Lungs" ]; then
				if [ $organs_group == "Biomarkers" ]; then
					views=( "Spirometry" )
				fi
			elif [ $organ == "BloodPressure" ]; then
				if [ $organs_group == "Biomarkers" ]; then
					views=( "Biomarkers" )
				fi
			elif [ $organ == "Artery" ]; then
				if [ $organs_group == "Biomarkers" ]; then
					views=( "Biomarker" )
				elif [ $organs_group == "TimeSeries" ]; then
					views=( "PWA" )
				fi
			elif [ $organ == "Carotids" ]; then
				if [ $organs_group == "Biomarkers" ]; then
					views=( "BiomarkersUltrasound" )
				elif [ $organs_group == "Images" ]; then
					views=( "Shortaxis" "Longaxis" "CIMT120" "CIMT150" "Mixed" )
				fi
			elif [ $organ == "Heart" ]; then
				if [ $organs_group == "Biomarkers" ]; then
					views=( "MRI" "ECG" )
				elif [ $organs_group == "TimeSeries" ]; then
					views=( "ECG" )
				elif [ $organs_group == "Images" ]; then
					views=( "MRI" )
				elif [ $organs_group == "Videos" ]; then
					views=( "MRI" )
				fi
			elif [ $organ == "Abdomen" ]; then
				if [ $organs_group == "Biomarkers" ]; then
					views=( "Liver" )
				elif [ $organs_group == "Images" ]; then
					views=( "Liver" "Pancreas" )
				fi
			elif [ $organ == "Spine" ]; then
				if [ $organs_group == "Images" ]; then
					views=( "Sagittal" "Coronal" )
				fi
			elif [ $organ == "Hips" ]; then
				if [ $organs_group == "Images" ]; then
					views=( "MRI" )
				fi
			elif [ $organ == "Knees" ]; then
				if [ $organs_group == "Images" ]; then
					views=( "MRI" )
				fi
			elif [ $organ == "FullBody" ]; then
				if [ $organs_group == "Images" ]; then
					views=( "Figure" "Skeleton" "Flesh" "Mixed" )
				fi
			elif [ $organ == "Anthropometry" ]; then
				if [ $organs_group == "Biomarkers" ]; then
					views=( "Impedance" "BodySize" "AllBiomarkers" )
				fi
			elif [ $organ == "Heel" ]; then
				if [ $organs_group == "Biomarkers" ]; then
					views=( "BoneDensitometry" )
				fi
			elif [ $organ == "Hand" ]; then
				if [ $organs_group == "Biomarkers" ]; then
					views=( "GripStrenght" )
				fi
			elif [ $organ == "PhysicalActivity" ]; then
				if [ $organs_group == "Biomarkers" ]; then
					views=( "Biomarkers" )
				elif [ $organs_group == "TimeSeries" ]; then
					views= ( TODO )
				fi
			elif [ $organ == "BloodCount" ]; then
				if [ $organs_group == "Biomarkers" ]; then
					views=( "BloodCount" )
				fi
			elif [ $organ == "BloodBiochemistry" ]; then
				if [ $organs_group == "Biomarkers" ]; then
					views=( "Biochemistry" )
				fi
			elif [ $organ == "Urine" ]; then
				if [ $organs_group == "Biomarkers" ]; then
					views=( "Biochemistry" )
				fi
			else
				echo "Organ $organ does not match any view!"
			fi
			if [ $organs_group == "Biomarkers" ]; then
				if [ $organ == "Heart" ]; then
					if [ $view == "MRI" ]; then
						transformations=( "Size" "PWA" "AllBiomarkers" )
					elif [ $view == "ECG" ]; then
						transformations=( "Biomarkers" )
					fi
				elif [ $organ == "Abdomen"]; then
					if [ $view == "Liver" ]; then
						transformations=( "Biomarkers" )
					fi
				else
					transformations=( "Raw" )
				fi
			elif [ $organs_group == "Images" ]; then
				if [ $organ == "Brain" ]; then
					transformations=( "Raw" "Reference" )
				elif [ $organ == "Heart" ]; then
					transformations=( "2chambersRaw" "2chambersContrast" "3chambersRaw" "3chambersContrast" "4chambersRaw" "4chambersContrast" )
				elif [ $organ == "Abdomen" ]; then
					transformations=( "Raw" "Contrast" )
				else
					transformations=( "Raw" )
				fi
			else
				transformations=( "Raw" )
			fi
			for view in "${views[@]}"; do
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
done

# Produce the list of job dependencies fr the next step
printf -v IDs_list '%s:' "${IDs[@]}"
dependencies="${IDs_list%:}"
echo $dependencies

