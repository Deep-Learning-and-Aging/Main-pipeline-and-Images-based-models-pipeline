#!/bin/bash
regenerate_performances=false
#targets=( "Age" "Sex" )
targets=( "Age" )
folds=( "train" "val" "test" )
folds=( "val" "test" )
pred_types=( "instances" "eids" )
#pred_types=( "instances" )
memory=2G
n_cpu_cores=1
n_gpus=1
declare -a IDs=()
# For reference, order of the organs (by similarity): Brain, Eyes, Hearing, Lungs, BloodPressure, Artery,  Carotids, Heart,  Abdomen, Spine, Hips, Knees, FullBody, Anthropometry, Heel, Hand, PhysicalActivity, BloodCount, BloodBiochemistry, Urine
organs_groups=( "Biomarkers" "TimeSeries" "Images" "Videos" )
organs_groups=( "Biomarkers" )
for organs_group in "${organs_groups[@]}"; do
	if [ $organs_group == "Biomarkers" ]; then
		organs=( "Brain" "Eyes" "Hearing" "Lungs" "Vascular" "Heart" "Abdomen" "Musculoskeletal" "PhysicalActivity" "Biochemistry" "ImmuneSystem" )
		organs=( "Eyes" )
		architectures=( "ElasticNet" "LightGBM" "NeuralNetwork" )
		architectures=( "ElasticNet" )
		n_fc_layers="0"
		n_fc_nodes="0"
		optimizer="0"
		learning_rate="0"
		weight_decay="0"
		dropout_rate="0"
		data_augmentation_factor="0"
	elif [ $organs_group == "TimeSeries" ]; then
		organs=( "Vascular" "Heart" "PhysicalActivity" )
		architectures=( TODO )
		n_fc_layers=TODO
		n_fc_nodes=TODO
		optimizer=TODO
		learning_rate=TODO
		weight_decay=TODO
		dropout_rate=TODO
		data_augmentation_factor=TODO
	elif [ $organs_group == "Images" ]; then
		organs=( "Brain" "Eyes" "Vascular" "Heart" "Abdomen" "Spine" "Hips" "Knees" "FullBody" ) # "PhysicalActivity" )
		#architectures=( "VGG16" "VGG19" "MobileNet" "MobileNetV2" "DenseNet121" "DenseNet169" "DenseNet201" "NASNetMobile" "Xception" "InceptionV3" "InceptionResNetV2" )
		architectures=( "InceptionV3" )
		n_fc_layers="1"
		n_fc_nodes="1024"
		optimizer="Adam"
		learning_rate="0.001"
		weight_decay="0.1"
		dropout_rate="0.5"
		data_augmentation_factor="1.0"
	elif [ $organs_group == "Videos" ]; then
		organs=( "Heart" )
		architectures=( "3DCNN" )
		n_fc_layers=TODO
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
			if [ $organs_group == "Biomarkers" ]; then
				if [ $organ == "Brain" ]; then
					views=( "All" "Cognitive" "MRI" )
				elif [ $organ == "Eyes" ]; then
					views=( "All" "Autorefraction" "Acuity" "IntraocularPressure" )
				elif [ $organ == "Hearing" ]; then
					views=( "HearingTest" )
				elif [ $organ == "Lungs" ]; then
					views=( "Spirometry" )
				elif [ $organ == "Vascular" ]; then
					views=( "All" "BloodPressure" "PulseWaveAnalysis" "Carotids" )
				elif [ $organ == "Heart" ]; then
					views=( "All" "ECG" "MRI" )
				elif [ $organ == "Musculoskeletal" ]; then
					views=( "Scalars" )
				elif [ $organ == "Biochemistry" ]; then
					views=( "All" "Urine" "Blood" )
				elif [ $organ == "ImmuneSystem" ]; then
					views=( "BloodCount" )
				else
					echo "Organ $organ does not match any Biomarkers organs."
				fi
			elif [ $organs_group == "TimeSeries" ]; then
				if [ $organ == "Vascular" ]; then
					views=( "PulseWaveAnalysis" )
				elif [ $organ == "Heart" ]; then
					views=( "ECG" )
				elif [ $organ == "PhysicalActivity" ]; then
					views=( "FullWeek" "Walking" "Biking" "Sleeping" )
				else
					echo "Organ $organ does not match any TimeSeries organs."
				fi
			elif [ $organs_group == "Images" ]; then
				if [ $organ == "Brain" ]; then
					views=( "MRI" )
				elif [ $organ == "Eyes" ]; then
					views=( "Fundus" "OCT" )
				elif [ $organ == "Vascular" ]; then
					views=( "Carotids" )
				elif [ $organ == "Heart" ]; then
					views=( "MRI" )
				elif [ $organ == "Abdomen" ]; then
					views=( "Liver" "Pancreas" )
				elif [ $organ == "Musculoskeletal" ]; then
					views=( "Spine" "Hips" "Knees" "FullBody" )
				elif [ $organ == "PhysicalActivity" ]; then
					views=( "FullWeek" )
				else
					echo "Organ $organ does not match any Images organs."
				fi
			elif [ $organs_group == "Videos" ]; then
				if [ $organ == "Heart" ]; then
					views=( "MRI" )
				else
					echo "Organ $organ does not match any Videos organs."
				fi
			else
				echo "organs_group ${organs_group} is not among Biomarkers, TimeSeries, Images, or Videos"
			fi
			for view in "${views[@]}"; do
				if [ $organs_group == "Biomarkers" ]; then
					if [ $organ == "Brain" ]; then
						if [ $view == "All" ]; then
							transformations=( "Scalars" )
						elif [ $view == "Cognitive" ]; then
							transformations=( "AllScalars" "ReactionTime" "MatrixPatternCompletion" "TowerRearranging" "SymbolDigitSubstitution" "PairedAssociativeLearning" "ProspectiveMemory" "NumericMemory" "FluidlIntelligence" "TrailMaking" "PairsMatching" )
						elif [ $view == "MRI" ]; then
							transformations=( "AllScalars" "dMRIWeightedMeans" "SubcorticalVolumes" "GreyMatterVolumes" )
						fi
					elif [ $organ == "Heart" ]; then
						if [ $view == "All" || $view == "ECG" ]; then
							transformations=( "Scalars" )
						elif [ $view == "MRI" ]; then
							transformations=( "AllScalars" "Size" "PulseWaveAnalysis" )
						fi
					elif [ $organ == "Musculoskeletal" ]; then
						transformations=( "AllScalars" "Anthropometry" "Impedance" "HeelBoneDensitometry" "HandGripStrength" )
					elif [ $organ == "Biochemistry" ] || [ $organ == "ImmuneSystem" ]; then
						transformations=( "Biomarkers" )
					elif [ $organ == "Eyes" ] || [ $organ == "HearingTest" ] || [ $organ == "Lungs" ] || [ $organ == "Vascular" ] || [ $organ == "PhysicalActivity" ]; then
						transformations=( "Scalars" )
					fi
				elif [ $organs_group == "TimeSeries" ]; then
					if [ $organ == "Vascular" || $organ == "ECG" ]; then
						transformations=( "TimeSeries" )
					elif [ $organ == "PhysicalActivity" ]; then
						transformations=( "FullWeek" "Walking" "Biking" "Sleeping" )
					fi
				elif [ $organs_group == "Images" ]; then
					if [ $organ == "Brain" ]; then
						transformations=( "SagittalRaw" "SagittalReference" "CoronalRaw" "CoronalReference" "TransverseRaw" "TransverseReference" )
					elif [ $organ == "Vascular" ]; then
						transformations=( "Mixed" "LongAxis" "CIMT120" "CIMT150" "ShortAxis" )
					elif [ $organ == "Heart" ]; then
						transformations=( "2chambersRaw" "2chambersContrast" "3chambersRaw" "3chambersContrast" "4chambersRaw" "4chambersContrast" )
					elif [ $organ == "Abdomen" ]; then
						transformations=( "Raw" "Contrast" )
					elif [ $organ == "Musculoskeletal" ]; then
						if [ $view == "Spine" ]; then
							transformations=( "Sagittal" "Coronal" )
						elif [ $view == "Hips" ] || [ $view == "Knees" ]; then
							transformations=( "MRI" )
						elif [ $view == "FullBody" ]; then
							transformations=( "Mixed" "Figure" "Skeleton" "Flesh" )
						fi
					elif [ $organ == "PhysicalActivity" ]; then
						transformations=( "ReccurencePlots" )
					elif [ $organ == "Eyes" ] || [ $organ == "Spine" ] || [ $organ == "Hips" ] || [ $organ == "Knees" ] || [ $organ == "FullBody" ]; then
						transformations=( "Raw" )
					else
						echo "Organ $organ does not match any Images organs."
					fi
				elif [ $organs_group == "Videos" ]; then
					if [ $organ == "Heart" ]; then
						views=( "3chambersRawVideo" "4chambersRawVideo" "34chambersRawVideo" )
					fi
				else
					echo "organs_group ${organs_group} is not among Biomarkers, TimeSeries, Images, or Videos"
				fi
				for transformation in "${transformations[@]}"; do
					for architecture in "${architectures[@]}"; do
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
									echo "Submitting job for ${version}"
									#ID=$(sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cpu_cores -t $time MI04A05B_Performances_generate.sh $target $organ $view $transformation $architecture $n_fc_layers $n_fc_nodes $optimizer $learning_rate $weight_decay $dropout_rate $data_augmentation_factor $fold $pred_type)
									#IDs+=($ID)
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

# Produce the list of job dependencies fr the next step
printf -v IDs_list '%s:' "${IDs[@]}"
dependencies="${IDs_list%:}"
echo $dependencies

