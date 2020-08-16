#!/bin/bash
#targets=( "Age" "Sex" )
targets=( "Age" )
organs=( "*" "Brain" "BrainCognitive" "BrainMRI" "Eyes" "EyesFundus" "EyesOCT" "Hearing" "Lungs" "Arterial" "ArterialPulseWaveAnalysis" "ArterialCarotids" "Heart" "HeartECG" "HeartMRI" "Abdomen" "AbdomenLiver" "AbdomenPancreas" "Musculoskeletal" "MusculoskeletalSpine" "MusculoskeletalHips" "MusculoskeletalKnees" "MusculoskeletalFullBody" "MusculoskeletalScalars" "PhysicalActivity" "Biochemistry" "BiochemistryUrine" "BiochemistryBlood" "ImmuneSystem" )
organs=( "BrainCognitive" )
chromosomesS=( "autosome" "X" )
chromosomesS=( "X" )
n_cpus=1 #TODO 10
time=400

#loop through the jobs to submit
declare -a IDs=()
for target in "${targets[@]}"; do
	for organ in "${organs[@]}"; do
		if [ $organ == "*" ]; then
			mem_per_cpu=5G
		elif [ $organ == "Brain" ]; then
			mem_per_cpu=5G
		elif [ $organ == "BrainCognitive" ]; then
			mem_per_cpu=5G
		elif [ $organ == "BrainMRI" ]; then
			mem_per_cpu=5G
		elif [ $organ == "Eyes" ]; then
			mem_per_cpu=5G
		elif [ $organ == "EyesFundus" ]; then
			mem_per_cpu=5G
		elif [ $organ == "EyesOCT" ]; then
			mem_per_cpu=5G
		elif [ $organ == "Hearing" ]; then
			mem_per_cpu=5G
		elif [ $organ == "Lungs" ]; then
			mem_per_cpu=5G
		elif [ $organ == "Arterial" ]; then
			mem_per_cpu=5G
		elif [ $organ == "ArterialPulseWaveAnalysis" ]; then
			mem_per_cpu=5G
		elif [ $organ == "ArterialCarotids" ]; then
			mem_per_cpu=5G
		elif [ $organ == "Heart" ]; then
			mem_per_cpu=5G
		elif [ $organ == "HeartECG" ]; then
			mem_per_cpu=5G
		elif [ $organ == "HeartMRI" ]; then
			mem_per_cpu=5G
		elif [ $organ == "Abdomen" ]; then
			mem_per_cpu=5G
		elif [ $organ == "AbdomenLiver" ]; then
			mem_per_cpu=5G
		elif [ $organ == "AbdomenPancreas" ]; then
			mem_per_cpu=5G
		elif [ $organ == "Musculoskeletal" ]; then
			mem_per_cpu=5G
		elif [ $organ == "MusculoskeletalSpine" ]; then
			mem_per_cpu=5G
		elif [ $organ == "MusculoskeletalHips" ]; then
			mem_per_cpu=5G
		elif [ $organ == "MusculoskeletalKnees" ]; then
			mem_per_cpu=5G
		elif [ $organ == "MusculoskeletalFullBody" ]; then
			mem_per_cpu=5G
		elif [ $organ == "MusculoskeletalScalars" ]; then
			mem_per_cpu=5G
		elif [ $organ == "PhysicalActivity" ]; then
			mem_per_cpu=5G
		elif [ $organ == "Biochemistry" ]; then
			mem_per_cpu=5G
		elif [ $organ == "BiochemistryBlood" ]; then
			mem_per_cpu=5G
		elif [ $organ == "BiochemistryUrine" ]; then
			mem_per_cpu=5G #TODO 32
		elif [ $organ == "ImmuneSystem" ]; then
			mem_per_cpu=64G
		fi
		for chromosomes in "${chromosomesS[@]}"; do
			if [ $chromosomes == "X" ]; then
				analyses=( "lmm" "reml" )
			elif [ $chromosomes == "autosome" ]; then
				analyses=( "lmm" )
			fi
			for analysis in "${analyses[@]}"; do
				version=MI08C_${analysis}_${target}_${organ}_${chromosomes}_${analysis}
				job_name="$version.job"
				out_file="../eo/$version.out"
				err_file="../eo/$version.err"
				ID=$(sbatch --dependency=$1 --parsable -t $time -n 1 -c 10 --mem-per-cpu $mem_per_cpu --error=$err_file --output=$out_file --job-name=$job_name MI08CD_GWAS_bolt.sh $target $organ $chromosomes $analysis)
				IDs+=($ID)
			done
		done
	done
done

# Produce the list of job dependencies for the next step
printf -v IDs_list '%s:' "${IDs[@]}"
dependencies="${IDs_list%:}"
echo $dependencies

