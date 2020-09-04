#!/bin/bash
regenerate_data=false
#targets=( "Age" "Sex" )
targets=( "Age" )
organs=( "*" "Brain" "BrainCognitive" "BrainMRI" "Eyes" "EyesFundus" "EyesOCT" "Hearing" "Lungs" "Arterial" "ArterialPulseWaveAnalysis" "ArterialCarotids" "Heart" "HeartECG" "HeartMRI" "Abdomen" "AbdomenLiver" "AbdomenPancreas" "Musculoskeletal" "MusculoskeletalSpine" "MusculoskeletalHips" "MusculoskeletalKnees" "MusculoskeletalFullBody" "MusculoskeletalScalars" "PhysicalActivity" "Biochemistry" "BiochemistryUrine" "BiochemistryBlood" "ImmuneSystem" )
#organs=( "Hearing" "Lungs" "ArterialPulseWaveAnalysis" "Biochemistry" "BiochemistryUrine" "BiochemistryBlood" "ImmuneSystem" )
#organs=( "Arterial" )
chromosomesS=( "autosome" "X" )

#loop through the jobs to submit
declare -a IDs=()
for target in "${targets[@]}"; do
	for organ in "${organs[@]}"; do
		sample_size=$(wc -l < ../data/GWAS_data_"${target}"_"${organ}".tab)
		mem_per_cpu=$(printf "%.0f" $(expr 0.27*$sample_size*0.1+1000 | bc))
		base_time=$(printf "%.0f" $(expr 0.003*$sample_size+20 | bc))
		for chromosomes in "${chromosomesS[@]}"; do
			if [ $chromosomes == "X" ]; then
				analyses=( "lmm" "reml" )
			elif [ $chromosomes == "autosome" ]; then
				analyses=( "lmm" )
			fi
			for analysis in "${analyses[@]}"; do
				if [ $analysis == reml ]; then
					time=$(( 2*$base_time ))
				else
					time=$base_time
				fi
				if [ $time -lt 720 ]; then
					partition=short
				elif [ $time -lt 7200 ]; then
					partition=medium
				else
					partition=long
				fi
				to_run=true
				path_out=../eo/MI09C_"${analysis}"_"${target}"_"${organ}"_"${chromosomes}".out
				echo $path_out
				if ! $regenerate_data && ( ( [ $analysis == "lmm" ] && test -f "../data/GWAS_"${target}"_"${organ}"_"${chromosomes}".stats" ) || ( [ $analysis == "reml" ] && test -f "${path_out}" && grep -q "h2g" "${path_out}" ) ) ; then
					to_run=false
				fi
				if $to_run; then
					version=MI09C_${analysis}_${target}_${organ}_${chromosomes}
					job_name="$version.job"
					out_file="../eo/$version.out"
					err_file="../eo/$version.err"
					echo $version
					echo $time
					#ID=$(sbatch --dependency=$1 --parsable -p $partition -t $time -c 10 --mem-per-cpu $mem_per_cpu --error=$err_file --output=$out_file --job-name=$job_name MI09CD_GWAS_bolt.sh $target $organ $chromosomes $analysis)
					#IDs+=($ID)
				fi
			done
		done
	done
done

# Produce the list of job dependencies for the next step
printf -v IDs_list '%s:' "${IDs[@]}"
dependencies="${IDs_list%:}"
echo $dependencies

