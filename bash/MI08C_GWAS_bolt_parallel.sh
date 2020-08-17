#!/bin/bash
regenerate_data=false
#targets=( "Age" "Sex" )
targets=( "Age" )
organs=( "*" "Brain" "BrainCognitive" "BrainMRI" "Eyes" "EyesFundus" "EyesOCT" "Hearing" "Lungs" "Arterial" "ArterialPulseWaveAnalysis" "ArterialCarotids" "Heart" "HeartECG" "HeartMRI" "Abdomen" "AbdomenLiver" "AbdomenPancreas" "Musculoskeletal" "MusculoskeletalSpine" "MusculoskeletalHips" "MusculoskeletalKnees" "MusculoskeletalFullBody" "MusculoskeletalScalars" "PhysicalActivity" "Biochemistry" "BiochemistryUrine" "BiochemistryBlood" "ImmuneSystem" )
#organs=( "Hearing" "Lungs" "Biochemistry" "BiochemistryUrine" "BiochemistryBlood" "ImmuneSystem" )
organs=( "BrainCognitive" "Hearing" "Lungs" "Biochemistry" "BiochemistryUrine" "BiochemistryBlood" )
chromosomesS=( "autosome" "X" )
#chromosomesS=( "X" )
n_cpus=1 #TODO 10

#loop through the jobs to submit
declare -a IDs=()
for target in "${targets[@]}"; do
	for organ in "${organs[@]}"; do
		sample_size=$(wc -l < ../data/GWAS_data_${target}_${organ}.tab)
		mem_per_cpu=$(printf "%.0f" $(expr 0.26*$sample_size*0.1 | bc))
		base_time=$(printf "%.0f" $(expr 0.003*$sample_size | bc))
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
				to_run=true
				if ! $regenerate_data && ( ( [ $analysis == "lmm" ] && test -f "../data/GWAS_${target}_${organ}_${chromosomes}.stats" ) || ( [ $analysis == "reml" ] &&  grep -q "h2g" ../eo/MI08C_${analysis}_${target}_${organ}_${chromosomes}_${analysis}.out ) ); then
					to_run=false
				fi
				if $to_run; then
					version=MI08C_${analysis}_${target}_${organ}_${chromosomes}}
					job_name="$version.job"
					out_file="../eo/$version.out"
					err_file="../eo/$version.err"
					echo $version
					echo $mem_per_cpu
					echo $time
					#ID=$(sbatch --dependency=$1 --parsable -t $time -n 1 -c 10 --mem-per-cpu $mem_per_cpu --error=$err_file --output=$out_file --job-name=$job_name MI08CD_GWAS_bolt.sh $target $organ $chromosomes $analysis)
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

