#!/bin/bash
regenerate_data=false
targets=( "Age" )
time=400
organs_to_run=( "Hearing" "Lungs" "ArterialPulseWaveAnalysis" "Biochemistry" "BiochemistryUrine" "BiochemistryBlood" "ImmuneSystem" )

#loop through the jobs to submit
declare -a IDs=()
for target in "${targets[@]}"; do
	while IFS= read -r line; do
		organ1=$(echo $line | cut -d ',' -f1)
		organ2=$(echo $line | cut -d ',' -f2)
		sample_size=$(wc -l < ../data/GWAS_data_"${target}"_"${organ1}"_"${organ2}".tab)
		mem_per_cpu=$(printf "%.0f" $(expr 0.26*$sample_size*0.1 | bc))
		time=$(printf "%.0f" $(expr 0.007*$sample_size | bc))
		if [ $time -lt 720 ]; then
			partition=short
		elif [ $time -lt 7200 ]; then
			partition=medium
		else
			partition=long
		fi
		to_run=false
		path_out=../eo/MI09D_"${target}"_"${organ1}"_"${organ2}".out
		if $regenerate_data || ! test -f ${path_out} || ! grep -q "gen corr (1,2)" ${path_out}; then
			to_run=true
		fi
		if [[ ! " ${organs_to_run[@]} " =~ " ${organ1} " ]] || [[ ! " ${organs_to_run[@]} " =~ " ${organ2} " ]]; then
			to_run=false
		fi
		if $to_run; then
			version=MI09D_${target}_${organ1}_${organ2}
			job_name="$version.job"
			out_file="../eo/$version.out"
			err_file="../eo/$version.err"
			echo $version
			#ID=$(sbatch --dependency=$1 --parsable -p $partition -t $time -c 10 --mem-per-cpu $mem_per_cpu --error=$err_file --output=$out_file --job-name=$job_name MI09CD_GWAS_bolt.sh $target $organ1 $organ2 reml_correlation)
			#IDs+=($ID)
		fi
	done < ../data/GWAS_genetic_correlations_pairs_${target}.csv
outer_folds=( "0" )
done

# Produce the list of job dependencies for the next step
printf -v IDs_list '%s:' "${IDs[@]}"
dependencies="${IDs_list%:}"
echo $dependencies

