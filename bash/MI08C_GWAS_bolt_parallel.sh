#!/bin/bash
#targets=( "Age" "Sex" )
targets=( "Age" )
organs=( "Liver" )
chromosomesS=( "autosome" "X" )
n_cpus=1 #TODO 10
time=600

#loop through the jobs to submit
declare -a IDs=()
for target in "${targets[@]}"; do
	for organ in "${organs[@]}"; do
		if [ $organ == "Liver" ]; then
			mem_per_cpu=5G #TODO 32
		fi
		for chromosomes in "${chromosomesS[@]}"; do
			version=MI08C_${target}_${organ}_${chromosomes}
			job_name="$version.job"
			out_file="../eo/$version.out"
			err_file="../eo/$version.err"
			ID=$(sbatch --dependency=$1 --parsable -t $time -n 1 -c 10 --mem-per-cpu $mem_per_cpu --error=$err_file --output=$out_file --job-name=$job_name MI08C_GWAS_bolt.sh $target $organ $chromosomes)
			IDs+=($ID)
		done
	done
done

# Produce the list of job dependencies for the next step
printf -v IDs_list '%s:' "${IDs[@]}"
dependencies="${IDs_list%:}"
echo $dependencies

