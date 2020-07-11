#!/bin/bash
#targets=( "Age" "Sex" )
targets=( "Age" )
organs=( "Liver" )

#loop through the jobs to submit
declare -a IDs=()
for target in "${targets[@]}"; do
	for organ in "${organs[@]}"; do
		version=MI08C_${target}_${organ}
		job_name="$version.job"
		out_file="../eo/$version.out"
		err_file="../eo/$version.err"
		ID=$(sbatch --x11 --error=$err_file --output=$out_file --job-name=$job_name  MI08C_GWAS_bolt.sh $target $organ)
		IDs+=($ID)
	done
done
# Produce the list of job dependencies fr the next step
printf -v IDs_list '%s:' "${IDs[@]}"
dependencies="${IDs_list%:}"
echo $dependencies

