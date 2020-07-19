#!/bin/bash
#SBATCH -n 1
#SBATCH -p priority
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alanlegoallec@g.edu

#Define parameters
target=$1
organ=$2
chromosomes=$3
# Define number of chromosomes
if [ $# -eq 4 ] && [ $4 == "debug_mode" ]; then
	debug="_debug"
	NC=2
else
	debug=""
	NC=22
fi

# Define parameters
args=( 
	--lmm
	--bed=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_cal_chr{1:$NC}_v2.bed
	--bim=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_snp_chr{1:$NC}_v2.bim
	--fam=/n/groups/patel/Alan/Aging/Medical_Images/data/GWAS.fam
	--remove=/n/groups/patel/Alan/Aging/Medical_Images/data/bolt.in_plink_but_not_imputed.FID_IID.${chromosomes}.txt 
	--remove=/n/groups/patel/Alan/Aging/Medical_Images/data4/GWAS_remove_${target}_${organ}.tab
	--phenoFile=/n/groups/patel/Alan/Aging/Medical_Images/data4/GWAS_data_${target}_${organ}.tab
	--phenoCol=${organ}
	--covarFile=/n/groups/patel/Alan/Aging/Medical_Images/data4/GWAS_data_${target}_${organ}.tab
	--covarCol=Assessment_center
	--covarCol=Sex
	--covarCol=Ethnicity
	--covarMaxLevels=30
	--qCovarCol=Age
	--qCovarCol=PC{1:20}
	--LDscoresFile=/n/groups/patel/bin/BOLT-LMM_v2.3.2/tables/LDSCORE.1000G_EUR.tab.gz
	--geneticMapFile=/n/groups/patel/bin/BOLT-LMM_v2.3.2/tables/genetic_map_hg19_withX.txt.gz
	--numThreads=10
	--statsFile=/n/groups/patel/Alan/Aging/Medical_Images/data4/GWAS_${target}_${organ}_${chromosomes}${debug}.stats.gz
	--verboseStats
)

# Add chromosome X. Still need all other chromosomes included for the optimal LMM
if [ $chromosomes == "X" ]; then
	args+=(
		--bed=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_cal_chrX_v2.bed
		--bim=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_snp_chrX_v2.bim
	)
fi

# Add arguments for imputed data GWAS
if [ $# -eq 3 ] || [ $debug == "" ]; then
	args+=(
		--bgenSampleFileList=/n/groups/patel/uk_biobank/project_52887_genetics/bgenSampleFileList_${chromosomes}.txt
		--bgenMinMAF=1e-3
		--bgenMinINFO=0.3
		--noBgenIDcheck
		--statsFileBgenSnps=/n/groups/patel/Alan/Aging/Medical_Images/data4/GWAS_${target}_${organ}_${chromosomes}${debug}.bgen.stats.gz 
	)
fi

echo "${args[@]}"

# Run the job
cd /n/groups/patel/bin/BOLT-LMM_v2.3.2/
./bolt "${args[@]}" && echo "Done"

