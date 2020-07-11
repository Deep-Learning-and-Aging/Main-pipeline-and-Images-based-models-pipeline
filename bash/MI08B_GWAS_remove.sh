#!/bin/bash

#SBATCH --job-name=MI08B.job
#SBATCH --output=../eo/MI08B.out
#SBATCH --error=../eo/MI08B.err
#SBATCH --parsable
#SBATCH --open-mode=truncate
#SBATCH -n 1
#SBATCH -t 1
#SBATCH -p priority
#SBATCH --mem=1G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alanlegoallec@g.edu

#Define arguments
args=( 
	--lmm
	--bed=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_cal_chr{1:22}_v2.bed
	--bim=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_snp_chr{1:22}_v2.bim
	--fam=/n/groups/patel/Alan/Aging/Medical_Images/data/GWAS.fam
	--phenoFile=/n/groups/patel/Alan/Aging/Medical_Images/data/GWAS_exhaustive_placeholder.tab
	--phenoCol=phenotype
	--LDscoresFile=/n/groups/patel/bin/BOLT-LMM_v2.3.2/tables/LDSCORE.1000G_EUR.tab.gz
	--geneticMapFile=/n/groups/patel/bin/BOLT-LMM_v2.3.2/tables/genetic_map_hg19.txt.gz
	--numThreads=10
	--statsFile=GWAS_nonimputed.stats.gz
	--verboseStats
	--bgenFile=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_imp_chr{1:22}_v3.bgen
	--bgenMinMAF=1e-3
	--bgenMinINFO=0.3
	--sampleFile=/n/groups/patel/uk_biobank/project_52887_genetics/ukb52887_imp_chr1_v3_s487296.sample
	--statsFileBgenSnps=/n/groups/patel/Alan/Aging/Medical_Images/data/GWAS_nonimputed.bgen.stats.gz
)

# Run the job
cd /n/groups/patel/bin/BOLT-LMM_v2.3.2/
output=$(./bolt "${args[@]}" 2>&1)

# Move the non imputed ids file to data
toremove_ids=$(echo $output | grep -o "bolt.in_plink_but_not_imputed.FID_IID.*.txt")
mv $toremove_ids /n/groups/patel/Alan/Aging/Medical_Images/data/bolt.in_plink_but_not_imputed.FID_IID.txt
echo "Done"

