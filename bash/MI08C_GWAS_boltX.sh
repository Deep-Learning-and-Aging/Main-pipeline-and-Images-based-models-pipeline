#!/bin/bash
#SBATCH -n 1
#SBATCH -t 5
#SBATCH -p priority
#SBATCH --mem=1G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alanlegoallec@g.edu

#Define parameters
target=$1
organ=$2
# Define number of chromosomes
if [ $# -eq 3 ] && [ $3 == "debug_mode" ]; then
	debug="_debug"
	NC=2
else
	debug=""
	NC=22
fi

# Define parameters
args=( 
	--lmm
	--bed=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_cal_chr1_v2.bed
	--bed=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_cal_chr2_v2.bed
	--bed=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_cal_chr3_v2.bed
	--bed=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_cal_chr4_v2.bed
	--bed=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_cal_chr5_v2.bed
	--bed=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_cal_chr6_v2.bed
	--bed=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_cal_chr7_v2.bed
	--bed=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_cal_chr8_v2.bed
	--bed=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_cal_chr9_v2.bed
	--bed=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_cal_chr10_v2.bed
	--bed=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_cal_chr11_v2.bed
	--bed=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_cal_chr12_v2.bed
	--bed=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_cal_chr13_v2.bed
	--bed=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_cal_chr14_v2.bed
	--bed=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_cal_chr15_v2.bed
	--bed=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_cal_chr16_v2.bed
	--bed=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_cal_chr17_v2.bed
	--bed=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_cal_chr18_v2.bed
	--bed=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_cal_chr19_v2.bed
	--bed=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_cal_chr20_v2.bed
	--bed=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_cal_chr21_v2.bed
	--bed=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_cal_chr22_v2.bed
	--bed=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_cal_chrX_v2.bed
	--bim=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_snp_chr1_v2.bim
	--bim=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_snp_chr2_v2.bim
	--bim=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_snp_chr3_v2.bim
	--bim=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_snp_chr4_v2.bim
	--bim=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_snp_chr5_v2.bim
	--bim=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_snp_chr6_v2.bim
	--bim=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_snp_chr7_v2.bim
	--bim=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_snp_chr8_v2.bim
	--bim=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_snp_chr9_v2.bim
	--bim=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_snp_chr10_v2.bim
	--bim=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_snp_chr11_v2.bim
	--bim=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_snp_chr12_v2.bim
	--bim=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_snp_chr13_v2.bim
	--bim=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_snp_chr14_v2.bim
	--bim=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_snp_chr15_v2.bim
	--bim=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_snp_chr16_v2.bim
	--bim=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_snp_chr17_v2.bim
	--bim=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_snp_chr18_v2.bim
	--bim=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_snp_chr19_v2.bim
	--bim=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_snp_chr20_v2.bim
	--bim=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_snp_chr21_v2.bim
	--bim=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_snp_chr22_v2.bim
	--bim=/n/groups/patel/uk_biobank/project_52887_genetics/ukb_snp_chrX_v2.bim
	--fam=/n/groups/patel/Alan/Aging/Medical_Images/data4/GWAS.fam
	--remove=/n/groups/patel/Alan/Aging/Medical_Images/data/bolt.in_plink_but_not_imputed.FID_IID.txt 
	--remove=/n/groups/patel/Alan/Aging/Medical_Images/data4/GWAS_remove_${target}_${organ}.tab
	--phenoFile=/n/groups/patel/Alan/Aging/Medical_Images/data4/GWAS_data_${target}_${organ}${debug}.tab
	--phenoCol=${organ}
	--covarFile=/n/groups/patel/Alan/Aging/Medical_Images/data4/GWAS_data_${target}_${organ}${debug}.tab
	--covarCol=Assessment_center
	--covarCol=Sex
	--covarCol=Ethnicity
	--covarMaxLevels=30
	--qCovarCol=Age
	--qCovarCol=PC{1:20}
	--LDscoresFile=/n/groups/patel/bin/BOLT-LMM_v2.3.2/tables/LDSCORE.1000G_EUR.tab.gz
	--geneticMapFile=/n/groups/patel/bin/BOLT-LMM_v2.3.2/tables/genetic_map_hg19_withX.txt.gz
	--numThreads=10
	--statsFile=/n/groups/patel/Alan/Aging/Medical_Images/data4/GWAS_${target}_${organ}.stats.gz
	--verboseStats
)

# Add arguments for imputed data GWAS
if [ $# -eq 2 ] || [ $debug == "" ]; then
	args+=( --bgenSampleFileList=/n/groups/patel/uk_biobank/project_52887_genetics/bgenSampleFileList_debug.txt
			--bgenMinMAF=1e-3
			--bgenMinINFO=0.3
			--noBgenIDcheck
			--statsFileBgenSnps=/n/groups/patel/Alan/Aging/Medical_Images/data4/GWAS_${target}_${organ}.bgen.stats.gz 
		)
fi

# Run the job
cd /n/groups/patel/bin/BOLT-LMM_v2.3.2/
./bolt "${args[@]}" && echo "Done"

