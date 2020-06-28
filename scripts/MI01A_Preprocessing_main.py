import sys
from MI_Classes import PreprocessingMain

# Compute results
Preprocessing_Main = PreprocessingMain()
Preprocessing_Main.generate_data()
Preprocessing_Main.save_data()

# Exit
print('Done.')
sys.exit(0)

for eid in eids_missing_ethnicity:
    sample = self.data_raw.loc[eid, :]
    if not math.isnan(sample['Ethnicity_1']):
        self.data_raw.loc[eid, 'Ethnicity'] = self.data_raw.loc[eid, 'Ethnicity_1']
    elif not math.isnan(sample['Ethnicity_2']):
        self.data_raw.loc[eid, 'Ethnicity'] = self.data_raw.loc[eid, 'Ethnicity_2']
