import sys
from MI_Classes import PreprocessingMain

# Compute results
Preprocessing_Main = PreprocessingMain()
Preprocessing_Main.generate_data()
Preprocessing_Main.save_data()

# Exit
print('Done.')
sys.exit(0)


data_raw = pd.read_csv('/n/groups/patel/uk_biobank/project_52887_41230/ukb41230.csv', nrows=1)

