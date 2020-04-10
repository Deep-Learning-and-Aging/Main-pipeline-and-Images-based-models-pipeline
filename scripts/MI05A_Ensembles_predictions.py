from MI_Libraries import *
from MI_Classes import EnsemblesPredictions

# Default parameters
if len(sys.argv) != 2:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target

# Compute results
Ensemble_Predictions = EnsemblesPredictions(target=sys.argv[1])
Ensemble_Predictions.load_data()
Ensemble_Predictions.generate_ensemble_predictions()
Ensemble_Predictions.save_predictions()

# Exit
print('Done.')
sys.exit(0)
