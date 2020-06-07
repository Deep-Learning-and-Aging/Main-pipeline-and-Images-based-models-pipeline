import sys
from MI_Classes import EnsemblesPredictions

# Default parameters
if len(sys.argv) != 3:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target
    sys.argv.append('eids')  # pred_type

# Compute results
Ensemble_Predictions = EnsemblesPredictions(target=sys.argv[1], pred_type=sys.argv[2])
Ensemble_Predictions.load_data()
Ensemble_Predictions.generate_ensemble_predictions()
Ensemble_Predictions.save_predictions()

# Exit
print('Done.')
sys.exit(0)
