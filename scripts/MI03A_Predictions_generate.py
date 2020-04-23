from MI_Libraries import *
from MI_Classes import PredictionsGenerate

# options
# debug mode
debug_mode = True
# save predictions
save_predictions = True

# Default parameters
if len(sys.argv) != 9:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target
    sys.argv.append('EyeFundus_210156_right')  # organ_id_view, Heart_20208_3chambers
    sys.argv.append('raw')  # transformation
    sys.argv.append('InceptionV3')  # architecture
    sys.argv.append('Adam')  # optimizer
    sys.argv.append('0.000001')  # learning_rate
    sys.argv.append('0.0')  # weight decay
    sys.argv.append('0.0')  # dropout

# Compute results
Predictions_Generate = PredictionsGenerate(target=sys.argv[1], organ_id_view=sys.argv[2], transformation=sys.argv[3],
                                           architecture=sys.argv[4], optimizer=sys.argv[5], learning_rate=sys.argv[6],
                                           weight_decay=sys.argv[7], dropout_rate=sys.argv[8], debug_mode=debug_mode)
Predictions_Generate.generate_predictions()
if save_predictions:
    Predictions_Generate.save_predictions()

# Exit
print('Done.')
Predictions_Generate.clean_exit()


self = Predictions_Generate
fold = 'train'
pred_batch = self.model.predict_generator(self.GENERATORS_BATCH[fold], steps=1, verbose=0)
pred_leftovers = self.model.predict_generator(self.GENERATORS_LEFTOVERS[fold], steps=1, verbose=0)

steps = math.ceil(len(self.list_ids)/self.batch_size)




