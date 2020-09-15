import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import sys
from MI_Classes import AttentionMaps

# Options
# Use a small subset of the data VS. run the actual full data pipeline to get accurate results
# /!\ if True, path to save weights will be automatically modified to avoid rewriting them
debug_mode = False

# Default parameters
if len(sys.argv) != 5:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target
    sys.argv.append('Eyes')  # organ
    sys.argv.append('Fundus')  # view
    sys.argv.append('Raw')  # transformation

# Generate results
Attention_Maps = AttentionMaps(target=sys.argv[1], organ=sys.argv[2], view=sys.argv[3], transformation=sys.argv[4],
                                    debug_mode=debug_mode)
Attention_Maps.preprocessing()
Attention_Maps.generate_filters()

# Exit
print('Done.')
sys.exit(0)
