from MI_Libraries import *
from MI_Classes import PlotsAttentionMaps

# Default parameters
if len(sys.argv) != 5:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target
    sys.argv.append('Heart_20208_4chambers')  # organ_id_view
    sys.argv.append('contrast')  # transformation
    sys.argv.append('test')  # fold

# Generate results
Plots_AttentionMaps = PlotsAttentionMaps(target=sys.argv[1], organ_id_view=sys.argv[2], transformation=sys.argv[3],
                                         fold=sys.argv[4])
Plots_AttentionMaps.preprocessing()
Plots_AttentionMaps.generate_plots()

# Exit
print('Done.')
sys.exit(0)
