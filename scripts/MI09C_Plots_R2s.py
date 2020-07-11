import sys
#from MI_Classes import PlotsR2s
from MI_Classes import Hyperparameters
import plotly.graph_objects as go

# Default parameters
if len(sys.argv) != 2:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target


class PlotsR2s(Hyperparameters):
    
    def __init__(self, target=None):
        self.target = target
        self.PERFORMANCES = {}
        for pred_type in ['instances', 'eids']:
            for fold in self.folds:
                self.PERFORMANCES[pred_type][fold] = pd.read_csv('../data/PERFORMANCES_withEnsembles_ranked_' +
                                                                 pred_type + '_' + self.target + '_' + fold + '.csv')
    
    def _generate_plot(self, Performances, metric, title):
        # Preprocess the data
        organs = Performances['organ'].unique()
        Ys = {}
        transformations = df['transformation'].unique()
        for transformation in transformations:
            Ys[transformation] = []
        level_1 = []
        level_2 = []
        level_3 = []
        for organ in organs:
            df_organ = df[(df['organ'] == organ) & (~df['architecture'].isin([',', '?']))]
            views = df_organ['view'].unique()
            views.sort()
            for view in views:
                df_view = df_organ[df_organ['view'] == view]
                architectures = df_view['architecture'].unique()
                architectures.sort()
                for architecture in architectures:
                    df_architecture = df_view[df_view['architecture'] == architecture]
                    level_1.append(organ)
                    level_2.append(view)
                    level_3.append(architecture)
                    for transformation in transformations:
                        try:
                            score = df_architecture['R-Squared_all'][
                                df_architecture['transformation'] == transformation].values[0]
                            Ys[transformation].append(score)
                        except:
                            Ys[transformation].append(0)  # TODO try NA instead np.nan
        # Plot figure
        x = [level_1, level_2, level_3]
        fig = go.Figure()
        for transformation in transformations:
            fig.add_bar(x=x, y=Ys[transformation], name=transformation)
        fig.show()
    
    def generate_all_plots(self):
        for metric in self.metrics_names[self.target]:
            for fold in self.folds:
                Performances = self.PERFORMANCES[fold][['organ', 'view', 'architecture', 'transformation',
                                                        metric + '_all']]
                self._generate_plot(Performances, metric, title)

# Generate results
Plots_R2s = PlotsR2s(target=sys.argv[1])
Plots_R2s.generate_all_plots()

# Exit
print('Done.')
sys.exit(0)
