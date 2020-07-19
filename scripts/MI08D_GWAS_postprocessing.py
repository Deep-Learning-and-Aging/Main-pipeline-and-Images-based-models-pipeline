import sys
from MI_Classes import GWASPostprocessing

# Default parameters
if len(sys.argv) != 2:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target


import os
import pandas as pd
from bioinfokit import visuz
os.getcwd()
os.chdir('/Users/Alan/Desktop/')

#Use a fixed color for all the plots
color=("#a7414a", "#696464", "#00743f", "#563838", "#6a8a82", "#a37c27", "#5edfff", "#282726", "#c0334d", "#c9753d")


df = pd.read_csv('GWAS_Age_Liver.stats', sep='\t')
visuz.marker.mhat(df=df, chr='CHR', pv='P_BOLT_LMM_INF', gwas_sign_line=True, gwasp=1/len(df.index),
                  markernames=({"rs116720794": "gene1", "rs10482810": "gene2"}), markeridcol='SNP', color=color,
                  gstyle=2, r=600)
os.rename('manhatten.png', 'GWAS_Liver_ManhattanPlot.png')
#TODO decide which p-value.
#Volcano plot.


#choose best colors
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 1, 10)
for i in range(23):
    plt.plot(x, i * x + i, label='$y = {i}x + {i}$'.format(i=i))

plt.legend(loc='best')
plt.show()

#todebug for Samuel's volcano plot:
df3 = df[df['P_BOLT_LMM_INF'] < 1e-4]
df3['gene'] = ['gene_' + str(i+1) for i in range(len(df3.index))]
df3
df3.to_csv('GWAS_for_Samuel_to_debug_colvano_plot.csv')




dict_colors = {'white': '#f2f3f4', 'black': '#222222', 'yellow', 'purple', 'orange', 'light_blue', 'red', 'buff', 'gray', 'green',
               'purplish_pink', 'blue', 'yellowish pink', 'violet', 'orange_yellow', 'purplish_red', 'greenish_yellow',
               'reddish_brown', 'yellow_green', 'yellowish_brown', 'reddish_orange', 'olive_green'}
               
               
               
               
               