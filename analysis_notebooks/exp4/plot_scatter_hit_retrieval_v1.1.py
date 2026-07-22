
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
import random
#import matplotlib.colors as colors
#import matplotlib.cbook as cbook
#from matplotlib import cm
#import seaborn as sns

# def add_molecule_to_plot(ax, smiles, xy, zoom=0.1):
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         return
    
#     img = Draw.MolToImage(mol, size=(150, 150))
#     imagebox = OffsetImage(img, zoom=zoom)
#     ab = AnnotationBbox(imagebox, xy, frameon=True, pad=0.3,
#                         bboxprops=dict(boxstyle="round,pad=0.2", 
#                                       facecolor='white', 
#                                       edgecolor='black', 
#                                       linewidth=0.5))
#     ax.add_artist(ab)

def smiles_to_image(smiles, size=(200, 200)):
    """Convert SMILES to PIL Image"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=size)
    return img

def add_molecule_to_plot(ax, smiles, xy, zoom=0.15):
    """Add molecule image to plot at position xy"""
    img = smiles_to_image(smiles)
    if img is None:
        return
    
    # Convert PIL image to array
    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(imagebox, xy, frameon=True, pad=0.5, 
                        bboxprops=dict(boxstyle="round,pad=0.3", 
                                      facecolor='white', 
                                      edgecolor='gray', 
                                      linewidth=1))
    ax.add_artist(ab)


def add_molecule_with_arrow(ax, smiles, xy_point, xy_mol, zoom=0.1):
    """
    Add molecule image at xy_mol with arrow pointing to xy_point
    Arrow stops at the edges of both the molecule box and the point
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return
    
    # Create molecule image
    img = Draw.MolToImage(mol, size=(150, 150))
    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(imagebox, xy_mol, frameon=True, pad=0.3,
                        bboxprops=dict(boxstyle="round,pad=0.2", 
                                      facecolor='white', 
                                      edgecolor='black', 
                                      linewidth=1))
    ax.add_artist(ab)
    
    # Draw arrow from molecule to point on line
    # shrinkA: shrink from the text/annotation end (molecule)
    # shrinkB: shrink from the point end
    ax.annotate('', xy=xy_point, xytext=xy_mol,
                arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', 
                               connectionstyle="arc3,rad=0.3",
                               shrinkA=80, shrinkB=5))  # These values control the gap
    
    # Mark the point on the line
    ax.plot(xy_point[0], xy_point[1], 'ro', markersize=6, zorder=5)

df = pd.read_pickle('pstp_all_94044_clnsmi_murcko_iters_fps_clusters_umap_randiters_dockscores_btclusts.pkl')
smiles_list = []
# print(df)
# df.to_csv('pstp_all_94044_clnsmi_murcko_iters_fps_clusters_umap_randiters_dockscores_btclusts.csv')
max_iteration = int(df['iter_num'].max())
# iter_hits = [[0,0,0,0,0]]
# dock_hits = [[0,0,0,0,0]]
# rand_hits = [[0,0,0,0,0]]
iter_hits = []
dock_hits = []
rand_hits = []

unique_lens = []
## Added unique hits to each of them, but I think Unique hits and total hits are the same?
for i in range(50):
    df_iter = df.loc[ df['iter_num'] <= i ]
    df_unique = df_iter.drop_duplicates(subset=['rdkit_smiles_cln'])
    smiles_list.append(df_iter['SMILES'].values)
    #n_tested = len(df_iter)
    n_tested = i
    df_iter.loc[ df_iter['PstP True Active'] == 1 ]
    n_hits = len( df_iter.loc[ df_iter['PstP True Active'] == 1 ] )
    n_unique_hits = int(df_unique['PstP True Active'].astype(int).sum())
    n_murcko = len( df_iter.loc[ df_iter['PstP True Active'] == 1, 'gen_murcko' ].unique() )
    n_clust = len( df_iter.loc[ df_iter['PstP True Active'] == 1, 'BT_0.4 ID' ].unique() )
    iter_hits.append( [n_tested, n_hits, n_unique_hits, n_murcko, n_clust] )

for i in range(50):
    df_iter = df.loc[ df['rand_iter'] <= i ]
    df_unique = df_iter.drop_duplicates(subset=['rdkit_smiles_cln'])
    unique_lens.append((len(df_iter), len(df_unique)))
    #n_tested = len(df_iter)
    n_tested = i
    df_iter.loc[ df_iter['PstP True Active'] == 1 ]
    n_hits = len( df_iter.loc[ df_iter['PstP True Active'] == 1 ] )
    n_unique_hits = int(df_unique['PstP True Active'].astype(int).sum())
    n_murcko = len( df_iter.loc[ df_iter['PstP True Active'] == 1, 'gen_murcko' ].unique() )
    n_clust = len( df_iter.loc[ df_iter['PstP True Active'] == 1, 'BT_0.4 ID' ].unique() )
    rand_hits.append( [n_tested, n_hits, n_unique_hits, n_murcko, n_clust] )

df = df.sort_values( by='ECR_rank' )
for i in range(50):
    N = (i+1) * 80
    df_iter = df.loc[ df['ECR_rank'] <= N ]
    df_unique = df_iter.drop_duplicates(subset=['rdkit_smiles_cln'])
    #n_tested = len(df_iter)
    n_tested = i
    df_iter.loc[ df_iter['PstP True Active'] == 1 ]
    n_hits = len( df_iter.loc[ df_iter['PstP True Active'] == 1 ] )
    n_unique_hits = int(df_unique['PstP True Active'].astype(int).sum())
    n_murcko = len( df_iter.loc[ df_iter['PstP True Active'] == 1, 'gen_murcko' ].unique() )
    n_clust = len( df_iter.loc[ df_iter['PstP True Active'] == 1, 'BT_0.4 ID' ].unique() )
    dock_hits.append( [n_tested, n_hits, n_unique_hits, n_murcko, n_clust] )

iter_hits = np.array( iter_hits )
dock_hits = np.array( dock_hits )
rand_hits = np.array( rand_hits )

## Setting if only want to plot 1 for simplicity
plot_one = True
plot_index = 0

if(plot_one):
    fig, axs = plt.subplots( 1, 1, figsize=(18, 12))
    #fig, axs = plt.subplots( 1, 1)
else:
    fig, axs = plt.subplots( 1, 4, figsize=(9,3), sharey=True )

plt.ylim((0,30))
x_iter_ticks = np.arange(0,4080,80)
#x_major_ticks = np.arange(0,4800,800)
x_major_ticks = np.arange(0, max_iteration +1, 1)
y_minor_ticks = np.arange(0,31,1)

hit_type_list = ['Total Hits', 'Total Unique Hits','generic_murcko','clusters_BT_0.4']
# x = len(axs)
fontsize = 27
sns.set_context("paper", font_scale=2.68)

if(plot_one):
    j = plot_index + 1

    #axs.set_title( hit_type_list[plot_index] , fontsize=fontsize)
    sns.lineplot(x=iter_hits[:,0], y=iter_hits[:,j], c='green', label='CBWS',   alpha=1.0, lw=5 )#, markeredgecolor='black' )
    sns.lineplot(x=dock_hits[:,0], y=dock_hits[:,j], c='red',   label='Docking',   alpha=1.0, lw=5 )#, markeredgecolor='black' )
    sns.lineplot(x=rand_hits[:,0], y=rand_hits[:,j], c='blue',  label='Random', alpha=1.0, lw=5 )#, markeredgecolor='black' )
    #axs.set_xticks( x_iter_ticks, minor=True )
    axs.set_xticks( x_major_ticks, minor=True)
    axs.set_yticks( y_minor_ticks, minor=True )
    axs.set_xlim((0,max_iteration+1))
    axs.tick_params(axis='both', labelsize=20)

    ## Setting legend off the plot
    axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    interval_list = [10, 20, 30, 40]
    smiles_to_plot = [random.choice(smiles_list[i]) for i in interval_list]
    x_positions = [iter_hits[:,0][i-1] for i in interval_list]
    y_positions = [iter_hits[:,j][i-1] for i in interval_list]
    molecules_to_show = [(10, (10, 25)), (20, (20, 25)), (30, (30, 14)), (40, (40,15.5))]

    for i, (x_off, y_off) in molecules_to_show:
        point_on_line = (iter_hits[:,0][i-1], iter_hits[:,j][i-1])
        mol_pos = (x_off, y_off)
        add_molecule_with_arrow(axs, random.choice(smiles_list[i]), point_on_line, mol_pos, zoom=0.85)

    # for smiles, x_pos, y_pos in zip(smiles_to_plot, x_positions, y_positions):
        # add_molecule_to_plot(axs, smiles, (x_pos, y_pos), zoom=0.5)
        # axs.plot(x_pos, y_pos, 'ro', markersize=8)

    # for i in [10, 20, 40]:
    #     smiles = random.choice(smiles_list[i])
    #     add_molecule_to_plot(axs, smiles, (iter_hits[:,0][i-1], iter_hits[:, j][i-1]),zoom=0.25 )

else:
    for i, hit_type in enumerate( hit_type_list):
        j = i + 1
        axs[i].set_title( hit_type )
        axs[i].plot( iter_hits[:,0], iter_hits[:,j], '-', c='green', label='CBWS',   alpha=1.0, lw=1 )#, markeredgecolor='black' )
        axs[i].plot( dock_hits[:,0], dock_hits[:,j], '-', c='red',   label='dock',   alpha=1.0, lw=1 )#, markeredgecolor='black' )
        axs[i].plot( rand_hits[:,0], rand_hits[:,j], '-', c='blue',  label='random', alpha=1.0, lw=1 )#, markeredgecolor='black' )
        axs[i].set_xticks( x_iter_ticks, minor=True )
        axs[i].set_xticks( x_major_ticks )
        axs[i].set_yticks( y_minor_ticks, minor=True )
        axs[i].set_xlim((0,4000))
        # axs[i].grid( which='both' )
        # axs[i].grid( which='minor', alpha=0.2 )
        # axs[i].grid( which='major', alpha=0.5 )
        if i == 0:
            axs[i].legend()

fig.supxlabel("Iteration Number", fontsize=fontsize)
fig.supylabel("Cumulative Total Hits", fontsize=fontsize)
#fig.suptitle( "PstP active compound retrieval" )
plt.tight_layout()
plt.show()
plt.savefig( 'hit_retrieval_CBWS_docking_random.png', dpi=2400 )
plt.close()
