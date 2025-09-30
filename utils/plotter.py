import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib import rcParams
import numpy as np
from .other_utils import directBinStrSum
from os import path, makedirs
from dolfin import plot as dlf_plt

#pass N_set values in to selectively pick for which N_values
#the difference between the empirical and analytical 
#sobol indices should be plotted.
def plot_ishi_diffs(model, 
                    N_set=[], 
                    which_experiment=0, 
                    multi_experiment=False,
                    only_singulars=False,
                    withOutliers=True,
                    withTrend=False,
                    save_fig=False,
                    no_title=False,
                    base_fontsize=12,
                    fig_name='', 
                    save_directory=''):
    """
    Plot absolute differences for Ishigami Sobol indices.
    
    Parameters:
    - N_set (list of int): List of sample sizes.
    - save_fig (bool): Whether to save the plot as a PDF.
    """
    def getPlotLabel(key):
        return ''.join(f"x_{len(key) - 1 - i}" if key[i] == '1' else '' for i in range(len(key)))
    
    rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'STIXGeneral'
    if base_fontsize < 10:
        base_fontsize = 10
    #needed or just set N_set to N_set [??]
    if len(N_set) == 0:
        N_set = model.N_set
    
    ###
    if not multi_experiment:
        if which_experiment not in model.exprimentDataDict[0].keys():
            print("Wrong experiment number (key) passed in. Experiment number set to 0.")
            which_experiment = 0
        model.sobolVals_clos = model.exprimentDataDict[0][which_experiment]['sobolVals_clos']
        ###
    if withTrend:
        y_expected_trend = 1 / np.sqrt(N_set)
    if not multi_experiment:
        plt.figure()
    if only_singulars:
        boxDataDict = {
                    '001': [],
                    '010': [],
                    '100': [],
                }
    else:
        boxDataDict = {
                    '001': [],
                    '010': [],
                    '100': [],
                    '011': [],
                    '101': [],
                    '110': [],
                    '111': []
                }
    for key in boxDataDict.keys():
        for _ in N_set:
            boxDataDict[key].append([])
    for expNum in range(len(model.exprimentDataDict[0].keys())):
        #Case where plotting an overlay of all experiments
        if multi_experiment:
            model.sobolVals_clos = model.exprimentDataDict[0][expNum]['sobolVals_clos']
        #Case where plotting just one experiment
        else:
            if which_experiment not in model.exprimentDataDict[0].keys():
                print("Wrong experiment number (key) passed in. Experiment number set to 0.")
                which_experiment = 0
            model.sobolVals_clos = model.exprimentDataDict[0][which_experiment]['sobolVals_clos']

        absDiffs = model.getAbsDiffs_ishi(N_set=N_set)
        #each diffs for each key contains diffs for all the different N_set Ns for for each key
        #[[N-val1, N-val2, ...]<-expNum1, []<-expNum2, ...]
        for key, diffs in absDiffs.items():
            if only_singulars:
                if directBinStrSum(key) > 1:
                    boxDataDict.pop(key, None)
                    continue
            if only_singulars:
                myLabel = {
                    '001': r'$S^{clos}_{x_1}$',
                    '010': r'$S^{clos}_{x_2}$',
                    '100': r'$S^{clos}_{x_3}$',
                }.get(key, getPlotLabel(key))
                myColors = {
                    '001': 'goldenrod',
                    '010': 'purple',
                    '100': 'firebrick',
                }.get(key, getPlotLabel(key))

            else:
                myLabel = {
                    '001': r'$S^{clos}_{x_1}$',
                    '010': r'$S^{clos}_{x_2}$',
                    '100': r'$S^{clos}_{x_3}$',
                    '011': r'$S^{clos}_{x_1x_2}$',
                    '101': r'$S^{clos}_{x_1x_3}$',
                    '110': r'$S^{clos}_{x_2x_3}$',
                    '111': r'$S^{clos}_{x_1x_2x_3}$'
                }.get(key, getPlotLabel(key))
                
                myColors = {
                    '001': 'goldenrod',
                    '010': 'purple',
                    '100': 'firebrick',
                    '011': 'blue',
                    '101': 'orange',
                    '110': 'green',
                    '111': 'red'
                }.get(key, getPlotLabel(key))

            for i in range(len(N_set)):
                boxDataDict[key][i].append(diffs[i])

            if not multi_experiment:
                plt.loglog(N_set, diffs, label=myLabel, c=myColors)
                plt.scatter(N_set, diffs, c=myColors)
        
        if not multi_experiment:
            break
    if not multi_experiment:
        plt.xscale('log')
        plt.yscale('log')
        if withTrend:
            plt.loglog(N_set, y_expected_trend, label=r'1/$\sqrt{N}$', color='royalblue', linestyle='--', linewidth=5)
        plt.legend()
        plt.xlabel(r"Number of Samples, $N$")
        plt.ylabel(r"$|S^{clos}_{A} - S^{clos}_{A, pf}|$")
        if not no_title:
            plt.title(r"Absolute difference between analytically and empirically calculated $S^{clos}_{A}$ using Pick-Freeze")
    else:
        if only_singulars: 
            fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)  # One subplot per key
            colors = ["tomato", "royalblue", "seagreen"]
        else:
            fig, axes = plt.subplots(1, 7, figsize=(20, 6), sharey=True)  # One subplot per key
            colors = ['goldenrod', 'purple', 'firebrick', 'blue', 'orange', 'green', 'red']
        if not no_title:
            fig.suptitle("Boxplots for absolute differences between analytically and empirically\n" rf"calculated $S^{{clos}}_{{A}}$ using Pick-Freeze, for {len(model.exprimentDataDict[0].keys())} experiments", fontsize=(base_fontsize+6))
        # fig.suptitle('Boxplots for each sobol index', fontsize=(base_fontsize+6))
        ax_counter = 0
        for ax, (key, values), color in zip(axes, boxDataDict.items(), colors):
            box = ax.boxplot(values, 
                            patch_artist=True, 
                            showfliers=withOutliers,
                            meanprops={"marker": "^",        
                                    "markerfacecolor": "red",   
                                    "markeredgecolor": "black"})
            ax.set_ylim(-0.29,1.29)
            if only_singulars:
                    myLabel = {
                    '001': r'$S^{clos}_{x_1}$',
                    '010': r'$S^{clos}_{x_2}$',
                    '100': r'$S^{clos}_{x_3}$',
                }.get(key, getPlotLabel(key))
            else:
                myLabel = {
                    '001': r'$S^{clos}_{x_1}$',
                    '010': r'$S^{clos}_{x_2}$',
                    '100': r'$S^{clos}_{x_3}$',
                    '011': r'$S^{clos}_{x_1x_2}$',
                    '101': r'$S^{clos}_{x_1x_3}$',
                    '110': r'$S^{clos}_{x_2x_3}$',
                    '111': r'$S^{clos}_{x_1x_2x_3}$'
                }.get(key, getPlotLabel(key))
            
            for patch in box['boxes']:
                patch.set_facecolor(color)
                patch.set_edgecolor("black")
            for whisker in box['whiskers']:
                whisker.set_color(color)
            for cap in box['caps']:
                cap.set_color(color)
            for median in box['medians']:
                median.set_color("black")

            if not no_title:
                ax.set_title(myLabel)
            # ax.set_xlabel(r"Number of Samples, $N$")
            # if ax_counter == 0:
            #     if only_singulars:
            #         ax.set_ylabel(r"$|S_{A} - \hat{S}_{A}|$")
            #     else:
            #         ax.set_ylabel(r"$|S^{clos}_{A} - S^{clos}_{A, pf}|$")
            #     ax_counter = 1
            
            if withTrend:
                # ax.plot(np.log10(N_set), y_expected_trend)
                # ax.plot(range(1, len(N_set) + 1), y_expected_trend, label=r'1/$\sqrt{N}$', color='black', linestyle='--', linewidth=2)
                ax.plot(range(1, len(N_set) + 1), y_expected_trend, label=r'$N^{-1/2}$', color='black', linestyle='--', linewidth=2)
            
            ax.tick_params(axis='x', labelsize=base_fontsize)
            ax.tick_params(axis='y', labelsize=base_fontsize)

            ax.set_xticks(range(1, len(N_set) + 1))  # Ensure x-ticks match the number of groups
            myXTicks = [rf"$10^{int(np.log10(x))}$" for x in N_set]
            ax.set_xticklabels(myXTicks, fontsize=base_fontsize)
            # ax.legend()
        if only_singulars:
            fig.supylabel(r"$|S_{A} - \hat{S}_{A}|$", fontsize=(base_fontsize+2))
        else:
            fig.supylabel(r"$|S^{clos}_{A} - S^{clos}_{A, pf}|$", fontsize=(base_fontsize+2))

        # plt.rcParams['font.family'] = 'STIXGeneral'
        fig.supxlabel(r"Number of Samples, $N$", fontsize=(base_fontsize+2), y=0.02)
        box_handles = [mpatches.Patch(color=c, label=rf"$U_{i+1}$") for i, c in enumerate(colors)]
        # trend_handle = mlines.Line2D([], [], color="black", linestyle="--", linewidth=2, label=r"1/$\sqrt{N}$")
        trend_handle = mlines.Line2D([], [], color="black", linestyle="--", linewidth=2, label=r"$N^{-1/2}$")
        if base_fontsize <= 11:
            vert_space_add_to_legend = 0.02
        else:
            vert_space_add_to_legend = (base_fontsize-14)/100
        fig.legend(handles=box_handles + [trend_handle],
                    loc="upper center",
                    ncol=len(colors) + 1,
                    bbox_to_anchor=(0.5, 1.00+vert_space_add_to_legend),
                    fontsize=(base_fontsize+1),
                    frameon=False) 
        # fig.legend(handles=box_handles + [trend_handle], loc="upper right", fontsize=10)#bbox_to_anchor=(1.1, 0.5))
        # plt.xscale('symlog')
        # plt.yscale('log')
            # ax.set_xscale('log')
        fig.tight_layout(rect=[0.01, 0, 1, 0.97]) 
    if save_fig:
        if save_directory == '':
            save_directory = model.figsave_directory
        else:
            if not path.exists(save_directory):
                makedirs(save_directory)
        if fig_name == '':
            fig_name = 'ishigami_pickfreeze.pdf'
        plt.savefig(f"{save_directory}/{fig_name}", dpi=900, bbox_inches='tight')
    else:
        plt.show()

def plot_ishi(model, 
                N_set=[], 
                which_experiment=0, 
                multi_experiment=False,
                only_singulars=False,
                only_aggr=False,
                withOutliers=False,
                which_N=None,
                plot_which_sobols='clos',
                which_interval=[0,0],
                interval_toggle=False,
                base_fontsize=10,
                withTrend=False,
                no_title=False,
                save_fig=False, 
                fig_name='', 
                save_directory='',
                toy_1_or_2=False,
                grid_toggle=False):
    """
    Plot result of Closed sobol indices empirical estimations.
    
    Parameters:
    - N_set (list of int): List of sample sizes.
    - save_fig (bool): Whether to save the plot as a PDF.
    """
    def getPlotLabel(key):
        return ''.join(f"x_{len(key) - 1 - i}" if key[i] == '1' else '' for i in range(len(key)))
    
    rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'STIXGeneral'
    #needed or just set N_set to N_set [??]
    if len(N_set) == 0:
        N_set = model.N_set
    
    ###
    which_interval_str = str(which_interval).replace(" ", "")
    if which_N is None:
        which_N = model.N_set[0]
    if model.model_type == 'ishigami':
        if not multi_experiment:
            if which_experiment not in model.exprimentDataDict[0].keys():
                print("Wrong experiment number (key) passed in. Experiment number set to 0.")
                which_experiment = 0
            model.sobolVals_clos = model.exprimentDataDict[0][which_experiment]['sobolVals_clos']
            ###
    else:
        if interval_toggle:
            sobolDict = model.exprimentDataDict[which_interval_str][which_experiment]["sobolVals_clos_aggr"][f"{which_N}"]
    if withTrend:
        y_expected_trend = 1 / np.sqrt(N_set)
    if not multi_experiment:
        plt.figure()
    if interval_toggle or model.specifyX3:
        if only_singulars:
            boxDataDict = {
                    '01': [],
                    '10': [],
                    }
        else:
            boxDataDict = {
                    '01': [],
                    '10': [],
                    '11': [],
                }
    else:
        if only_singulars and not toy_1_or_2:
            boxDataDict = {
                    '001': [],
                    '010': [],
                    }
        elif not toy_1_or_2:
            boxDataDict = {
                    '001': [],
                    '010': [],
                    '011': [],
                }
        else:
            boxDataDict = {
                    '001': [],
                    '010': [],
                    '100': [],
                }
    for key in boxDataDict.keys():
        for _ in N_set:
            boxDataDict[key].append([])
    if interval_toggle:
        expLen = len(model.exprimentDataDict[which_interval_str].keys())
    else:
        expLen = len(model.exprimentDataDict[0].keys())
    for expNum in range(expLen):
        #Case where plotting an overlay of all experiments
        if multi_experiment:
            # if <---------
            if interval_toggle:
                sobolDict = model.exprimentDataDict[which_interval_str][expNum]["sobolVals_clos_aggr"]
            elif only_aggr:
                sobolDict = model.exprimentDataDict[0][expNum]['sobolVals_clos_aggr']
            else:
                #THIS IS VERY INEFFICIENT... FIX SO NO DATA PINGPONGING BETWEEN MODEL & PLOTTER!!
                model.sobolVals_clos = model.exprimentDataDict[0][expNum]['sobolVals_clos']
                sobolDict = model.exprimentDataDict[0][expNum]['sobolVals_clos']
        #Case where plotting just one experiment
        else:
            if which_experiment not in model.exprimentDataDict[0].keys():
                print("Wrong experiment number (key) passed in. Experiment number set to 0.")
                which_experiment = 0
            model.sobolVals_clos = model.exprimentDataDict[0][which_experiment]['sobolVals_clos']

        absDiffs = model.getAbsDiffs_ishi(N_set=N_set, get_diffs=False)
        #each diffs for each key contains diffs for all the different N_set Ns for for each key
        #[[N-val1, N-val2, ...]<-expNum1, []<-expNum2, ...]
        
        
        if interval_toggle or model.specifyX3 and not toy_1_or_2:
            if only_singulars:
                # NOTe THESE ARE INCUUSION SetS noT EXCLUSION!!!
                exclusion_set = ['01', '10']
            else:
                exclusion_set = ['01', '10', '11']
        elif not toy_1_or_2:
            if only_singulars:
                exclusion_set = ['001', '010']
            else:
                exclusion_set = ['001', '010', '011']
        else:
            if only_singulars:
                exclusion_set = ['001', '010', '100']
            else:
                exclusion_set = ['001', '010', '100', '011', '110', '101']
        if interval_toggle or model.specifyX3 or toy_1_or_2:
            N_idx_counter = 0 
            for key_N, val_at_N in sobolDict.items():
                for key_S, val_at_S in val_at_N.items():
                    if key_S not in exclusion_set:
                        continue
                    if not toy_1_or_2:
                        myLabel = {
                        '01': r'$S^{clos}_{x_1}$',
                        '10': r'$S^{clos}_{x_2}$',
                        '11': r'$S^{clos}_{x_1x_2}$',
                        }.get(key, getPlotLabel(key))
                        myColors = {
                            '01': 'goldenrod',
                            '10': 'purple',
                            '11': 'blue',
                        }.get(key, getPlotLabel(key))
                    else:
                        myLabel = {
                        '001': r'$S^{clos}_{x_1}$',
                        '010': r'$S^{clos}_{x_2}$',
                        '100': r'$S^{clos}_{x_3}$',
                        }.get(key, getPlotLabel(key))
                        myColors = {
                            '001': 'goldenrod',
                            '010': 'purple',
                            '100': 'blue',
                        }.get(key, getPlotLabel(key))
                    boxDataDict[key_S][N_idx_counter].append(val_at_S)
                N_idx_counter += 1
        else:
            for key, diffs in absDiffs.items():
                if key not in exclusion_set:
                    continue
                myLabel = {
                    '001': r'$S^{clos}_{x_1}$',
                    '010': r'$S^{clos}_{x_2}$',
                    '100': r'$S^{clos}_{x_3}$',
                    '011': r'$S^{clos}_{x_1x_2}$',
                    '101': r'$S^{clos}_{x_1x_3}$',
                    '110': r'$S^{clos}_{x_2x_3}$',
                    '111': r'$S^{clos}_{x_1x_2x_3}$'
                }.get(key, getPlotLabel(key))
                
                myColors = {
                    '001': 'goldenrod',
                    '010': 'purple',
                    '100': 'firebrick',
                    '011': 'blue',
                    '101': 'orange',
                    '110': 'green',
                    '111': 'red'
                }.get(key, getPlotLabel(key))

                for i in range(len(N_set)):
                    boxDataDict[key][i].append(diffs[i])

            if not multi_experiment:
                plt.loglog(N_set, diffs, label=myLabel, c=myColors)
                plt.scatter(N_set, diffs, c=myColors)
        
        if not multi_experiment:
            break
    if withTrend:
        # the mean of values for each key, at the largest sample number in N_set (N_set[-1]). This gets added to the trend-line
        most_accurate_mean_for_all_S = {key: np.mean(boxDataDict[key][-1])-(1/np.sqrt(N_set[-1])) for key in boxDataDict.keys()}
    # for key_S in boxDataDict.keys():
        # mostAccurateMeans[key_S] = np.mean(boxDataDict[key_S][-1])
    if not multi_experiment:
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
        plt.xscale('log')
        plt.yscale('log')
        if withTrend:
            plt.loglog(range(1, len(N_set) + 1), y_expected_trend, label=r'1/$\sqrt{N}$', color='royalblue', linestyle='--', linewidth=2)
        plt.legend()
        plt.xlabel(r"Number of Samples, $N$")
        plt.ylabel(r"$ S^{clos}_{A, pf}$")
        if not no_title:
            plt.title(r"Empirically calculated $S^{clos}_{A}$ using Pick-Freeze")
    else:
        if only_singulars and not toy_1_or_2:
            fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)  # One subplot per key
            colors = ["tomato", "royalblue"]
        else:
            fig, axes = plt.subplots(1, 3, figsize=(20, 8), sharey=True)  # One subplot per key
            colors = ["tomato", "royalblue", "seagreen"]
        if not no_title:
            if 'toy' in model.model_type:
                if only_aggr:
                    fig.suptitle(rf"Boxplots for empirically calculated $S^{{aggr, clos}}_{{A}}$ using Pick-Freeze, for {len(model.exprimentDataDict[0].keys())} experiments", fontsize=(base_fontsize+6))
                else:
                    fig.suptitle(rf"Boxplots for empirically calculated $S^{{clos}}_{{A}}$ using Pick-Freeze, for {len(model.exprimentDataDict[0].keys())} experiments", fontsize=(base_fontsize+6))
            elif model.specifyX3:
                fig.suptitle(rf"Boxplots for empirically calculated $S^{{clos}}_{{A}}$ using Pick-Freeze, for {len(model.exprimentDataDict[0].keys())} experiments"+"\nFixed "+rf"$x_3$={model.x_3}", fontsize=(base_fontsize+6))
            elif interval_toggle:
                which_interval_title_str = rf"$x_3\in{list(np.round(which_interval,decimals=2))}$".replace("np.float64(", "").replace(")", "")
                fig.suptitle(rf"Boxplots for empirically calculated $S^{{clos}}_{{A}}$ using Pick-Freeze, for {len(model.exprimentDataDict[which_interval_str].keys())} experiments on"+"\n"+which_interval_title_str, fontsize=(base_fontsize+6))
            else:
                fig.suptitle(rf"Boxplots for empirically calculated $S^{{clos}}_{{A}}$ using Pick-Freeze, for {len(model.exprimentDataDict[0].keys())} experiments", fontsize=(base_fontsize+6))
        # fig.suptitle('Boxplots for each sobol index', fontsize=16)
        ax_counter = 0
        for ax, (key, values), color in zip(axes, boxDataDict.items(), colors):
            if key not in exclusion_set:
                continue
            box = ax.boxplot(values, 
                            patch_artist=True, 
                            showmeans=True, 
                            showfliers=withOutliers,
                            meanprops={"marker": "^",        
                                    "markerfacecolor": "red",   
                                    "markeredgecolor": "black"})
            ax.set_ylim(-0.29,1.29)
            ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
            ax.axhline(y=1, color='black', linestyle='--', linewidth=1)

            for patch in box['boxes']:
                patch.set_facecolor(color)
                patch.set_edgecolor("black")
            for whisker in box['whiskers']:
                whisker.set_color(color)
            for cap in box['caps']:
                cap.set_color(color)
            for median in box['medians']:
                median.set_color("black")

            # if interval_toggle or model.specifyX3:
            #     myLabel = {
            #         '01': r'$S^{clos}_{x_1}$',
            #         '10': r'$S^{clos}_{x_2}$',
            #         '11': r'$S^{clos}_{x_1x_2}$',
            #     }.get(key, getPlotLabel(key))
            # else:
            #     myLabel = {
            #         '001': r'$S^{clos}_{x_1}$',
            #         '010': r'$S^{clos}_{x_2}$',
            #         '100': r'$S^{clos}_{x_3}$',
            #         '011': r'$S^{clos}_{x_1x_2}$',
            #         '101': r'$S^{clos}_{x_1x_3}$',
            #         '110': r'$S^{clos}_{x_2x_3}$',
            #         '111': r'$S^{clos}_{x_1x_2x_3}$'
            #     }.get(key, getPlotLabel(key))
            # ax.set_title(myLabel)
            # ax.set_xlabel(r"Number of Samples, $N$")
            if ax_counter == 0:
                # ax.set_ylabel(r"$\hat{S}^{aggr}_{A}$", fontsize=base_fontsize)
                ax_counter = 1
            if grid_toggle:
                ax.grid(True, axis='y')
                ax.set_yticks(np.arange(0, 1.01, 0.1))
            ax.tick_params(axis='x', labelsize=base_fontsize)
            ax.tick_params(axis='y', labelsize=base_fontsize)
            ax.set_xticks(range(1, len(N_set) + 1))  # Ensure x-ticks match the number of groups
            myXTicks = [rf"$10^{int(np.log10(x))}$" for x in N_set]
            ax.set_xticklabels(myXTicks)
            if withTrend:
                # ax.plot(np.log10(N_set), y_expected_trend)
                y_expected_trend_curr_key = y_expected_trend + most_accurate_mean_for_all_S[key]
                ax.plot(range(1, len(N_set) + 1), y_expected_trend_curr_key, label=r'$N^{-1/2}$', color='black', linestyle='--', linewidth=2)
        
        # plt.rcParams['font.family'] = 'STIXGeneral'
        fig.supylabel(r"$\hat{S}^{aggr}_{A}$", fontsize=(base_fontsize+2))
        fig.supxlabel(r"Number of Samples, $N$", fontsize=(base_fontsize+2), y=0.02)
        box_handles = [mpatches.Patch(color=c, label=rf"$U_{i+1}$") for i, c in enumerate(colors)]
        trend_handle = mlines.Line2D([], [], color="black", linestyle="--", linewidth=2, label=r"$N^{-1/2}$")
        if base_fontsize <= 11:
            vert_space_add_to_legend = 0.02
        else:
            vert_space_add_to_legend = (base_fontsize-14)/100
        fig.legend(handles=box_handles + [trend_handle],
                    loc="upper center",
                    ncol=len(colors) + 1,
                    bbox_to_anchor=(0.5, 1.00+vert_space_add_to_legend),
                    fontsize=(base_fontsize+1),
                    frameon=False) 
        fig.tight_layout(rect=[0.01, 0, 1, 0.97]) 

    if save_fig:
        if save_directory == '':
            save_directory = model.figsave_directory
        else:
            if not path.exists(save_directory):
                makedirs(save_directory)
        if fig_name == '':
            fig_name = 'ishigami_pickfreeze.pdf'
        plt.savefig(f"{save_directory}/{fig_name}", dpi=900, bbox_inches='tight')
    else:
        plt.show()


#if calculations done for multiple N values, pass the N value
#for which you want to plot the empirically-calculated sobol indices

#plot_which: closed, main, closed_aggr
def plot_1d_diffusion(model,                   
                    N_set = [],                   
                    withOutliers=True,                    
                    which_interval=None,
                    which_N=None,
                    withTrend=True,
                    grid_toggle=False,
                    base_fontsize=18,                   
                    only_singulars=True,
                    save_fig=False, 
                    fig_name='', 
                    save_directory='',
                    which_experiment=0,
                    multi_experiment=False,
                    which_index=None,
                    with_singulars=True, 
                    interval_toggle=False,                  
                    no_title=False,
                    plot_aggr=False,
                    plot_main=False,
                    plot_which_sobols='closed_aggr',):     
    
    rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'STIXGeneral'
    if which_N is None:
        which_N = model.N_set[0]

    if len(N_set) == 0:
        N_set = model.N_set

    if which_interval is None and which_index is None:
        which_interval = list(model.exprimentDataDict.keys())[0]
    if which_interval is not None and which_index is None:
        which_interval_str = str(which_interval).replace(" ", "")

    random_input_indices_oneHot_list = []
    if only_singulars:
        if which_interval is not None:
            for index_str in list(model.exprimentDataDict[which_interval_str][0]["sobolVals_clos_aggr"][f"{N_set[0]}"].keys()):
                if directBinStrSum(index_str) == 1:
                    random_input_indices_oneHot_list.append(index_str)
        else:
            for index_str in list(model.exprimentDataDict[which_index][0]["sobolVals_clos"][f"{N_set[0]}"].keys()):
                if directBinStrSum(index_str) == 1:
                    random_input_indices_oneHot_list.append(index_str)
    else:
        if which_interval is not None:
            random_input_indices_oneHot_list = list(model.exprimentDataDict[which_interval_str][0]["sobolVals_clos_aggr"][N_set[0]].keys())
        else:
            random_input_indices_oneHot_list = list(model.exprimentDataDict[which_index][0]["sobolVals_clos"][f"{N_set[0]}"].keys())

    boxDataDict = {index_str: [] for index_str in random_input_indices_oneHot_list}
    for key in boxDataDict.keys():
        for _ in N_set:
            boxDataDict[key].append([])

    if which_interval is not None:
        expLen = len(model.exprimentDataDict[which_interval_str].keys())
    else:
        expLen = len(model.exprimentDataDict[which_index].keys())

    for expNum in range(expLen):
        if which_interval is not None:
            sobolDict = model.exprimentDataDict[which_interval_str][expNum]["sobolVals_clos_aggr"]
        else:
            sobolDict = model.exprimentDataDict[which_index][expNum]["sobolVals_clos"]
        N_idx_counter = 0 
        for _, val_at_N in sobolDict.items():
            for key_S, val_at_S in val_at_N.items():
                if only_singulars and directBinStrSum(key_S) > 1:
                    continue
                boxDataDict[key_S][N_idx_counter].append(val_at_S)
            N_idx_counter += 1
        
    if withTrend:
        y_expected_trend = 1 / np.sqrt(N_set)
        # the mean of values for each key, at the largest sample number in N_set (N_set[-1]). This gets added to the trend-line
        most_accurate_mean_for_all_S = {key: np.mean(boxDataDict[key][-1])-(1/np.sqrt(N_set[-1])) for key in boxDataDict.keys()}

    num_of_A_keys = len(random_input_indices_oneHot_list)
    if len(random_input_indices_oneHot_list) <= 3:
        fig, axes = plt.subplots(1, num_of_A_keys, figsize=(20, 8), sharey=True)  # One subplot per key
    else:
        fig, axes = plt.subplots(int(np.floor(num_of_A_keys / 3))+1, 3, figsize=(20, 8), sharey=True) 
    colors = ["tomato", "royalblue", "seagreen", "goldenrod", "orchid", "sienna", "turquoise", "darkorange", "slategray", "deeppink"][:num_of_A_keys]
    if which_interval is not None:
        print(f"Showing plot for interval: {which_interval_str}")
    else:
        print(f"Showing plot for index: {which_index}")
    for ax, (key, values), color in zip(axes, boxDataDict.items(), colors):
        box = ax.boxplot(values, 
                        patch_artist=True, 
                        showmeans=True, 
                        showfliers=withOutliers,
                        meanprops={"marker": "^",        
                                    "markerfacecolor": "red",   
                                    "markeredgecolor": "black"})
        ax.set_ylim(-0.29,1.29)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.axhline(y=1, color='black', linestyle='--', linewidth=1)

        for patch in box['boxes']:
            patch.set_facecolor(color)
            patch.set_edgecolor("black")
        for whisker in box['whiskers']:
            whisker.set_color(color)
        for cap in box['caps']:
            cap.set_color(color)
        for median in box['medians']:
            median.set_color("black")

        if grid_toggle:
            ax.grid(True, axis='y')
            ax.set_yticks(np.arange(0, 1.01, 0.1))
        ax.tick_params(axis='x', labelsize=base_fontsize)
        ax.tick_params(axis='y', labelsize=base_fontsize)
        ax.set_xticks(range(1, len(N_set) + 1))  # Ensure x-ticks match the number of groups
        myXTicks = [rf"$10^{int(np.log10(x))}$" for x in N_set]
        ax.set_xticklabels(myXTicks)
        if withTrend:
            # ax.plot(np.log10(N_set), y_expected_trend)
            y_expected_trend_curr_key = y_expected_trend + most_accurate_mean_for_all_S[key]
            ax.plot(range(1, len(N_set) + 1), y_expected_trend_curr_key, label=r'$N^{-1/2}$', color='black', linestyle='--', linewidth=2)
    
    # plt.rcParams['font.family'] = 'STIXGeneral'
    fig.supylabel(r"$\hat{S}^{aggr}_{A}$", fontsize=(base_fontsize+2))
    fig.supxlabel(r"Number of Samples, $N$", fontsize=(base_fontsize+2), y=0.02)
    box_handles = [mpatches.Patch(color=c, label=rf"$\xi_{i+1}$") for i, c in enumerate(colors)]
    if withTrend:
        trend_handle = mlines.Line2D([], [], color="black", linestyle="--", linewidth=2, label=r"$N^{-1/2}$")
    if base_fontsize <= 11:
        vert_space_add_to_legend = 0.02
    else:
        vert_space_add_to_legend = (base_fontsize-16)/100
    
    if withTrend:
        fig.legend(handles=box_handles + [trend_handle],
                    loc="upper center",
                    ncol=len(colors) + 1,
                    bbox_to_anchor=(0.5, 1.00+vert_space_add_to_legend),
                    fontsize=(base_fontsize+1),
                    frameon=False) 
    else:
        fig.legend(handles=box_handles,
                loc="upper center",
                ncol=len(colors) + 1,
                bbox_to_anchor=(0.5, 1.00+vert_space_add_to_legend),
                fontsize=(base_fontsize+1),
                frameon=False) 
    if len(random_input_indices_oneHot_list) > 3:
        for i in range(num_of_A_keys, (int(np.floor(num_of_A_keys / 3))+1)*3):
            fig.delaxes(axes[i])

    fig.tight_layout(rect=[0.01, 0, 1, 0.97]) 
    if save_fig:
        if save_directory == '':
            save_directory = model.figsave_directory
        if not path.exists(save_directory):
            makedirs(save_directory)
        if fig_name == '':
            fig_name = '1d_diffusion_pf.pdf'
        if '.pdf' not in fig_name[-4:]:
            fig_name += '.pdf'
        plt.savefig(f"{save_directory}/{fig_name}", dpi=900, bbox_inches='tight')
    # fig.show()
    # match plot_which_sobols:
    #     case "closed_aggr":
    #         if interval_toggle:
    #             sobolDict = model.exprimentDataDict[which_interval_str][which_experiment]["sobolVals_clos_aggr"][f"{which_N}"]
    
    

    


    
    
    # if plot_aggr:
    #     sobolDict = model.exprimentDataDict[which_experiment]["sobolVals_clos_aggr"][f"{which_N}"]
    # elif plot_main:
    #     sobolDict = model.exprimentDataDict[which_experiment]["sobolVals_clos"][f"{which_N}"]
    # else:
    #     sobolDict = model.exprimentDataDict[which_experiment]["sobolVals_main"][f"{which_N}"]

    # a_combo_len = len(sobolDict)

    # # Determine x-axis values based on singular terms and total combinations
    # if with_singulars:
    #     x_axis_vals = np.linspace(1, model.P, model.P) if only_singulars else np.linspace(1, a_combo_len, a_combo_len)
    # else:
    #     axis_offset = len(np.binary_repr(a_combo_len,))
    #     x_axis_vals = np.linspace(1, a_combo_len - axis_offset, a_combo_len - axis_offset)

    # # Helper function for labels
    # def get_plot_label(key):
    #     label = ""
    #     #flip eg 110 to 011, since coefficient indicies are saved right to left and need to be flipped
    #     #eg 110 corresponds to a2a3, but at index 0, we have the value 0
    #     key = flipStr(key)
    #     for i in range(len(key)):
    #         if key[i] == '1':
    #             label += f"a{i+1}" 
    #     return label

    # # Plotting
    # plt.figure()
    # xIdx = 0
    # myLabels = []

    # for key in sobolDict.keys():
    #     # Filtering out non-singular indices when needed
    #     if only_singulars:
    #         if directBinStrSum(key) != 1:
    #             continue
    #     if not with_singulars:
    #         if directBinStrSum(key) == 1:
    #             continue

    #     myLabel = get_plot_label(key)
    #     myLabels.append(myLabel)
    #     # print(f"{type(x_axis_vals)}, \n {x_axis_vals}\n\n{type(sobolDict)}, \n{sobolDict}")
    #     plt.scatter(x_axis_vals[xIdx], sobolDict[key], label=myLabel, color='b')
    #     xIdx += 1

    
    # plt.xticks(x_axis_vals, myLabels, rotation=30)
    # plt.xlabel(r"Diffusion coefficient expansion indices")
    # if 'explicit' not in model.model_type and 'aggr' not in plot_which_sobols:
    #     which_index = f'{which_index/model.meshInterval:.5f}'
    # elif 'explicit' in model.model_type:
    #     which_index = f'{which_index:.5f}'
    # match plot_which_sobols:
    #     case "closed":
    #         plt.ylabel(r"$S^{clos}_{A}$")
    #         if not no_title:
    #             plt.title(r"Sobol' indices $S^{clos}_{A}$ for 1D Diffusion " + f"with random coefficients, using Pick-Freeze\nx={which_index}\n" fr"N={which_N} | h={model.meshInterval} | P={model.P} | $\mu$={model.mean} | $\sigma$={model.std}")
    #     case "closed_aggr":
    #         plt.ylabel(r"$S^{clos,Y}_{A}$")
    #         if not no_title:
    #             plt.title(r"Sobol' indices $S^{clos,Y}_{A}$ for 1D Diffusion with random coefficients, using (vector-valued) Pick-Freeze" + "\n" + r"$x\in$ " + f"{which_interval}" + "\n" + fr"N={which_N} | h={model.meshInterval} | P={model.P} | $\mu$={model.mean} | $\sigma={model.std}$")
    #     case "main":
    #         plt.ylabel(r"$S_{A}$")
    #         if not no_title:
    #             plt.title(r"Main Sobol' indices $S_{A}$ for 1D Diffusion " + f"with random coefficients, using Pick-Freeze\nx={which_index}\n" fr"N={which_N} | h={model.meshInterval} | P={model.P} | $\mu$={model.mean} | $\sigma$={model.std}")
    # # if model.model_type == 'diffusion_1D' or plot_aggr:
    # #     plt.ylabel(r"$S^{clos,Y}_{A}$")
    # #     plt.title(r"Sobol' indices $S^{clos,Y}_{A}$ for 1D Diffusion with random coefficients, using (vector-valued) Pick-Freeze" + "\n" + fr"N={which_N} | h={model.meshInterval} | P={model.P} | $\mu$={model.mean} | $\sigma={model.std}$")
    # # else:
    # #     plt.ylabel(r"$S^{clos}_{A}$")
    # #     plt.title(r"Sobol' indices $S^{clos}_{A}$ for 1D Diffusion" + f"with random coefficients, using Pick-Freeze\nx={model.scalarDiffuIdx}\n" fr"N={which_N} | h={model.meshInterval} | P={model.P} | $\mu$={model.mean} | $\sigma$={model.std}")
    # if save_fig:
    #     if save_directory == '':
    #         save_directory = model.figsave_directory
    #     else:
    #         if not path.exists(save_directory):
    #             makedirs(save_directory)
    #     if fig_name == '':
    #         fig_name ==  'diffusion_sobol_indices.pdf'
    #     plt.savefig(f"{save_directory}/{fig_name}", dpi=900, bbox_inches='tight')
    
    plt.show()

def plot_model_realization(model,
                        base_fontsize=16,
                        no_title=False, 
                        with1D_Diffu_mean=False,
                        save_fig=False, 
                        fig_name='', 
                        save_directory=''):
    rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'STIXGeneral'

    fig = plt.figure(figsize=(18,8))
    min_x = np.min(model.mesh_coords)
    max_x = np.max(model.mesh_coords)
    plt.xlim([min_x, max_x])
    x_vals = model.mesh_coords
    for key in model.realizationDataDict.keys():
        y_vals = model.realizationDataDict[key]
        plt.plot(x_vals, y_vals, label='_nolegend_')
    plt.xlabel("Spatial Domain", fontsize=base_fontsize+2)
    plt.ylabel(r"u(x,$\omega$)",  fontsize=base_fontsize+2)
    # plt.rcParams['font.family'] = 'STIXGeneral'
    if with1D_Diffu_mean:
        y_analyt = -0.5*(np.multiply(x_vals,x_vals)-x_vals)
        plt.plot(x_vals,y_analyt,'bo-', linewidth=1.5, label=r'$u(x)=-\frac{1}{2\mu}(x^2-x)$')
    if not no_title:
        if "diffusion_1D" in model.model_type:
            plt.title("Realizations of diffusion_1D with random coefficient")
        else:
            plt.title(f"Realizations of {model.model_type} with random coefficient")
    plt.ylim(-0.005, 0.155)
    plt.minorticks_on()
    plt.grid(True, which='both')
    # plt.grid(True)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    plt.xticks(fontsize=base_fontsize)
    plt.yticks(fontsize=base_fontsize)
    # if base_fontsize <= 11:
    #     vert_space_add_to_legend = 0.02
    # else:
    #     vert_space_add_to_legend = (base_fontsize-12)/100
    # fig.legend(handles=box_handles + [trend_handle],
    #             loc="upper center",
    #             ncol=len(colors) + 1,
    #             bbox_to_anchor=(0.5, 1.00+vert_space_add_to_legend),
    #             fontsize=(base_fontsize+1),
    #             frameon=False) 
    fig.tight_layout(rect=[0.01, 0, 1, 0.97]) 
    plt.legend(fontsize=base_fontsize+2, frameon=True)
    if save_fig:
        if save_directory == '':
            save_directory = model.figsave_directory
        else:
            if not path.exists(save_directory):
                makedirs(save_directory)
        if fig_name == '':
            fig_name = '1d_diffusion_realization.pdf'
        plt.savefig(f"{save_directory}/{fig_name}", dpi=900, bbox_inches='tight')
    plt.show()
    # return 0

def plot_2D_mesh(model,
                no_title=False,
                base_fontsize=16,
                save_fig=False, 
                fig_name='spat_dom_mesh', 
                save_directory=''):
    coordinates = model.mesh_coords
    cells = model.mesh_cells
    plt.figure(figsize=(18,8))
    plt.rcParams['font.family'] = 'STIXGeneral'
    for cell in cells:
        # Get the vertices of the current cell
        vertices = coordinates[cell]
        
        # Plot the edges of the triangle
        for i in range(len(cell)):
            # Create the line between vertices[i] and vertices[(i+1)%3]
            start, end = vertices[i], vertices[(i + 1) % 3]
            plt.plot([start[0], end[0]], [start[1], end[1]], color='black')

    # Set equal scaling and display the plot
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('cm', fontsize=base_fontsize+2)
    plt.ylabel('cm', fontsize=base_fontsize+2)
    plt.xticks(fontsize=base_fontsize)
    plt.yticks(fontsize=base_fontsize)
    if not no_title:
        plt.title(f'Spatial-domain mesh\nResolution: {model.mesh_steps}cm')
    if save_fig:
        if not path.exists(save_directory):
                makedirs(save_directory)
        print(f'Saving in \n{save_directory}\nas\n{fig_name}.pdf')
        plt.savefig(f'{save_directory}/{fig_name}.pdf', dpi=900, bbox_inches='tight')
    plt.show()

def plot_cdr_output_at_t_now(model, 
                            no_title=False,
                            save_fig=False, 
                            fig_save_dir=''):
    fuel_field_t_now = model.fuel_field_t_now
    oxyxen_field_t_now = model.oxyxen_field_t_now
    product_field_t_now = model.product_field_t_now
    temp_field_t_now = model.temp_field_t_now

    fuel_field_t_now.rename("Y_F", "Fraction of Fuel")
    oxyxen_field_t_now.rename("Y_O", "Fraction of Oxygen")
    product_field_t_now.rename("Y_P", "Fraction of Product")
    temp_field_t_now.rename("T", "Temperature")

    num_steps = model.num_steps

    components = [(fuel_field_t_now, [r"Mass Fraction of $H_2$ (Fuel)" + f"\nTime Steps:{num_steps}", "fuel_field"]),
                (oxyxen_field_t_now, [r"Mass Fraction of $O_2$ (Oxygen)" + f"\nTime Steps:{num_steps}", "oxygen_field"]),
                (product_field_t_now, [r"Mass Fraction of $H_2O$ (Product)" + f"\nTime Steps:{num_steps}", "product_field"]),
                (temp_field_t_now, [r"Temperature $K$" + f"\nTime Steps:{num_steps}", "temp_field"])]

    for component, title in components:
        plt.figure()
        plot_title = title[0]
        p = dlf_plt(component, title=plot_title)
        plt.colorbar(p)
        plt.xlabel(r"x-axis $[cm]$")
        plt.ylabel(r"y-axis $[cm]$")
        if not no_title:
            plt.title(plot_title)
        if save_fig:
            plt.savefig(f'{fig_save_dir}/{title[1]}.pdf', dpi=900, bbox_inches='tight')
    plt.show()


def plot(model, plot_type='sobols', not_differences=False, **kwargs):
    # #NOT AS ROBUST OF A CHECK FOR N_SET CONDITIONS!!! RE-DO!!
    # if "N_set" in kwargs.keys() and len(kwargs["N_set"]) != 0:
    #     N_set=model.N_set
    """
    Main plotting function, calling the appropriate plotting method based on model type.
    
    Parameters:
    - N_set (list of int): List of sample sizes.
    - save_fig (bool): Whether to save the plot as a PDF.
    """
    if plot_type == 'sobols':
        if model.model_type in ["ishigami", "ishigami_vect"] and not not_differences:
            plot_ishi_diffs(model, **kwargs)
        elif model.model_type in ["ishigami", "ishigami_vect", "toy_model_vect", 'toy_1_vect', 'toy_2_vect'] and not_differences:
            plot_ishi(model, **kwargs)
        elif "diffusion_1D" in model.model_type:
            plot_1d_diffusion(model, **kwargs)
        else:
            raise ValueError("Invalid model type. Cannot plot.")
    elif plot_type == 'realizations':
        plot_model_realization(model, **kwargs)
    elif plot_type == '2D_mesh':
        plot_2D_mesh(model, **kwargs)
    elif plot_type == 'cdr_results':
        plot_cdr_output_at_t_now(model, **kwargs)
    else:
        raise ValueError("Invalid plot type. Cannot plot.")


# #TO DO#
# """
# -> Integrate Vector_diffusion module
# -> Grapher for vector_diffusion module
# -> Grapher for comparing the two
# 
# -->> Fix boxplot title, axis labels & any other annotations for plot_sobols with multi_experiment=True
# """