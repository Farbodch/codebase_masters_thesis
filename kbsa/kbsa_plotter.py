import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from utils.other_utils import directBinStrSum, getSingletonIndexAsInt
from os import path, makedirs
import numpy as np

def _plot_this_(ax,
                my_data_list,
                num_of_kernels = 1,
                num_of_A_keys = 1,
                per_kernel_colors = plt.cm.tab10.colors,
                HSIC_U_keys_list = [],
                kernel_names_list = [],
                model_name = '',
                num_of_trials = 1,
                no_title=False,
                n = 1,
                m = 1,
                estimator_type='',
                base_fontsize=16,
                no_individual_legends=False,
                grid_toggle=False):
        
        per_kernel_box_spacing = 2.0
        box_width = 0.8 / num_of_kernels
        # fig, ax = plt.subplots()
        for per_kernel_idx, per_kernel_data in enumerate(my_data_list):
            offset = per_kernel_idx * box_width
            bplot_positions = np.arange(num_of_A_keys)*per_kernel_box_spacing + offset
            bp = ax.boxplot(per_kernel_data, 
                            positions=bplot_positions, 
                            widths=box_width, 
                            patch_artist=True, 
                            showmeans=True, 
                            showfliers=False,
                            meanprops={"marker": "^",        
                                    "markerfacecolor": "red",   
                                    "markeredgecolor": "black",
                                    "markersize": 4},
                            medianprops={"color": "black"}) 
                                    # "markersize": 8},)
            per_kernel_colors_fin = per_kernel_colors[per_kernel_idx % len(per_kernel_colors)]
            for box in bp['boxes']:
                box.set_facecolor(per_kernel_colors_fin)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.axhline(y=1, color='black', linestyle='--', linewidth=1)
        #adjust x-ticks
        cent_offset = (num_of_kernels - 1) * box_width/2
        xtick_positions = np.arange(num_of_A_keys) * per_kernel_box_spacing + cent_offset
        ax.set_xticks(xtick_positions)
        if model_name == '1D_diffu_FEM':
            HSIC_U_keys_list = [f"$\\xi_{getSingletonIndexAsInt(x)+1}$" for x in HSIC_U_keys_list]
        elif model_name == 'cdr':
            HSIC_U_keys_list = ['$A$', '$E$', '$T_i$', '$T_0$', '$\\phi$']
        else:
            HSIC_U_keys_list = [f"$U_{getSingletonIndexAsInt(x)+1}$" for x in HSIC_U_keys_list]
        
        ax.tick_params(axis='x', labelsize=base_fontsize)
        ax.tick_params(axis='y', labelsize=base_fontsize-int(grid_toggle))
        ax.set_xticklabels(HSIC_U_keys_list)
        if grid_toggle:
            ax.grid(True)
            ax.set_yticks(np.arange(0, 1.01, 0.1))
            
        if not no_individual_legends:
            legend_handles = [plt.Line2D([0], 
                                    [0], 
                                    color=kernel_color, 
                                    lw=4, 
                                    label=kernel_names_list[kernel_idx]) 
                                for kernel_idx, kernel_color in enumerate(per_kernel_colors[:num_of_kernels])]

            ax.legend(handles=legend_handles, title="Kernel(s)")
        if model_name == '1D_diffu_expl':
            model_name = '1D\;Diffusion\;(expl)'
        elif model_name == '1D_diffu_FEM':
            model_name = '1D\;Diffusion\;(expl)'
        else:
            model_name = model_name.replace('_', '\;')
        if not no_title:
            if estimator_type == 'R2':
                ax.set_title(r'$\hat{S}^{R^2,H_{set}}_i$$\vphantom{X}$' + f"\n$\\bf{{{model_name}}}$ - ${estimator_type}$ $unbiased$ $estimator$" +r"$\vphantom{X}$"+ f"\nm={m}, n={n}, repeated {num_of_trials} times")
            else:
                ax.set_title(r'$\hat{S}^{H_{set}}_i$$\vphantom{X}$' + f"\n$\\bf{{{model_name}}}$ - ${estimator_type}$ $estimator$" +r"$\vphantom{X}$"+ f"\nm={m}, n={n}, repeated {num_of_trials} times")
        ax.set_ylabel('')
        # plt.tight_layout()
        # # plt.savefig("figs/hsic_new_kSob_20_1000_1000_all.png")
        # plt.show()

def _get_fig_name_to_save(model_name: str, 
                        x_domains_str: str, 
                        num_of_trials: int, 
                        estimator_types_list: list, 
                        kernels_list: list, 
                        n_list: list):
    fig_name = ''
    fig_name += f"{model_name}_"
    fig_name += x_domains_str
    fig_name += "_"
    for kernel in kernels_list:
        fig_name += f"{kernel}_"
    fig_name += "_"
    for type in estimator_types_list:
        fig_name += f"{type}_"
    fig_name += "_"
    for n in n_list:
        fig_name += f"{n}_"
    fig_name += "_"
    fig_name += f"{num_of_trials}"
    fig_name += '.pdf'
    return fig_name

def plot_hsic(main_data_list,
                model_name='',
                no_title=False,
                no_individual_legends=False,
                base_fontsize=20,
                save_plot=False,
                save_directory='',
                save_name='',
                old_R2=False,
                per_kernel_colors = plt.cm.tab10.colors,
                flip_row_cols=False,
                grid_toggle=False,
                sup_legend_toggle=True,
                sup_x_label_toggle=True,
                sup_y_label_toggle=True,
                separate_input_boxplots=False,
                withTrend=False,
                trend_type='1/sqrtN',
                multi_setting=False,
                plot_corr_of_var=False,
                verbose=False,
                set_x_in_pows_of_10=True,
                box_plots_toggle=True):
    

    if save_plot and save_name == '':
        estimator_types_list = list(main_data_list[0]['data'].keys())
        kernels_list = list(main_data_list[0]['data'][estimator_types_list[0]].keys())
        n_list = [main_data_list[i]['meta_data']['n'] for i in range(len(main_data_list))]
        x_domains_str = f"{str(main_data_list[0]['meta_data']['x_domains_dict'][model_name].tolist()).replace(' ', '').replace('.', '_')}_"
        num_trials = main_data_list[0]['meta_data']['num_of_trials']   
        save_name = _get_fig_name_to_save(model_name=model_name,
                                        x_domains_str=x_domains_str,
                                        num_of_trials=num_trials,
                                        estimator_types_list=estimator_types_list,
                                        kernels_list=kernels_list,
                                        n_list=n_list)
    save_name = save_name.replace('.pkl', '').replace(',','__').replace('[','<').replace(']','>').replace('-','neg')
    # print(save_name)
    if save_plot:    
        print(f'Saving as:\n{save_name}')

    rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'STIXGeneral'
    cardin_of_n_list = len(main_data_list)
    num_of_estimators = len(main_data_list[0]['data'])

    estimator_type_supxaxis_labels = list(main_data_list[0]['data'].keys())
    first_estimator_key = estimator_type_supxaxis_labels[0]
    kernel_names_list_to_check = list(main_data_list[0]['data'][first_estimator_key].keys())
    num_of_kernels_to_check = len(kernel_names_list_to_check)

    if separate_input_boxplots and num_of_estimators == 1 and num_of_kernels_to_check == 1:
        N_set = np.array([indiv_N_data['meta_data']['n'] for indiv_N_data in main_data_list])
        if withTrend:
            if trend_type == '1/sqrtN':
                y_expected_trend = np.sqrt(1/N_set)
            elif trend_type == '1/N':
                y_expected_trend = np.sqrt(1/N_set)
        kernel_name = kernel_names_list_to_check[0]

        HSIC_U_keys_list = [x for x in main_data_list[0]['data'][first_estimator_key][kernel_name].keys() if directBinStrSum(x)<=1]
        # HSIC_U_keys_list = [x.replace("A","U") for x in data['unbiased'][kernel_names_list[0]].keys()]
        num_of_A_keys = len(HSIC_U_keys_list)
        colors = ["tomato", "royalblue", "seagreen", "goldenrod", "orchid", "sienna", "turquoise", "darkorange", "slategray", "deeppink"][:num_of_A_keys]
        fig, axes = plt.subplots(1, num_of_A_keys, figsize=(20, 8), sharey=True)
        
        # per_N_box_spacing = 2.0
        # box_width = 0.8 / cardin_of_n_list
        
        my_data_list = [[indiv_N_data['data'][first_estimator_key][kernel_name][key_A] for indiv_N_data in main_data_list] for key_A in HSIC_U_keys_list]
        if plot_corr_of_var:
            if verbose:
                groups = [[np.array(g) for g in my_data_list[i]] for i in range(len(my_data_list))]
                vars_  = [np.array([g.var() for g in groups[i]]) for i in range(len(my_data_list))]  
                coefs = [np.polyfit(np.log(N_set), np.log(vars_[i]), 1) for i in range(len(my_data_list))]  # slope, intercept
                print(f'{[coefs[i][0] for i in range(len(coefs))]}')
            betas_list = []
        for ax, vals, color, key_A in zip(axes, my_data_list, colors, HSIC_U_keys_list):
            if plot_corr_of_var:
                vals = np.array(vals)
                curr_means = vals.mean(axis=1)
                vars_sqrt = np.sqrt(vals.var(axis=1))
                cOv = vars_sqrt/curr_means
                # cOv_mean = cOv.mean(axis=1)
                # cOv_std = cOv.std(axis=1)

                vals_ = ((vals-curr_means[:, None])**2).mean(axis=1)
                vals_ = np.array([[val] for val in vals_])
                # vals_ = (np.array(vals)-curr_best_mean)**2

                # means = np.sqrt(vals_.mean(axis=1))
                # stds = vals_.var(axis=1)

                x = np.arange(1, len(curr_means) + 1)
                ax.scatter(x,cOv, color=color)
                scatt = ax.loglog(x,cOv,linestyle='--', color=color)
                
                f1 = 1 / np.sqrt(N_set)
                c_hat_sqrt = (f1 @ cOv) / (f1 @ f1)
                # f2 = 1 / N_set
                # c_hat_1N = (f2 @ cOv) / (f2 @ f2)
                X = np.vstack([np.ones_like(N_set), np.log(N_set)]).T
                alpha, beta = np.linalg.lstsq(X, np.log(cOv), rcond=None)[0]
                c_loglog = np.exp(alpha)       # multiplicative constant
                slope_loglog = beta
                ax.loglog(range(1, len(N_set) + 1), c_loglog * N_set**slope_loglog, '--',color='black', linewidth=2)
                # ax.loglog(range(1, len(N_set) + 1), c_hat_sqrt/np.sqrt(N_set), '-.',color='black')
                # ax.loglog(range(1, len(N_set) + 1), c_hat_1N/N_set, '--',color='brown')]
                if verbose:
                    print(f'{key_A}: '+ r"α" + f"={alpha}, "+ r"β"+f"={beta} | {color}")
                betas_list.append(np.round(beta,2))

            else:
                if box_plots_toggle:
                    box = ax.boxplot(vals, 
                                    patch_artist=True, 
                                    showmeans=True, 
                                    showfliers=False,
                                    meanprops={"marker": "^",        
                                                "markerfacecolor": "red",   
                                                "markeredgecolor": "black"})
                    ax.set_ylim(-0.29,1.29)
                    for patch in box['boxes']:
                        patch.set_facecolor(color)
                        patch.set_edgecolor("black")
                    for whisker in box['whiskers']:
                        whisker.set_color(color)
                    for cap in box['caps']:
                        cap.set_color(color)
                    for median in box['medians']:
                        median.set_color("black")
                else:
                    vals = np.array(vals)
                    means = vals.mean(axis=1)
                    stds = vals.std(axis=1)
                    x = np.arange(1, len(means) + 1)

                    err = ax.errorbar(x, means, yerr=stds,
                                    fmt='o',                  # marker for the mean
                                    color=color,            # marker color
                                    ecolor=color,             # whisker color
                                    elinewidth=2,
                                    capsize=8)
                    
                    for line in err[2]:  # err[2] holds the caps
                        line.set_color(color)

                ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
                ax.axhline(y=1, color='black', linestyle='--', linewidth=1)

            if withTrend:
                if not plot_corr_of_var:
                    
                # ax.loglog(range(1, len(N_set) + 1), y_expected_trend_curr_key, label=r'$N^{-1/2}$', color='black', linestyle='--', linewidth=2)
                    if trend_type == '1/sqrtN':
                        y_expected_trend_curr_key = y_expected_trend + np.mean(vals[-1])-np.sqrt(1/N_set[-1])
                        ax.plot(range(1, len(N_set) + 1), y_expected_trend_curr_key**2, label=r'$N^{-1/2}$', color='black', linestyle='--', linewidth=2)
                    elif trend_type == '1/N':
                        y_expected_trend_curr_key = 2*y_expected_trend + (np.mean(vals[-1]))
                        ax.plot(range(1, len(N_set) + 1), y_expected_trend_curr_key**2, label=r'$N^{-1}$', color='black', linestyle='--', linewidth=2)
                # , $ \\beta $ $ \\approx $ {beta_i}" betas_list
                else:
                    trend_handle = mlines.Line2D([], [], color="black", linestyle="--", linewidth=2, label=f"$\\alpha \\approx {alpha:.2f}, \\; \\beta \\approx {beta:.2f} $")
                    ax.legend(handles=[trend_handle],
                            loc="upper right",
                            # ncol=len(colors) + 1,
                            bbox_to_anchor=(1.0, 1.0),
                            fontsize=(base_fontsize),

                            frameon=False)
            ax.set_xticks(range(1, len(N_set) + 1))  # Ensure x-ticks match the number of groups
            ax.tick_params(axis='x', labelsize=base_fontsize)
            ax.tick_params(axis='y', labelsize=base_fontsize-int(grid_toggle))
            if set_x_in_pows_of_10:
                myXTicks = [rf"$10^{int(np.log10(x))}$" for x in N_set]
            else:
                myXTicks = [f"{my_N}" for my_N in N_set]
                # for my_N in N_set:
                #     if my_N % 10 == 0:
                #         myXTicks.append(rf"$10^{int(np.log10(my_N))}$")
                #     else:
                #         myXTicks.append(my_N)
            ax.set_xticklabels(myXTicks)

            if grid_toggle:
                ax.grid(True, axis='y')
                ax.set_yticks(np.arange(0, 1.01, 0.1))
                
            
        
                

        fig.supxlabel(r"Number of Samples, $N$", fontsize=(base_fontsize+2), y=0.02)
        # 
        if plot_corr_of_var:
            fig.supylabel(r"$\widehat{c.o.v.}\sqrt{\hat{var}(\hat{S}^{HSIC}_{A})}/\hat{E}[\hat{S}^{HSIC}_{A}]$", fontsize=(base_fontsize+2))
        else:
            fig.supylabel(r"$\hat{S}^{HSIC}_{A}$", fontsize=(base_fontsize+2))
        if model_name == '1D_diffu_FEM':
            HSIC_U_keys_list = [f"$\\xi_{getSingletonIndexAsInt(x)+1}$" for x in HSIC_U_keys_list]
        elif model_name == 'cdr':
            HSIC_U_keys_list = ['$A$', '$E$', '$T_i$', '$T_0$', '$\\phi$']
        else:
            HSIC_U_keys_list = [f"$U_{getSingletonIndexAsInt(x)+1}$" for x in HSIC_U_keys_list]
        
        box_handles = [mpatches.Patch(color=c_i, label=label_i) for c_i, label_i in zip(colors, HSIC_U_keys_list)]
        if withTrend:
            if plot_corr_of_var:
                trend_handle = mlines.Line2D([], [], color="black", linestyle="--", linewidth=2, label=r"$e^{\alpha} N^{\beta}$")
            else:
                if trend_type == '1/sqrtN':
                    trend_handle = mlines.Line2D([], [], color="black", linestyle="--", linewidth=2, label=r"$N^{-1/2}$")
                elif trend_type == '1/N':
                    trend_handle = mlines.Line2D([], [], color="black", linestyle="--", linewidth=2, label=r"$N^{-1}$")
        
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
        fig.tight_layout(rect=[0.01, 0, 1, 0.97]) 


    else:
        if flip_row_cols:
            fig, axes = plt.subplots(num_of_estimators, cardin_of_n_list, figsize=(4*cardin_of_n_list, 4*num_of_estimators), sharex=True, sharey=True, squeeze=False)
        else:
            fig, axes = plt.subplots(cardin_of_n_list, num_of_estimators, figsize=(4*num_of_estimators, 4*cardin_of_n_list), sharex=True, sharey=True, squeeze=False)

        # if cardin_of_n_list == 1:
        #     axes = [axes]
        # if num_of_estimators == 1:
        #         axes = [[ax] for ax in axes]

        # set sup-x- and sup-y- axes labels
        for estimator_index in range(len(estimator_type_supxaxis_labels)):
            estimator_type_supxaxis_labels[estimator_index] = estimator_type_supxaxis_labels[estimator_index].capitalize()
            if multi_setting:
                if 'R2' in estimator_type_supxaxis_labels[estimator_index]:
                    estimator_type_supxaxis_labels[estimator_index] = '$R^2$-HSIC (Unbiased)'
                else:
                    if estimator_type_supxaxis_labels[estimator_index] == 'Fellman':
                        estimator_type_supxaxis_labels[estimator_index] += 'n'
                    estimator_type_supxaxis_labels[estimator_index] = f'$\\hat{{S}}^{{HSIC}}_A$ ({estimator_type_supxaxis_labels[estimator_index]})'
        num_of_sample_supyaxis_labels = [f"n = m = {data['meta_data']['n']}" for data in main_data_list]
        
        if flip_row_cols:
            # sup-y-axis-labels
            if sup_y_label_toggle:
                for ax, col in zip(axes[0], num_of_sample_supyaxis_labels):
                    ax.annotate(col, xy=(0.5, 1.02), xycoords='axes fraction',
                                ha='center', va='bottom', fontsize=base_fontsize+2)
            # sup-x-axis-labels
            if sup_x_label_toggle:
                for ax, row in zip(axes[:, 0], estimator_type_supxaxis_labels):
                    ax.annotate(row, xy=(-0.16, 0.5), xycoords='axes fraction',
                                ha='right', va='center', fontsize=base_fontsize+2, rotation=90)
        else:
            # sup-x-axis-labels
            if sup_x_label_toggle:          
                for ax, col in zip(axes[0], estimator_type_supxaxis_labels):
                    ax.annotate(col, xy=(0.5, 1.02), xycoords='axes fraction',
                                ha='center', va='bottom', fontsize=base_fontsize+2)
            # sup-y-axis-labels
            if sup_y_label_toggle:
                for ax, row in zip(axes[:, 0], num_of_sample_supyaxis_labels):
                    ax.annotate(row, xy=(-0.16, 0.5), xycoords='axes fraction',
                                ha='right', va='center', fontsize=base_fontsize+2, rotation=90)

        # set legend for different kernels
        if sup_legend_toggle:
            kernels_names_list_for_legend = [f"{kernel.capitalize()} Kernel" for kernel in main_data_list[0]['data'][first_estimator_key].keys()]
            legend_handles = [plt.Line2D([0], 
                                        [0], 
                                        color=kernel_color, 
                                        lw=4, 
                                        label=kernels_names_list_for_legend[kernel_idx]) 
                                    for kernel_idx, kernel_color in enumerate(per_kernel_colors[:len(kernels_names_list_for_legend)])]

            fig.legend(handles=legend_handles,
                        loc="lower center",
                        ncol=len(kernels_names_list_for_legend),
                        bbox_to_anchor=(0.5, 0.985),
                        fontsize=(base_fontsize+1),
                        frameon=False,) 

        for idx_rows, data_dict in enumerate(main_data_list):
            data = data_dict['data']
            estimator_types_list = list(data.keys())
            num_of_estimators = len(estimator_types_list)
            meta_data = data_dict['meta_data']
            n = meta_data['n']
            m = meta_data['m']
            num_of_trials = meta_data['num_of_trials']    

            for idx_cols, estimator_type in enumerate(estimator_types_list):

                kernel_names_list = list(data[estimator_type].keys())
                # kernel_names_list = [kernel for kernel in kernel_names_list if kernel not in ['matern32', 'matern52', 'gaussian']]
                num_of_kernels = len(kernel_names_list)

                # exclude_length = len(list(data['unbiased'][kernel_names_list[0]].keys())[0])
                HSIC_U_keys_list = [x for x in data[estimator_type][kernel_names_list[0]].keys() if directBinStrSum(x)<=1]
                # HSIC_U_keys_list = [x.replace("A","U") for x in data['unbiased'][kernel_names_list[0]].keys()]
                num_of_A_keys = len(HSIC_U_keys_list)
                
                # my_data_list_unbiased = [[data['unbiased'][kernel][key_A] for key_A in HSIC_U_keys_list] for kernel in kernel_names_list if kernel not in ['gaussian', 'matern32', 'matern52']]
                if estimator_type == 'R2':
                    my_data_list = [np.sqrt(np.maximum([data[estimator_type][kernel][key_A] for key_A in HSIC_U_keys_list],0)).tolist() for kernel in kernel_names_list]
                else:
                    my_data_list = [[data[estimator_type][kernel][key_A] for key_A in HSIC_U_keys_list] for kernel in kernel_names_list]

                if flip_row_cols:
                    idx_cols_tmp = idx_rows
                    idx_rows_tmp = idx_cols
                else:
                    idx_cols_tmp = idx_cols
                    idx_rows_tmp = idx_rows
                _plot_this_(ax=axes[idx_rows_tmp,idx_cols_tmp],
                        my_data_list=my_data_list,
                        num_of_kernels = num_of_kernels,
                        num_of_A_keys = num_of_A_keys,
                        per_kernel_colors = per_kernel_colors,
                        HSIC_U_keys_list = HSIC_U_keys_list,
                        kernel_names_list = kernel_names_list,
                        model_name = model_name,
                        num_of_trials = num_of_trials,
                        n = n,
                        m = m,
                        estimator_type=estimator_type,
                        no_title=no_title,
                        no_individual_legends=no_individual_legends,
                        base_fontsize=base_fontsize,
                        grid_toggle=grid_toggle)
                
        plt.tight_layout()

    if save_plot:
        if not path.exists(save_directory):
            makedirs(save_directory)
        if plot_corr_of_var:
            save_name += '_fitted_cov'
        if '.pdf' not in save_name[-4:]:
            save_name += '.pdf'
        plt.savefig(f"{save_directory}/{save_name}", dpi=900, bbox_inches='tight')
        
    plt.show()


def plot_hsic_2x2(data_grid,
                model_name='',
                save_plot=False,
                save_directory='',
                save_name='',
                per_kernel_colors=plt.cm.tab10.colors,
                row_labels=None,
                col_labels=None,
                x_domain=None):
    rcParams['text.usetex'] = True
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    if x_domain is not None:
        formatted_x_dom = ', '.join([f"{x:.2f}" for x in x_domain])
        model_name_txt = '1D\;Diffusion'
        fig.suptitle(f"\n$\\bf{{{model_name_txt}}}$" + f"\nx_domain = [{formatted_x_dom}]", fontsize=14, y=1.02)
    for i in range(2):
        for j in range(2):
            ax = axes[i][j]
            data_dict = data_grid[i][j]

            if data_dict is None:
                ax.set_title(f"Pending...", fontsize=12)
                ax.axis('off')
                continue

            data = data_dict['data']
            meta_data = data_dict['meta_data']
            n = meta_data['n']
            m = meta_data['m']
            num_of_trials = meta_data['num_of_trials']
            meshInterval = meta_data['model_params_dict']['1D_diffu_FEM']['meshInterval']
            estimator_types_list = list(data.keys())
            
            for idx_col, estimator_type in enumerate(estimator_types_list):
                kernel_names_list = list(data[estimator_type].keys())
                HSIC_U_keys_list = [x for x in data[estimator_type][kernel_names_list[0]].keys() if directBinStrSum(x) <= 1]
                num_of_A_keys = len(HSIC_U_keys_list)

                if estimator_type == 'R2':
                    my_data_list = [np.sqrt(np.maximum([data[estimator_type][kernel][key_A] for key_A in HSIC_U_keys_list], 0)).tolist() for kernel in kernel_names_list]
                else:
                    my_data_list = [[data[estimator_type][kernel][key_A] for key_A in HSIC_U_keys_list] for kernel in kernel_names_list]

                _plot_this_(
                    ax=ax,
                    my_data_list=my_data_list,
                    num_of_kernels=len(kernel_names_list),
                    num_of_A_keys=num_of_A_keys,
                    per_kernel_colors=per_kernel_colors,
                    HSIC_U_keys_list=HSIC_U_keys_list,
                    kernel_names_list=kernel_names_list,
                    model_name=model_name,
                    num_of_trials=num_of_trials,
                    n=n,
                    m=m,
                    estimator_type=estimator_type
                )

            # Title with axis info
            title = f"n={n}, h={meshInterval}"
            ax.set_title(title, fontsize=12)

    plt.tight_layout()
    if save_plot:
        if not path.exists(save_directory):
            makedirs(save_directory)
        plt.savefig(f"{save_directory}/{save_name}", dpi=900, bbox_inches='tight')
    plt.show()