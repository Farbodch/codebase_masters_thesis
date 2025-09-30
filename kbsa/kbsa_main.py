import pickle
import numpy as np
from os import path, makedirs
from utils.other_utils import flipStr, getIndexSuperset, directBinStrSum, getIndexSubsets
from kbsa.utils_kbsa import get_HSIC
import time

def get_kbsa_datafile_name(models_list, 
                        x_domains_dict, 
                        kernel_types_list, 
                        hsic_types_dict, 
                        n_list, 
                        num_of_trials,
                        model_params_dict,
                        meshInterval=0,                      
                        auto_mean=False):
    file_name = ''
    for model in models_list:
        file_name += f"{model}_"
        if 'meshInterval' in model_params_dict[model].keys():
        # if 'FEM' in model:
            meshInterval = model_params_dict[model]['meshInterval']
            file_name += f"h_{meshInterval}_"
        file_name += f"{str(x_domains_dict[model].tolist()).replace(' ', '').replace('.', '_')}_"
    file_name += "_"
    for kernel in kernel_types_list:
        file_name += f"{kernel}_"
    file_name += "_"
    for type, val in hsic_types_dict.items():
        if val:
            file_name += f"{type}_"
    file_name += "_"
    for n in n_list:
        file_name += f"{n}_"
    file_name += "_"
    file_name += f"{num_of_trials}"
    if auto_mean:
        file_name += '_auto_mean'
    print(f"file name:\n{file_name}")
    return file_name

def run_kbsa(hsic_types_dict: dict={'unbiased': True, 'biased': False, 'R2': True, 'fellman': False}, 
            kernel_types_list: list=['exponential'],
            models_list: list=['ishigami'],
            model_params_dict: dict={'ishigami': {'g_ineq_c': 3, 'a': 7, 'b': 0.1}},
            x_domains_dict: dict={'ishigami': [[-1, 1]]},
            sample_x_from_mesh: bool=False,
            u_domains_dict: dict={'ishigami': [[-np.pi, np.pi], [-np.pi, np.pi]]},
            u_distributions_dict: dict={},
            num_of_trials: int=1,
            n_list: list=[100],
            m_same_as_n = True,
            m: int=0,
            save_data: bool=False,
            save_directory: str='',
            file_name: str='',
            with_gpu: bool=False,
            which_cdr_output: str='temp',
            higher_order: bool=False, #set to True if interested in calculating higher-order HSIC indices other than main effects
            verbose_inner_inner: bool=False,
            verbose_inner: bool=False,
            verbose_outer: bool=True,) -> list:
    
    
    # hsic_types_dict['unbiased'] = True # setting 'unbiased' to always be true, to calculate and return the 'unbiased' estimation by default
    get_biased = False
    get_R2 = False
    get_fellman = False
    get_unbiased = True
    if 'unbiased' in hsic_types_dict.keys():
        get_unbiased = hsic_types_dict['unbiased']
    if 'biased' in hsic_types_dict.keys():
        get_biased = hsic_types_dict['biased']
    if 'fellman' in hsic_types_dict.keys():
        get_fellman = hsic_types_dict['fellman']
    if 'R2' in hsic_types_dict.keys():
        get_R2 = hsic_types_dict['R2']
        
    main_data_list = {}
    for model in models_list:
        if verbose_outer and len(models_list)>1:
            print(f'Starting for model: {model}.')
        main_data_list[model] = []
        for n_idx in range(len(n_list)):
            if verbose_outer and len(n_list)>1:
                print(f'Starting for  N = {n_list[n_idx]}.')
            data_dict = {key: {} for key in hsic_types_dict.keys() if hsic_types_dict[key] is True}
            n = n_list[n_idx]
            if m_same_as_n:
                m = n
            u_tot_dim = len(u_domains_dict[model])
            u_indexSuperset_oneHot = [flipStr(i) for i in sorted(getIndexSuperset(u_tot_dim, higher_order=higher_order), key=lambda x: directBinStrSum(x))]
            
            for input_kernel in kernel_types_list:
                if verbose_outer and len(kernel_types_list)>1:
                    print(f'Starting for  Kernel: {input_kernel}.')
                HSIC_arrs_dict = {key: {A: [] for A in u_indexSuperset_oneHot} for key in hsic_types_dict.keys() if hsic_types_dict[key] is True}
                for trial in range(num_of_trials):
                    t0 = time.time()
                    if verbose_outer:
                        print(f'trial #{trial+1}/{num_of_trials} calculate started...')
                    HSIC_dicts = get_HSIC(
                                        x_domain=x_domains_dict[model],
                                        sample_x_from_mesh=sample_x_from_mesh,
                                        u_domain=u_domains_dict[model], 
                                        n=n,
                                        m=m,
                                        kernel_U_type=input_kernel, 
                                        which_gamma_fen=model,
                                        gamma_fen_params=model_params_dict[model],
                                        get_unbiased=get_unbiased,
                                        get_biased=get_biased,
                                        get_R2=get_R2,
                                        get_fellman=get_fellman,
                                        with_gpu=with_gpu,
                                        which_cdr_output=which_cdr_output,
                                        verbose=verbose_inner,
                                        verbose_inner=verbose_inner_inner)
                                        # which_cdr_input_random={'A': False, 'E': False, 'T_i': True, 'T_0': True, 'phi': True})
                    t1 = time.time()
                    if verbose_outer:
                        print(f'trial #{trial+1}/{num_of_trials} calculate ended. Duration: {(t1-t0):.3f} (s).\n------')
                                            # gamma_fen_params={'g_ineq_c': my_g_ineq_c, 'a': 7, 'b': 0.1},)
                    HSIC_all_dict = {key: 0 for key in hsic_types_dict.keys() if hsic_types_dict[key] is True}
                    HSIC_arr_currs_dict = {key: {} for key in hsic_types_dict.keys() if hsic_types_dict[key] is True}

                    for key_A in u_indexSuperset_oneHot:
                        A_subset_list = getIndexSubsets(key_A)[1:]
                        # print(f"{key_A}: {A_subset_list}")
                        HSIC_A_dict = {key: 0 for key in hsic_types_dict.keys() if hsic_types_dict[key] is True}

                        if directBinStrSum(key_A) == u_tot_dim: ##!
                            continue ##!
                        for key_B in A_subset_list:
                            cardin_A = directBinStrSum(key_A)
                            cardin_B = directBinStrSum(key_B)

                            if get_unbiased:
                                HSIC_A_dict['unbiased'] += np.power(-1,(cardin_A-cardin_B))*HSIC_dicts['unbiased'][key_B]

                            if get_biased:

                                HSIC_A_dict['biased'] += np.power(-1,(cardin_A-cardin_B))*HSIC_dicts['biased'][key_B]
                            if get_fellman:
                                HSIC_A_dict['fellman'] += np.power(-1,(cardin_A-cardin_B))*HSIC_dicts['fellman'][key_B]



                        if get_unbiased:
                            HSIC_all_dict['unbiased'] = HSIC_dicts['unbiased'][u_indexSuperset_oneHot[-1]]
                            HSIC_arr_currs_dict['unbiased'][key_A] = HSIC_A_dict['unbiased']

                        if get_biased:
                            # HSIC_all_dict['biased'] += HSIC_A_dict['biased'] ##!
                            HSIC_all_dict['biased'] = HSIC_dicts['biased'][u_indexSuperset_oneHot[-1]]
                            HSIC_arr_currs_dict['biased'][key_A] = HSIC_A_dict['biased']
                            # HSIC_all_biased += HSIC_A_biased
                            # HSIC_arr_curr_biased[key_A] = HSIC_A_biased
                        if get_fellman:
                            # HSIC_all_dict['fellman'] += HSIC_A_dict['fellman'] ##!
                            HSIC_all_dict['fellman'] = HSIC_dicts['fellman'][u_indexSuperset_oneHot[-1]]
                            HSIC_arr_currs_dict['fellman'][key_A] = HSIC_A_dict['fellman']
                    
                    for key_A in u_indexSuperset_oneHot:
                        if directBinStrSum(key_A) == u_tot_dim: ##!
                            continue ##!
                        if get_unbiased:
                            HSIC_arrs_dict['unbiased'][key_A].append((HSIC_arr_currs_dict['unbiased'][key_A]/HSIC_all_dict['unbiased']))
                        if get_biased:
                            HSIC_arrs_dict['biased'][key_A].append((HSIC_arr_currs_dict['biased'][key_A]/HSIC_all_dict['biased']))
                        if get_fellman:
                            HSIC_arrs_dict['fellman'][key_A].append((HSIC_arr_currs_dict['fellman'][key_A]/HSIC_all_dict['fellman']))
                        if get_R2 and get_unbiased:
                            HSIC_arrs_dict['R2'][key_A].append((HSIC_dicts['unbiased'][key_A]/(np.sqrt(HSIC_dicts['R2']['HSIC_u_u'][key_A])*np.sqrt(HSIC_dicts['R2']['HSIC_Gam_Gam'][key_A]))))
                        elif get_R2 and get_fellman:
                            HSIC_arrs_dict['R2'][key_A].append((HSIC_dicts['fellman'][key_A]/(np.sqrt(HSIC_dicts['R2']['HSIC_u_u'][key_A])*np.sqrt(HSIC_dicts['R2']['HSIC_Gam_Gam'][key_A]))))
                    if get_unbiased:
                        data_dict['unbiased'][input_kernel] = HSIC_arrs_dict['unbiased']
                    if get_biased:
                        data_dict['biased'][input_kernel] = HSIC_arrs_dict['biased']
                    if get_fellman:
                        data_dict['fellman'][input_kernel] = HSIC_arrs_dict['fellman']
                    if get_R2:
                        data_dict['R2'][input_kernel] = HSIC_arrs_dict['R2']
            main_data_list[model].append({'data': data_dict, 
                                'meta_data': {
                                            'model_params_dict': model_params_dict,
                                            'n': n, 
                                            'm': m, 
                                            'num_of_trials': num_of_trials, 
                                            'x_domains_dict': x_domains_dict, 
                                            'u_domains_dict': u_domains_dict,
                                            'u_distributions_dict': u_distributions_dict}
                                })
    if save_data:
        if not path.exists(save_directory):
            makedirs(save_directory)
        with open(f'{save_directory}/{file_name}.pkl', 'wb') as f:
            pickle.dump(main_data_list, f)
    return main_data_list