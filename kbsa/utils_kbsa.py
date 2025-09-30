import numpy as np
import time
from utils.other_utils import flipStr, getIndexSuperset, directBinStrSum, get_x_boundary_indices_on_1D_FEM_mesh
from kbsa.kernels import get_U_kernel
from kbsa.model_generators import gen_toy_1_gamma, gen_toy_2_gamma, gen_ishigami, gen_linearDecr_1_gamma, gen_linearDecr_2_gamma, gen_1D_diff_explicit, get_1D_diff_FEM, get_CDR, gen_X1CosPiX2SinPiX2, gen_toy_model_vect
from dolfin import Point
from itertools import product
from math import ceil

def gen_fen_dict(x_samples_arr, index_superset, gamma_fen='toy_1', params={'g_ineq_c': 0, 'a': 7, 'b':0.1}):
# def gen_fen_dict(x_samples_arr, index_superset, gamma_fen='toy', params={}):
    if gamma_fen == 'toy_1':
        return {key: [gen_toy_1_gamma(x, params=params) for x in x_samples_arr[idx]] for idx, key in enumerate(index_superset)}
    elif gamma_fen == 'toy_2':
        return {key: [gen_toy_2_gamma(x, params=params) for x in x_samples_arr[idx]] for idx, key in enumerate(index_superset)}
    elif gamma_fen == 'ishigami':
        return {key: [gen_ishigami(x, params=params) for x in x_samples_arr[idx]] for idx, key in enumerate(index_superset)}
    elif gamma_fen == 'linearDecr_1':
        return {key: [gen_linearDecr_1_gamma(x, params=params) for x in x_samples_arr[idx]] for idx, key in enumerate(index_superset)}
    elif gamma_fen == 'linearDecr_2':
        return {key: [gen_linearDecr_2_gamma(x, params=params) for x in x_samples_arr[idx]] for idx, key in enumerate(index_superset)}
    elif gamma_fen == '1D_diffu_expl':
        return {key: [gen_1D_diff_explicit(x, params=params) for x in x_samples_arr[idx]] for idx, key in enumerate(index_superset)}
    elif gamma_fen == 'x1CosPiX2SinPiX2':
        return {key: [gen_X1CosPiX2SinPiX2(x, params=params) for x in x_samples_arr[idx]] for idx, key in enumerate(index_superset)}
    elif gamma_fen == 'toy_model_vect':
        return {key : [gen_toy_model_vect(x, params=params) for x in x_samples_arr[idx]] for idx, key in enumerate(index_superset)}
    
def get_FEM_fen(gamma_fen='1D_diffu_FEM', params={'P': 3, 'mu': 1, 'std': 5, 'meshInterval': 128}):
    if gamma_fen == '1D_diffu_FEM':
        return get_1D_diff_FEM(params=params)
def get_cdr_fen(params={'mesh_2D_dir': '', 'mesh_steps': 0, 't_end': 1, 'num_steps':10, 'g_ineq_c': {'fuel': 1, 'oxygen': 1, 'product': 1, 'temp': 900}}):
    return get_CDR(params=params)

def gen_mass_evaluator(fens):
    """a generator that takes in a list of m generated lambda functions, and generates
    a lambda function that takes in an array of n inputs u \in U \in R^nxd, then 
    makes an mx1 array copy of each u and uses the map function to evaluate 
    all the m generated functions for each u \in U. The final result is  
    an mxn array of the results of the mass-function-evaluations.
    This is done to avoid multiple nested for-loops and taking advantage of the 
    parallelism and vector calculus offered by python's map function.

    Here, we're passing in a list of m functions for each m spatial points x.
    Args:
        fens (list): a list of m lambda functions

    Returns:
        lambda function: a lambda function that takes in an R^nxd input to perform n evaluations of m R^d->R^1 lambda functions (doesn't have to be R^1 necessarily)
                        outputting a list of lists of dim mxn
    """

    def mass_evaluate(U):
        return [f(U) for f in fens] #there are m many fs in fens. 
    return mass_evaluate
    # return lambda U: list(map(lambda fen, u: fen(u), fens, np.full((len(fens), len(U)), U)))

def _get_u_indexSuperset_oneHot(dim_of_U, higher_order=False):
    return [flipStr(i) for i in sorted(getIndexSuperset(dim_of_U, higher_order=higher_order), key=lambda x: directBinStrSum(x))]

def _get_u_samples_arr_to_kernel(dim_of_u_superset, dim_of_U, n):
    return np.random.uniform(low=0, high=1, size=[dim_of_u_superset, n, dim_of_U]) #dim 1 of this arr: cardin of powerset(p) for A, dim2: n, dim3: num of random inputs p

#! NEED TESTCASES FOR ALL OF THIS!!!!
def _get_u_samples_arr(u_samples_arr_to_kernel, u_domain, dim_of_U, dim_of_u_superset, gamma_type=''):
    u_samples_arr = u_samples_arr_to_kernel.copy()
    for dim_idx_2 in range(dim_of_U):
        b = np.max(u_domain[dim_idx_2])
        a = np.min(u_domain[dim_idx_2])
        #convert a unif[0,1] to logUniform[a,b]
        if gamma_type == 'cdr' and dim_idx_2 <= 1:
            for supset_idx in range(dim_of_u_superset):
                u = u_samples_arr[supset_idx, :, dim_idx_2]
                u_samples_arr[supset_idx, :, dim_idx_2] = np.exp(np.log(a) + u*(np.log(b)-np.log(a)))
        #convert unif[0,1] to unif[a,b]
        else:
            for supset_idx in range(dim_of_u_superset):
                u_samples_arr[supset_idx, :, dim_idx_2] = u_samples_arr[supset_idx, :, dim_idx_2]*(b - a) + a
            
    return u_samples_arr


def _get_x_samples_FEM_arr(x_domain, m, dim_of_u_superset, meshInterval=128, minSpatialVal=0, maxSpatialVal=1):
    x_domain_FEM_indices = get_x_boundary_indices_on_1D_FEM_mesh(x_domain=x_domain, 
                                                                meshInterval=meshInterval,
                                                                minSpatialValue=minSpatialVal,
                                                                maxSpatialValue=maxSpatialVal)
    
    x_dom_lows = x_domain_FEM_indices[:, 0]
    x_dom_highs = x_domain_FEM_indices[:, 1]+1 #randint uniformly samples integers from [a,b), as such, must call it with high=b+1 to sample from [a,b] 
    dim_of_X = len(x_domain)
    return np.random.randint(low=x_dom_lows[None, None, :], high=x_dom_highs[None, None, :], size=(dim_of_u_superset, m, dim_of_X))

def _get_lambda_X(x_domain):
    lambda_X = 1
    for dim_idx_2 in range(len(x_domain)):
        b = np.max(x_domain[dim_idx_2])
        a = np.min(x_domain[dim_idx_2])
        lambda_X *= b-a # calculate lebesgue measure for input "deterministic"/"spatial" domain
    return lambda_X

def _get_x_samples_arr(x_domain, m, dim_of_u_superset, sample_x_from_mesh=False, h=8):
    x_dom_lows = x_domain[:, 0]
    x_dom_highs = x_domain[:, 1]
    dim_of_X = len(x_domain)
    if sample_x_from_mesh:
        # build discrete grids for each dimension (h intervals -> h+1 grid points)
        grids = [np.linspace(low, high, h+1) for low, high in zip(x_dom_lows, x_dom_highs)]
        return np.stack([np.random.choice(grid, size=(dim_of_u_superset, m)) for grid in grids], axis=-1)
    else:
        return np.random.uniform(low=x_dom_lows[None, None, :], high=x_dom_highs[None, None, :], size=(dim_of_u_superset, m, dim_of_X))


def _gamma_binary_numpy(result, lambda_X=1.0):
    A = np.asarray(result, dtype=np.float32)
    if A.ndim == 3:
        A = np.squeeze(A, axis=-1)
    n = A.shape[1]
    ones = A.sum(axis=1)
    both_one = A @ A.T
    hamming = ones[:,None] + ones[None,:] - 2*both_one
    return (hamming / n) * lambda_X

def _get_gamma_matrices_dict(n, 
                            x_samples_arr, 
                            u_samples_arr,
                            lambda_X,
                            u_indexSuperset_oneHot,
                            which_gamma_fen,
                            gamma_fen_params,
                            which_cdr_output='temp', 
                            verbose=False):
    
    if 'FEM' in which_gamma_fen:
        gamma_fen = get_FEM_fen(gamma_fen=which_gamma_fen,
                                params=gamma_fen_params)
    elif which_gamma_fen == 'cdr':
        gamma_fen = get_cdr_fen(params=gamma_fen_params)
    else:
        gamma_fens_list_dict = gen_fen_dict(x_samples_arr=x_samples_arr, 
                                            index_superset=u_indexSuperset_oneHot, 
                                            gamma_fen=which_gamma_fen,
                                            params=gamma_fen_params)
        
    cdr_str_to_idx = {'fuel': 0, 'oxygen': 1, 'product': 2, 'temp': 3} 
    gamma_matrices_dict = {}
    
    for A_idx, A_key in enumerate(u_indexSuperset_oneHot):
        t0_A = time.time()
        if 'FEM' in which_gamma_fen:
            results_all = np.array([gamma_fen(u=u) for u in u_samples_arr[A_idx]]) 
            
            index_set_n = np.arange(n)[:, None]
            index_set_m_of_x = x_samples_arr[A_idx].T
            result = results_all[index_set_n, index_set_m_of_x]
        elif which_gamma_fen == 'cdr':

            results_all = np.array([gamma_fen(u=u)[cdr_str_to_idx[which_cdr_output]] for u in u_samples_arr[A_idx]]) 
            result = np.array([[f(Point(x)) for x in x_samples_arr[A_idx]] for f in results_all])
        else:
            result = np.array([[f(u) for f in gamma_fens_list_dict[A_key]] for u in u_samples_arr[A_idx]]) 

        t1_A = time.time()
        
        t0_B = time.time()
        gamma_matrix=_gamma_binary_numpy(result=result,lambda_X=lambda_X)

        t1_B = time.time()
        

        t0_C = time.time()
        curr_sigma_sqrd = np.mean(gamma_matrix)
        t1_C = time.time()
            
            # print(curr_sigma_sqrd)
        t0_D = time.time()
        gamma_matrices_dict[A_key] = -1*gamma_matrix/(2*curr_sigma_sqrd)
        t1_D = time.time()

        if verbose:
            print(f'\tA={A_key}...')
            print(f'\ti) Results calculated for {A_key}. Duration: {t1_A-t0_A:.3f} (s).')
            print(f'\tii) Gamma_matrix calculated for {A_key}. Duration: {t1_B-t0_B:.3f} (s).')
            print(f'\tiii) curr_sigma_sqrd calculated for {A_key}. Duration: {t1_C-t0_C:.3f} (s).')
            print(f'\tiv) gamma_matrices_dict adjusted for {A_key}. Duration: {t1_D-t0_D:.3f} (s).')
            print('\t----------------')
    return gamma_matrices_dict


def get_HSIC(u_domain: list,
            x_domain: list, 
            n: int,
            m: int,
            kernel_U_type: str = 'sobolev',
            which_gamma_fen: str = 'toy',
            gamma_fen_params: dict = {'g_ineq_c': 0, 'a': 7, 'b':0.1},
            sample_x_from_mesh=False,
            get_unbiased: bool = True,
            get_biased: bool = False,
            get_R2: bool = True,
            get_fellman: bool = False,
            with_gpu=False,
            which_cdr_output='temp',
            higher_order=False,
            verbose=False,
            verbose_inner=False) -> dict:
    total_time_in_get_HSIC = 0
    t0_begin_get_HSIC = time.time()

    dim_of_U = len(u_domain) #p in algorithm1

    # includes all possible combination of u indices in one-hot encoding format. 
    # E.g. for dim_of_U = 3, u_1 == "100", u_1u_3 == "101"
    u_indexSuperset_oneHot = _get_u_indexSuperset_oneHot(dim_of_U=dim_of_U, higher_order=higher_order)
    dim_of_u_superset = len(u_indexSuperset_oneHot)

    #generating n samples for each of these sets
    u_samples_arr_to_kernel = _get_u_samples_arr_to_kernel(dim_of_u_superset=dim_of_u_superset, dim_of_U=dim_of_U, n=n)
    u_samples_arr = _get_u_samples_arr(u_samples_arr_to_kernel=u_samples_arr_to_kernel, u_domain=u_domain, dim_of_U=dim_of_U,dim_of_u_superset=dim_of_u_superset, gamma_type=which_gamma_fen)
    lambda_X = _get_lambda_X(x_domain=x_domain)

    if 'FEM' in which_gamma_fen:
        x_samples_arr = _get_x_samples_FEM_arr(x_domain=x_domain,
                                            m=m, 
                                            dim_of_u_superset=dim_of_u_superset,
                                            meshInterval=gamma_fen_params['meshInterval'],
                                            minSpatialVal=gamma_fen_params['minSpatialVal'],
                                            maxSpatialVal=gamma_fen_params['maxSpatialVal'])
    else:
        if sample_x_from_mesh:
            h = gamma_fen_params['meshInterval']
            x_samples_arr = _get_x_samples_arr(x_domain=x_domain, m=m, dim_of_u_superset=dim_of_u_superset, sample_x_from_mesh=sample_x_from_mesh, h=h)
        else:
            x_samples_arr = _get_x_samples_arr(x_domain=x_domain, m=m, dim_of_u_superset=dim_of_u_superset)


    t1_begin_get_HSIC = time.time()
    if verbose:
        print(f'Prliminary get_HSIC ended. Duration: {t1_begin_get_HSIC-t0_begin_get_HSIC:.3f} (s).')
    total_time_in_get_HSIC += (t1_begin_get_HSIC-t0_begin_get_HSIC)
    t0_gamm = time.time()
    if verbose:
        print('get_gamma_mat started...')
    gamma_matrices_dict = _get_gamma_matrices_dict(n=n, 
                                                x_samples_arr=x_samples_arr,
                                                u_samples_arr=u_samples_arr,
                                                lambda_X=lambda_X,
                                                u_indexSuperset_oneHot=u_indexSuperset_oneHot,
                                                which_gamma_fen=which_gamma_fen, 
                                                gamma_fen_params=gamma_fen_params,
                                                which_cdr_output=which_cdr_output,
                                                verbose=verbose_inner)

    t1_gamm = time.time()
    if verbose:
        print(f'get-gamma_mat ended. Duration: {t1_gamm-t0_gamm:.3f} (s).')
    total_time_in_get_HSIC += (t1_gamm-t0_gamm)

    t0_populate_kernel = time.time()
    K_sob_matrices_dict = {key: np.zeros((n,n)) for key in u_indexSuperset_oneHot}
    t0_get_thetas = time.time()
    thetas = {key: np.std(u_samples_arr[idx], 0) for idx, key in enumerate(u_indexSuperset_oneHot)}
    t1_get_thetas = time.time()
    if verbose:
        print(f'Thetas dict calculated. Duration: {t1_get_thetas-t0_get_thetas:.3f} (s).')
    total_time_in_get_HSIC += (t1_get_thetas-t0_get_thetas)

    _build_K_sob_vectorized(
        K_sob_matrices_dict=K_sob_matrices_dict,
        u_indexSuperset=u_indexSuperset_oneHot,
        u_samples_arr_to_kernel=u_samples_arr_to_kernel,
        kernel_U_type=kernel_U_type,
        chunk_S=None
    )

    for key, K in K_sob_matrices_dict.items():
        assert K.shape == (n, n)
        assert np.allclose(K, K.T, atol=1e-12)

    #LOOP 1
    for j in range(1, n):
        #LOOP 2
        for i in range(j):
            for supset_A_idx, idx_A_str in enumerate(u_indexSuperset_oneHot):
                cardinOfInputU = directBinStrSum(idx_A_str)
                
                if cardinOfInputU == 1: # e.g. 001, 010, 100           
                    u_i = u_samples_arr_to_kernel[supset_A_idx, i, supset_A_idx]
                    u_j = u_samples_arr_to_kernel[supset_A_idx, j, supset_A_idx]
                    K_sob_matrices_dict[idx_A_str][i,j] = get_U_kernel(params={'u': [u_i, u_j], 'theta': thetas[idx_A_str][supset_A_idx]}, kernel_U_type=kernel_U_type)
                
                if cardinOfInputU > 1: # e.g. 011, 101, 110, 111
                    U_kernel_i = u_samples_arr_to_kernel[supset_A_idx, i, :]
                    U_kernel_j = u_samples_arr_to_kernel[supset_A_idx, j, :]
                    counter = 0
                    for singularOf_B_idx, singularOf_B_str in enumerate(idx_A_str):
                        if singularOf_B_str == '0':
                            continue    
                        u_i = U_kernel_i[singularOf_B_idx]
                        u_j = U_kernel_j[singularOf_B_idx]

                        if counter == 0:
                            K_sob_matrices_dict[idx_A_str][i,j] = get_U_kernel(params={'u': [u_i, u_j], 'theta': thetas[idx_A_str][singularOf_B_idx]}, kernel_U_type=kernel_U_type)
                        else:
                            K_sob_matrices_dict[idx_A_str][i,j] *= (get_U_kernel(params={'u': [u_i, u_j], 'theta': thetas[idx_A_str][singularOf_B_idx]}, kernel_U_type=kernel_U_type))
                        counter += 1
                #both matrix K_U and K_Gamma are symmetric: 
                K_sob_matrices_dict[idx_A_str][j,i] = K_sob_matrices_dict[idx_A_str][i,j].copy()
    #     end LOOP 2
    # end LOOP 1
    #----------------

    #----------------#----------------
    t1_populate_kernel = time.time()
    if verbose:
        print(f'Kernel matrix K_sob_matrices_dict calculated. Duration: {t1_populate_kernel-t0_populate_kernel:.3f} (s).')
    total_time_in_get_HSIC += (t1_populate_kernel-t0_populate_kernel)
    #----------------#----------------
    t0_mat_muls = time.time()
    #----------------#----------------
    if get_biased:
        H = np.eye(n)-(1/n)
        const_0_biased = 1/((n-1)*(n-1))
        HSIC_dict_biased = {}
    if get_unbiased:
        const_0_unbiased = (1/(n*(n-3)))
        const_1_unbiased = (1/((n-1)*(n-2)))
        const_2_unbiased = (2/(n-2))
        
        HSIC_dict_unbiased = {}
    for _, idx_A_str in enumerate(u_indexSuperset_oneHot):
        K_gam = np.exp(gamma_matrices_dict[idx_A_str])
        K_u = K_sob_matrices_dict[idx_A_str]
        np.fill_diagonal(K_u, 0)
        np.fill_diagonal(K_gam, 0)

        
        if get_unbiased:
            KU_Kgam = K_u @ K_gam
            tr_KU_Kgam = np.trace(KU_Kgam)
            oneT_KU_one = np.sum(K_u)
            oneT_Kgam_one = np.sum(K_gam)
            oneT_KU_Kgam_one = np.sum(KU_Kgam)
            HSIC_dict_unbiased[idx_A_str] = const_0_unbiased*(tr_KU_Kgam + const_1_unbiased*(oneT_KU_one*oneT_Kgam_one) - const_2_unbiased*oneT_KU_Kgam_one)
        if get_biased:           
            HSIC_dict_biased[idx_A_str] = const_0_biased * np.trace(K_u @ H @ K_gam @ H)
    
    if get_R2:
        HSIC_u_u = {}
        for _, idx_A_str in enumerate(u_indexSuperset_oneHot):
            KU = K_sob_matrices_dict[idx_A_str]
            np.fill_diagonal(KU, 0)
            KU_KU = KU @ KU
            tr_KU_KU = np.trace(KU_KU)
            oneT_KU_one = np.sum(KU)
            oneT_KU_KU_one = np.sum(KU_KU)
            HSIC_u_u[idx_A_str] = (1/(n*(n-3)))*(tr_KU_KU + (1/((n-1)*(n-2)))*(oneT_KU_one*oneT_KU_one) - (2/(n-2))*oneT_KU_KU_one)
            if with_gpu:
                HSIC_u_u[idx_A_str] = HSIC_u_u[idx_A_str].cpu()
        HSIC_Gam_Gam = {}
        for _, idx_A_str in enumerate(u_indexSuperset_oneHot):
            K_gam = np.exp(gamma_matrices_dict[idx_A_str])
            np.fill_diagonal(K_gam, 0)
            Kgam_Kgam = K_gam @ K_gam
            tr_Kgam_Kgam = np.trace(Kgam_Kgam)
            oneT_Kgam_one = np.sum(K_gam)
            oneT_Kgam_Kgam_one = np.sum(Kgam_Kgam)
            HSIC_Gam_Gam[idx_A_str] = (1/(n*(n-3)))*(tr_Kgam_Kgam + (1/((n-1)*(n-2)))*(oneT_Kgam_one*oneT_Kgam_one) - (2/(n-2))*oneT_Kgam_Kgam_one)
    #----------------#----------------
    if get_fellman:
        #----------------
        #----------------
        HSIC_dict_fellman = {key: 0 for key in u_indexSuperset_oneHot}
        const_0 = 2/(n*(n-1))
        for _, idx_A_str in enumerate(u_indexSuperset_oneHot):
            for j in range(1,n):
                for i in range(j):
                    k_u_A_i_j = K_sob_matrices_dict[idx_A_str][i,j]
                    k_set_A_i_j = np.exp(gamma_matrices_dict[idx_A_str][i,j])
                    HSIC_dict_fellman[idx_A_str] += ((k_u_A_i_j-1)*k_set_A_i_j)
            HSIC_dict_fellman[idx_A_str] = const_0*HSIC_dict_fellman[idx_A_str]
        #----------------
        #----------------
    t1_mat_muls = time.time()
    if verbose:
        print(f'MatMuls ended. Duration: {t1_mat_muls-t0_mat_muls:.3f} (s).')
    total_time_in_get_HSIC += (t1_mat_muls-t0_mat_muls)
    HSIC_dicts = {}
    t0_assign = time.time()
    #----------------#----------------
    if get_unbiased:
        HSIC_dicts['unbiased'] = HSIC_dict_unbiased
    if get_biased:
        HSIC_dicts['biased'] = HSIC_dict_biased
    if get_R2:
        HSIC_dicts['R2'] = {'HSIC_u_u': HSIC_u_u, 'HSIC_Gam_Gam': HSIC_Gam_Gam}
    #----------------#----------------
    # if get_R2:
    #     HSIC_u_u = hsic_uu_fellman_tiled(u_indexSuperset_oneHot, u_samples_arr_to_kernel, tile=2048, block_dtype=np.float32)
    #     HSIC_Gam_Gam  = hsic_gamgam_fellman_tiled(u_indexSuperset_oneHot, gamma_matrices_dict, tile=2048, block_dtype=np.float32)
    #     HSIC_dicts['R2'] = {'HSIC_u_u': HSIC_u_u, 'HSIC_Gam_Gam': HSIC_Gam_Gam}
    if get_fellman:
        HSIC_dicts['fellman'] = HSIC_dict_fellman
    t1_assign = time.time()
    
    total_time_in_get_HSIC += (t1_assign-t0_assign)
    if verbose:
        print(f'Assigning dicts ended. Duration: {t1_assign-t0_assign:.3f} (s).')
        print(f'Total Duration in get_HSIC(): {total_time_in_get_HSIC:.3f} (s).')
    return HSIC_dicts


def _sobol_pairwise_all(U_Snk):
    Uk = np.transpose(U_Snk, (0, 2, 1))       
    u_i = Uk[..., :, None]                     
    u_j = Uk[..., None, :]                      
    du  = u_i - u_j                           
    K   = 1.0 + (u_i - 0.5) * (u_j - 0.5) + 0.5 * (du**2 - np.abs(du) + (1.0/6.0))
    return K

def _build_K_sob_vectorized(
    K_sob_matrices_dict,
    u_indexSuperset,           
    u_samples_arr_to_kernel,   
    kernel_U_type='sobolev',
    chunk_S=None               
):

    # if kernel_U_type != 'sobolev':
    #     raise ValueError(f"Invalid Kernel: {kernel_U_type}!")


    if not isinstance(u_samples_arr_to_kernel, np.ndarray) or u_samples_arr_to_kernel.ndim != 3:
        raise ValueError("u_samples_arr_to_kernel must be a 3D numpy array of shape (S, n, k).")
    S, n, k = u_samples_arr_to_kernel.shape


    if len(u_indexSuperset) != S:
        raise ValueError(f"len(u_indexSuperset)={len(u_indexSuperset)} but u_samples_arr_to_kernel.shape[0]={S}.")
    for key in u_indexSuperset:
        if len(key) != k:
            raise ValueError(f"Bitstring '{key}' length {len(key)} != k ({k}).")
        if "1" not in key:
            raise ValueError("u_indexSuperset must not include the empty set: found bitstring with no '1'.")


    missing = [key for key in u_indexSuperset if key not in K_sob_matrices_dict]
    if missing:
        raise ValueError(f"K_sob_matrices_dict is missing keys: {missing}")
    for key, Kmat in K_sob_matrices_dict.items():
        if Kmat.shape != (n, n):
            raise ValueError(f"K_sob_matrices_dict[{key}] has shape {Kmat.shape}, expected {(n, n)}.")


    mask_Sk = np.array([[c == '1' for c in bitstr] for bitstr in u_indexSuperset], dtype=bool)

    if chunk_S is None or chunk_S <= 0:
        chunk_S = S

    start = 0
    while start < S:
        end = min(start + chunk_S, S)

        U_chunk = u_samples_arr_to_kernel[start:end]    
        M_chunk = mask_Sk[start:end]                     


        K_Sknn = _sobol_pairwise_all(U_chunk)


        K_masked = np.where(M_chunk[:, :, None, None], K_Sknn, 1.0) 

        K_chunk = np.prod(K_masked, axis=1)


        for local_idx, global_idx in enumerate(range(start, end)):
            key = u_indexSuperset[global_idx]
            K_sob_matrices_dict[key][:] = K_chunk[local_idx]

        start = end