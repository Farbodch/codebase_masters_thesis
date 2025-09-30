import numpy as np
from utils.numeric_models import model
from utils.other_utils import getIndexSuperset, getIndexComplement, directBinStrSum, get_x_boundary_indices_on_1D_FEM_mesh

def _getSobolIdx_clos(model, 
                      indexSet, 
                      N, 
                      scalarDiffuIdx=0, 
                      x_interval_of_interest=None,
                      COMMENT=False):

    """Calculate Sobol index for a given subset and sample count."""
    inputSize = len(indexSet)
    A = indexSet
    A_c = getIndexComplement(A)

    y_arr, y_ref_arr, y_tild_arr = [], [], []

    if model.model_type in ['diffusion_1D_both', 'diffusion_1D_explicit']:
        y_arr_atIdx, y_ref_arr_atIdx, y_tild_arr_atIdx = [], [], []

    #possibly depricated code! Used to go use append, but vectorized and use np.mean at axis=0. Double check and delete!
    if model.model_type in ['diffusion_1D', 'diffusion_1D_both']:
        y_ref_mean, y_tild_mean, y_mean  = [], [], []

    if model.model_type == 'diffusion_1D_both':
        mesh_resolution = model.meshInterval
        if x_interval_of_interest is not None:
            min_boundary = model.mesh_coords[0]
            max_boundary = model.mesh_coords[mesh_resolution]

            if not model.projectOutputToCG:
                if type(x_interval_of_interest) is list:
                    x_interval_of_interest = np.array(x_interval_of_interest)
                x_boundary_indices = get_x_boundary_indices_on_1D_FEM_mesh(x_domain=x_interval_of_interest, 
                                                                            meshInterval=mesh_resolution,
                                                                            minSpatialValue=min_boundary,
                                                                            maxSpatialValue=max_boundary)
                min_boundary_idx = x_boundary_indices[0]
                max_boundary_idx = x_boundary_indices[1]
            else:
                indicesToEvalProjAt = np.linspace(min_boundary, max_boundary, model.diffuFen_sobol_vect_len)
                
    # if get_indiv_output_variance:
    #     indiv_output_variance = {}     
    for _ in range(N):
        X = model.sampleInput()
        X_1 = model.sampleInput()
        X_1_c = np.copy(X_1)
        X_2_c = model.sampleInput()

        # set indices to zero based on whether X is taking on A or A_c indices
        # by multiplying a given sampled index by 0.
        for idx in range(inputSize):
            X_1_c[idx] = X_1[idx] * int(A_c[inputSize - 1 - idx])
            X_1[idx] = X_1[idx] * int(A[inputSize - 1 - idx])
            X_2_c[idx] = X_2_c[idx] * int(A_c[inputSize - 1 - idx])

        # adding a 0 due to machine precision issues
        X_1 += 0
        X_1_c += 0
        X_2_c += 0

        input_ref = X_1 + X_1_c
        input_tild = X_1 + X_2_c

        if model.model_type in ["ishigami", "ishigami_vect"]:
            y_arr.append(model.ishiFen(X))
            y_ref_arr.append(model.ishiFen(input_ref))
            y_tild_arr.append(model.ishiFen(input_tild))

        elif model.model_type == 'toy_model_x1sqr_plus_x2sqr':
            y_arr.append(model.toy_model_x1sqr_plus_x2sqr(X))
            y_ref_arr.append(model.toy_model_x1sqr_plus_x2sqr(input_ref))
            y_tild_arr.append(model.toy_model_x1sqr_plus_x2sqr(input_tild))

        elif model.model_type == 'toy_model_vect':
            y_arr.append(model.toy_model_vect(X))
            y_ref_arr.append(model.toy_model_vect(input_ref))
            y_tild_arr.append(model.toy_model_vect(input_tild))

        elif model.model_type == 'toy_1_vect':
            y_arr.append(model.toy_1(X))
            y_ref_arr.append(model.toy_1(input_ref))
            y_tild_arr.append(model.toy_1(input_tild))
        
        elif model.model_type == 'toy_2_vect':
            y_arr.append(model.toy_2(X))
            y_ref_arr.append(model.toy_2(input_ref))
            y_tild_arr.append(model.toy_2(input_tild))

        elif model.model_type == 'diffusion_1D_explicit':
            y_arr.append(model.diffuFen_expl(X, scalarDiffuIdx=scalarDiffuIdx))
            y_ref_arr.append(model.diffuFen_expl(input_ref, scalarDiffuIdx=scalarDiffuIdx))
            y_tild_arr.append(model.diffuFen_expl(input_tild, scalarDiffuIdx=scalarDiffuIdx))

        elif "diffusion_1D" in model.model_type:
            if model.projectOutputToCG:
                y_arr_tmp = model.diffuFen(X, indicesToEvalProjAt=indicesToEvalProjAt)
                y_ref_arr_tmp = model.diffuFen(input_ref, indicesToEvalProjAt=indicesToEvalProjAt)
                y_tild_arr_tmp = model.diffuFen(input_tild, indicesToEvalProjAt=indicesToEvalProjAt)
            else:          
                y_arr_tmp = model.diffuFen(X)
                y_ref_arr_tmp = model.diffuFen(input_ref)
                y_tild_arr_tmp = model.diffuFen(input_tild)
            # if model.COMMENT:
            #     print(f"\nA: {A}")
            #     print(X)
            #     print(model.get_difference_bw_alpha_explAndInMesh(X))
                

            if 'scalar' in model.model_type:
                y_arr.append(y_arr_tmp[scalarDiffuIdx])
                y_ref_arr.append(y_ref_arr_tmp[scalarDiffuIdx])
                y_tild_arr.append(y_tild_arr_tmp[scalarDiffuIdx])
            else:
                if x_interval_of_interest is not None and not model.projectOutputToCG:
                    y_arr.append(y_arr_tmp[min_boundary_idx:(max_boundary_idx+1)])
                    y_ref_arr.append(y_ref_arr_tmp[min_boundary_idx:(max_boundary_idx+1)])
                    y_tild_arr.append(y_tild_arr_tmp[min_boundary_idx:(max_boundary_idx+1)])
                else:
                    y_arr.append(y_arr_tmp)
                    y_ref_arr.append(y_ref_arr_tmp)
                    y_tild_arr.append(y_tild_arr_tmp)

                if 'both' in model.model_type:
                    y_arr_atIdx.append(y_arr_tmp[scalarDiffuIdx])
                    y_ref_arr_atIdx.append(y_ref_arr_tmp[scalarDiffuIdx])
                    y_tild_arr_atIdx.append(y_tild_arr_tmp[scalarDiffuIdx])

        # elif model.model_type == "diffusion_1D_scalar":
        #     y_arr.append(model.diffuFen(X, scalarDiffuIdx))
        #     y_ref_arr.append(model.diffuFen(input_ref, scalarDiffuIdx))
        #     y_tild_arr.append(model.diffuFen(input_tild, scalarDiffuIdx))
        # elif model.model_type in ['diffusion_1D', 'bo':
        #     y_arr.append(model.diffuFen(X))
        #     y_ref_arr.append(model.diffuFen(input_ref))
        #     y_tild_arr.append(model.diffuFen(input_tild))

    if model.model_type in ['diffusion_1D', 'ishigami_vect', 'toy_model_vect', 'toy_1_vect', 'toy_2_vect']:
        y_mean = np.mean(np.array(y_arr), axis=0)
        y_ref_mean = np.mean(np.array(y_ref_arr), axis=0)
        y_tild_mean = np.mean(np.array(y_ref_arr), axis=0)
    elif model.model_type == 'diffusion_1D_both':
        y_mean = np.mean(np.array(y_arr), axis=0)
        y_ref_mean = np.mean(np.array(y_ref_arr), axis=0)
        y_tild_mean = np.mean(np.array(y_ref_arr), axis=0)

        y_mean_atIdx = np.mean(y_arr_atIdx)
        y_ref_mean_atIdx = np.mean(y_ref_arr_atIdx)
        y_tild_mean_atIdx = np.mean(y_tild_arr_atIdx)

        # for i in range(N):
        #     y_mean.append(np.mean(y_arr[i]))
        #     y_ref_mean.append(np.mean(y_ref_arr[i]))
        #     y_tild_mean.append(np.mean(y_tild_arr[i]))
    else:
        y_mean = np.mean(y_arr)
        y_ref_mean = np.mean(y_ref_arr)
        y_tild_mean = np.mean(y_tild_arr)

    if model.model_type in ['diffusion_1D', 'ishigami_vect', 'toy_model_vect', 'toy_1_vect', 'toy_2_vect']:
        T_hat_A = 0
        T_hat = 0
        k = len(y_arr[0])
        for i in range(N):
            for j in range(k):
                T_hat_A += (y_ref_arr[i][j]-y_ref_mean[j])*(y_tild_arr[i][j]-y_tild_mean[j])
                T_hat += (y_arr[i][j]-y_mean[j])**2
        return T_hat_A/T_hat
    elif model.model_type == 'diffusion_1D_both':
        T_hat_A = 0
        T_hat = 0
        k = len(y_arr[0])
        for i in range(N):
            for j in range(k):
                T_hat_A += (y_ref_arr[i][j]-y_ref_mean[j])*(y_tild_arr[i][j]-y_tild_mean[j])
                T_hat += (y_arr[i][j]-y_mean[j])**2

        V_hat_A = np.sum((np.array(y_ref_arr_atIdx) - y_ref_mean_atIdx) * (np.array(y_tild_arr_atIdx) - y_tild_mean_atIdx)) / (N - 1)
        V_hat = np.sum((np.array(y_arr_atIdx) - y_mean_atIdx) ** 2) / (N - 1)
        
        s_clos_aggr = T_hat_A / T_hat
        s_clos = V_hat_A / V_hat

        return (s_clos, s_clos_aggr)
    
    else:
        V_hat_A = np.sum((np.array(y_ref_arr) - y_ref_mean) * (np.array(y_tild_arr) - y_tild_mean)) / (N - 1)
        V_hat = np.sum((np.array(y_arr) - y_mean) ** 2) / (N - 1)
        # if get_indiv_output_variance:
        return (V_hat_A, V_hat)
        # else:
        #     return V_hat_A / V_hat

def _getMainSobolsFromClosed(sobolDict_clos: dict) -> dict:
    sobolDict_main={}
    def getChild(idxA):
        sum = 0
        if directBinStrSum(idxA) == 1:
            return sobolDict_clos[idxA]
        for j in range(len(idxA)):       
            currChild = idxA

            if currChild[j] == '1':
                currChild = list(currChild)
                currChild[j] = '0'
                currChild = "".join(currChild)

                sum -= getChild(currChild)
        sum += sobolDict_clos[idxA]
        if idxA not in sobolDict_main.keys():
            sobolDict_main[idxA] = sum

        return sum
    lenIdx = len(list(sobolDict_clos.keys())[0])
    for idx in sobolDict_clos.keys():
        if directBinStrSum(idx) == 1:
            sobolDict_main[idx] = sobolDict_clos[idx]
    
    A_hat = ''.join('1' for _ in range(lenIdx))
    getChild(A_hat)
    mainAll = 0
    for key in sobolDict_main:
        if key != A_hat:
            mainAll += sobolDict_main[key]
    sobolDict_main[A_hat] = sobolDict_clos[A_hat] - mainAll
    
    return sobolDict_main

def run_sobols(model: model, N_set=None, scalarDiffuIdx=0, itersPerN=1, x_interval_of_interest=None, returnResults=False):
    """
    Run the iterative solving process for calculating Sobol indices.
    
    Parameters:
    - N_set (list of int): List of sample sizes.
    - scalarDiffuIdx (int): Index for diffusion result extraction.
    
    Returns:
    - dict: Sobol indices for each sample size in N_set.
    """
    if model.model_type == 'ishigami':
        scalarDiffuIdx = 0
    if x_interval_of_interest is not None:
        x_interval_of_interest_str = str(x_interval_of_interest).replace(" ", "")
        if x_interval_of_interest_str not in model.numItersPerNAtIdx.keys():
            model.numItersPerNAtIdx[x_interval_of_interest_str] = 0

    if scalarDiffuIdx not in model.numItersPerNAtIdx.keys():
        model.numItersPerNAtIdx[scalarDiffuIdx] = 0
    #for each run, or num of idx, save results in the exprimentDataDict
    #then use exprimentDataDict to get boxplots at each iteration
    myIdxSupset = getIndexSuperset(model.vectSize)
    # model.sobolVals_clos = {}
    # model.sobolVals_clos_aggr = {}
    
    #Case where N_set is being passed in for the first time
    if N_set is not None and len(model.N_set)==0:
        N_set = list(set(N_set))
        N_set.sort()
        model.N_set = N_set

    #Case where model.N_set is maybe getting new values. Using this case is NOT recommended*
    #* this case implies there are more data for older N_set values and less data for 
    #  newly passed in values. This scenario is currently NOT handled in this code, and
    #  and will result in unreliable analyses.
    elif N_set is not None and len(model.N_set)!=0:
        for item in N_set:
            if item not in model.N_set:
                model.N_set.append(item)
        model.N_set.sort()
        N_set = model.N_set

    #Case where model.N_set is empty and nothing was sent in; default
    elif N_set is None and len(model.N_set)==0:
        N_set = [100]
        model.N_set = N_set

    #Case where model.N_set is already set and nothing new was passed in
    elif N_set is None and len(model.N_set)!=0:
        N_set = model.N_set

    else:
        raise ValueError("Error in N_set check: this condition should not be reachable!")

    #-> scalarDiffuIdx maybe can be set at initialization??
    #-> used for plotting purposes only. No functional use as the relevant method is
    # commented out in model.diffuFen()
    if model.model_type in ['diffusion_1D_scalar', 'diffusion_1D_both', 'diffusion_1D_explicit']:
        model.scalarDiffuIdx = scalarDiffuIdx
    for _ in range(itersPerN):
        # model.sobolVals_clos = {}
        # model.sobolVals_clos_aggr = {}
        sobolVals_clos = {}
        sobolVals_clos_aggr = {}
        sobolVals_total = {} #S^t_A = 1 - S^{clos}_A^c
        sobolVals_main = {}
        indiv_output_variance = {}
        if x_interval_of_interest is not None:
            if x_interval_of_interest_str not in model.exprimentDataDict.keys():
                model.exprimentDataDict[x_interval_of_interest_str] = {}
            model.exprimentDataDict[x_interval_of_interest_str][model.numItersPerNAtIdx[x_interval_of_interest_str]] = {}

        if scalarDiffuIdx not in model.exprimentDataDict.keys():
            model.exprimentDataDict[scalarDiffuIdx] = {}
        model.exprimentDataDict[scalarDiffuIdx][model.numItersPerNAtIdx[scalarDiffuIdx]] = {}
        for N in N_set:
            if model.model_type in ['diffusion_1D_both']:
                sobolDict = {}
                sobolDict_aggr = {}
                for indexSet in myIdxSupset:
                    sobol, sobol_aggr = _getSobolIdx_clos(model, indexSet, N, scalarDiffuIdx, x_interval_of_interest=x_interval_of_interest)
                    sobolDict[indexSet] = sobol
                    sobolDict_aggr[indexSet] = sobol_aggr
                    
                # model.sobolVals_clos[f"{N}"] = sobolDict
                # model.sobolVals_clos_aggr[f"{N}"] = sobolDict_aggr
                sobolVals_clos[f"{N}"] = sobolDict
                sobolVals_clos_aggr[f"{N}"] = sobolDict_aggr
            elif model.model_type in ['ishigami_vect', 'toy_model_vect',  'toy_1_vect', 'toy_2_vect']:
                sobolDict_aggr = {}
                for indexSet in myIdxSupset:
                    sobol_aggr = _getSobolIdx_clos(model, indexSet, N, scalarDiffuIdx, x_interval_of_interest=x_interval_of_interest)
                    sobolDict_aggr[indexSet] = sobol_aggr
                sobolVals_clos_aggr[f"{N}"] = sobolDict_aggr
            else:
                sobolDict = {}
                indiv_output_variance_curr_N = []
                for indexSet in myIdxSupset:
                    V_hat_A, V_hat = _getSobolIdx_clos(model, indexSet, N, scalarDiffuIdx)
                    sobolDict[indexSet] = V_hat_A / V_hat
                    indiv_output_variance_curr_N.append(V_hat)
                # sobolDict = {indexSet: _getSobolIdx_clos(model, indexSet, N, scalarDiffuIdx) for indexSet in myIdxSupset}
                sobolVals_clos[f"{N}"] = sobolDict
                indiv_output_variance[f"{N}"] = np.mean(indiv_output_variance_curr_N)

            if model.model_type not in ['ishigami_vect', 'toy_model_vect', 'toy_1_vect', 'toy_2_vect']:
                if model.model_type in ['ishigami'] and model.specifyX3:
                        sobolVals_total[f"{N}"] = {indexSet: 1-sobolVals_clos[f"{N}"][getIndexComplement(indexSet)] for indexSet in myIdxSupset if getIndexComplement(indexSet) != '00'}
                        sobolVals_total[f"{N}"]['11'] = 1.0 # by default set to 1.0
                else:
                    sobolVals_total[f"{N}"] = {indexSet: 1-sobolVals_clos[f"{N}"][getIndexComplement(indexSet)] for indexSet in myIdxSupset if getIndexComplement(indexSet) != '000'}
                    sobolVals_total[f"{N}"]['111'] = 1.0 # by default set to 1.0
                sobolVals_main[f"{N}"] = _getMainSobolsFromClosed(sobolDict)

        #write data to model's exprimentDataDict
        # model.exprimentDataDict[model.numItersPerNAtIdx[scalarDiffuIdx]]["sobolVals_clos"] = model.sobolVals_clos
        if model.model_type not in ['ishigami_vect', 'toy_model_vect', 'toy_1_vect', 'toy_2_vect']:
            model.exprimentDataDict[scalarDiffuIdx][model.numItersPerNAtIdx[scalarDiffuIdx]]["sobolVals_clos"] = sobolVals_clos
            model.exprimentDataDict[scalarDiffuIdx][model.numItersPerNAtIdx[scalarDiffuIdx]]["sobolVals_total"] = sobolVals_total
            model.exprimentDataDict[scalarDiffuIdx][model.numItersPerNAtIdx[scalarDiffuIdx]]["sobolVals_main"] = sobolVals_main
            model.exprimentDataDict[scalarDiffuIdx][model.numItersPerNAtIdx[scalarDiffuIdx]]["indiv_output_variance"] = indiv_output_variance
        if model.model_type in ['diffusion_1D_both', 'ishigami_vect', 'toy_model_vect', 'toy_1_vect', 'toy_2_vect']:
            # model.exprimentDataDict[model.numItersPerNAtIdx[scalarDiffuIdx]]["sobolVals_clos_aggr"] = model.sobolVals_clos_aggr
            if x_interval_of_interest is not None:
                model.exprimentDataDict[x_interval_of_interest_str][model.numItersPerNAtIdx[x_interval_of_interest_str]]["sobolVals_clos_aggr"] = sobolVals_clos_aggr
            else:
                model.exprimentDataDict[scalarDiffuIdx][model.numItersPerNAtIdx[scalarDiffuIdx]]["sobolVals_clos_aggr"] = sobolVals_clos_aggr
        model.numItersPerNAtIdx[scalarDiffuIdx] += 1
        if x_interval_of_interest is not None:
            model.numItersPerNAtIdx[x_interval_of_interest_str] += 1
        # if returnResults:
        #     return model.sobolVals_clos

def run_getRealizations(model: model, num_of_realizations: int = 5, model_input=None):
    numOfRealizations_prev = len(model.realizationDataDict.keys())
    for num in range(numOfRealizations_prev, numOfRealizations_prev+num_of_realizations):
        if model_input is None:
            modelInput = model.sampleInput()
        else:
            assert len(model_input) == num_of_realizations, f'Need num_of_realizations-many ({num_of_realizations}) passed in. Received {len(model_input)} realizations in model_input.'
            modelInput = model_input[num-numOfRealizations_prev]
        if model.model_type == 'diffusion_1D_explicit':
            output = [model.getModelOutput(input=modelInput, at_x=x) for x in model.mesh]
        else:
            output = model.getModelOutput(input=modelInput)
        model.realizationDataDict[num] = output
    
