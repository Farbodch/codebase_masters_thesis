from scipy import integrate
import numpy as np

def genXi() -> float:
    return np.random.uniform(-1,1)

#if withGenXi is False and no explicitXiList (with P-many elements), then
#the vanilla summation without xi is returned
def gen1DDiffusionCoeff(x: float, 
                        P: int=3, 
                        mu: float=1, 
                        sigma: float=5, 
                        withGenXi: bool=True, 
                        returnGenXi: bool=False, 
                        explicitXiList: list[float]=None
                        ) -> float:
    if returnGenXi:
        genedXi_arr = []

    # def innerSumArg(k: int, passedXi: float=None) -> float:
        
    a = mu 
    for k in range(1,P+1):
        if withGenXi and returnGenXi:
            # if returnGenXi:
            a_k, xi_k = diffuCoeff_innerSumArg(k=k, x=x, sigma=sigma, withGenXi=withGenXi, returnGenXi=returnGenXi)
            a += a_k
            genedXi_arr.append(xi_k)
            # else:
            #     a += diffuCoeff_innerSumArg(k=k, x=x, sigma=sigma, withGenXi=withGenXi)
        elif explicitXiList is not None:
            # assert explicitXiList is not None, "explicitXiList mode invoked, but None was passed for explicitXiList."
            a += diffuCoeff_innerSumArg(k=k, x=x, sigma=sigma, withGenXi=withGenXi, passedXi=explicitXiList[k-1])
        else:
            a += diffuCoeff_innerSumArg(k=k, x=x, sigma=sigma, withGenXi=withGenXi)

    if returnGenXi:
        return a, genedXi_arr
    else:
        return a

def diffuCoeff_innerSumArg(k: int,
                            x: float,
                            sigma: float=5,
                            withGenXi: bool=True,
                            returnGenXi: bool=False,
                            passedXi: float=None
                            ) -> float:
    if withGenXi:
        xi = genXi()
        if returnGenXi:
            return sigma/(k**2 * np.pi**2)*np.cos(np.pi*k*x)*xi, xi
        else:
            return sigma/(k**2 * np.pi**2)*np.cos(np.pi*k*x)*xi
    elif passedXi is not None:
        assert passedXi is not None, "explicitXi array contains values, but None was passed in through passedXi."
        # print(f"x:{x} passedXi:{passedXi} returning: {sigma/(k**2 * np.pi**2)*np.cos(np.pi*k*x)*passedXi}")
        return sigma/(k**2 * np.pi**2)*np.cos(np.pi*k*x)*passedXi
    else:
        return sigma/(k**2 * np.pi**2)*np.cos(np.pi*k*x)


def _diffuCoeff_sumFen(P: int, 
                    sigma: float, 
                    withGenXi: bool,
                    passedXiList: list[float]=None):
    if withGenXi:
        return lambda t: sum((sigma/(k**2*np.pi**2)) * np.cos(np.pi * k * t) * genXi() for k in range(1, P+1))
    elif passedXiList is not None:
        return lambda t: sum((sigma/(k**2*np.pi**2)) * np.cos(np.pi * k * t) * passedXiList[k-1] for k in range(1, P+1))
    else:
        return lambda t: sum((sigma/(k**2*np.pi**2)) * np.cos(np.pi * k * t) for k in range(1, P+1))

def _C1_num_integrand(t: float,
                    P: int,
                    mu: float,
                    sigma: float,
                    withGenXi: bool,
                    passedXiList: list[float]=None):   
    sumFen_P = _diffuCoeff_sumFen(P=P, sigma=sigma, withGenXi=withGenXi, passedXiList=passedXiList)
    return -t/(mu+sumFen_P(t))
def _C1_num(P, mu, sigma, withGenXi, passedXiList):
    result, error_est = integrate.quad(_C1_num_integrand, 0, 1, args=(P, mu, sigma, withGenXi, passedXiList))
    # print(f"C1_num error_est: {error_est}")
    return result
def _C1_denom_integrand(t: float,
                        P: int, 
                        mu: float, 
                        sigma: float,
                        withGenXi: bool,
                        passedXiList: list[float]=None):    
    sumFen_P = _diffuCoeff_sumFen(P=P, sigma=sigma, withGenXi=withGenXi, passedXiList=passedXiList)
    return 1/(mu+sumFen_P(t))
def _C1_denom(P: int,
            mu: float,
            sigma: float,
            withGenXi: bool,
            passedXiList: list[float]=None):  
    result, error_est = integrate.quad(_C1_denom_integrand, 0, 1, args=(P, mu, sigma, withGenXi, passedXiList))
    # print(f"C1_denom error_est: {error_est}")
    return result
def _get_C1(P: int=3, 
            mu: float=1, 
            sigma: float=5,
            withGenXi: bool=False,
            passedXiList: list[float]=None):
    numerator = _C1_num(P, mu, sigma, withGenXi, passedXiList)
    denominator = _C1_denom(P, mu, sigma, withGenXi, passedXiList)
    assert denominator != 0, "C1 denominator found to be zero!"
    return -(numerator/denominator)

def u_integrand(t, P, mu, sigma, withGenXi, passedXiList: list[float]=None):
    sumFen_P = _diffuCoeff_sumFen(P=P, sigma=sigma, withGenXi=withGenXi, passedXiList=passedXiList)
    C1 = _get_C1(P, mu, sigma, withGenXi=withGenXi, passedXiList=passedXiList)
    return (-t+C1)/(mu+sumFen_P(t))
def explicit_1D_Diff_fen(P: int=3, 
                        mu: float=1, 
                        sigma: float=5,
                        x: float=0,
                        withGenXi: bool=True, 
                        returnGenXi: bool=False, 
                        passedXiList: list[float]=None,
                        get_est_error = True
                        ):
    result, error_est = integrate.quad(u_integrand, 0, x, args=(P, mu, sigma, withGenXi, passedXiList))
    # print(f"explicit_1D_Diff error_est: {error_est}")
    if get_est_error:
        return result, error_est
    else:
        return result