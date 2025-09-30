import numpy as np

def kernel_sobol(u):
    return 1+(u[0]-0.5)*(u[1]-0.5) + 0.5*((u[0]-u[1])**2 - np.abs(u[0]-u[1])+(1/6))

def kernel_exponential(u, theta=0.5):
    return 1+np.exp(-np.abs(u[0]-u[1])/theta) - theta*(2-np.exp(-np.abs(u[0])/theta)-np.exp(-np.abs(1-u[0])/theta)) - theta*(2-np.exp(-np.abs(u[1])/theta)-np.exp(-np.abs(1-u[1])/theta)) + 2*theta*(1-theta+theta*np.exp(-1/theta))

def get_U_kernel(params, kernel_U_type='sobolev'):
    u = params['u']
    if kernel_U_type == 'sobolev':     
        return kernel_sobol(u)
    elif kernel_U_type == 'exponential':
        theta = params['theta']
        return kernel_exponential(u, theta)
    else:
        raise(f"Invalid Kernel: {kernel_U_type}!")