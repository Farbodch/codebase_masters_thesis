import numpy as np
from utils.math_utils import explicit_1D_Diff_fen as exp_1D_diff
from utils import numeric_models as nm
from dolfin import conditional, Constant, le, FunctionSpace, project

def gen_toy_1_gamma(x_arr, params):
    return lambda u: int((-(x_arr[0]*x_arr[0])+5*x_arr[1]-(u[0])+(u[1]*u[1])-1)<=params['g_ineq_c'])

def gen_toy_2_gamma(x_arr, params):
    return lambda u: int((-(x_arr[0]*x_arr[0])+5*x_arr[1]-(u[0]*u[0])+(u[1]*u[1])-1)<=params['g_ineq_c'])

def gen_linearDecr_1_gamma(x_arr, params):
    return lambda u: int((4*u[2]+2*u[1]+1*u[0]+0.5*x_arr[0])<=params['g_ineq_c'])

def gen_linearDecr_2_gamma(x_arr, params):
    return lambda u: int((4*u[2]+2*u[1]+4*u[0]+0.5*x_arr[0])<=params['g_ineq_c'])

def gen_X1CosPiX2SinPiX2(x_arr, params):
    return lambda u: int(((u[0]*np.cos(np.pi*u[1])))<=params['g_ineq_c'])

def gen_toy_model_vect(x_arr, params):
    return lambda u: (np.array([params['c_1']*u[0] + x_arr[0]*u[0] + params['c_2']*u[0]*u[1]])<=params['g_ineq_c']).astype(int)



def gen_ishigami(x_arr, params):
    return lambda u: int((np.sin(u[0])+params['a']*(np.sin(u[1])*np.sin(u[1]))+params['b']*(x_arr[0]*x_arr[0]*x_arr[0]*x_arr[0])*np.sin(u[0]))<=params['g_ineq_c'])


def gen_1D_diff_explicit(x_arr, params):   
    def _get_at_x(u):
        model = nm.model(model_type="diffusion_1D_explicit")
        model.P = params['P']
        model.mean = params['mu']
        model.std = params['std']
        model_output = model.diffuFen_expl(coefficients=u,scalarDiffuIdx=x_arr[0])
        if params['auto_mean']:
            params['g_ineq_c'] = np.mean(model_output)
        
        return int(model_output<=params['g_ineq_c'])
    return _get_at_x

def get_1D_diff_FEM(params):
    def _get_at_all_x(u):
        model = nm.model(model_type="diffusion_1D", 
                        P=params['P'], 
                        mean=params['mu'], 
                        std=params['std'],
                        meshInterval=params['meshInterval'])

        model_output = model.diffuFen(coefficients=u)
        if params['auto_mean']:
            params['g_ineq_c'] = np.mean(model_output) 
        return (model_output<=params['g_ineq_c']).astype(int)
    return _get_at_all_x

def get_CDR(params):
    def _get_output_fens(u):
        model = nm.model(model_type='cdr',
                        output_paraview=False,
                        mesh_2D_dir=params['mesh_2D_dir'],
                        mesh_steps=params['mesh_steps'],
                        t_end_cdr=params['t_end'],
                        num_steps_cdr=params['num_steps'])
        model.get_cdr(coefficients=u, reset=True)

        fuel_bool = project(conditional(le(model.fuel_field_t_now,params['g_ineq_c']['fuel']), Constant(1), Constant(0)), FunctionSpace(model.dolf_mesh, 'CG', 1))
        oxygen_bool = project(conditional(le(model.oxyxen_field_t_now,params['g_ineq_c']['oxygen']), Constant(1), Constant(0)), FunctionSpace(model.dolf_mesh, 'CG', 1))
        product_bool = project(conditional(le(model.product_field_t_now,params['g_ineq_c']['product']), Constant(1), Constant(0)), FunctionSpace(model.dolf_mesh, 'CG', 1))
        temp_bool = project(conditional(le(model.temp_field_t_now,params['g_ineq_c']['temp']), Constant(1), Constant(0)), FunctionSpace(model.dolf_mesh, 'CG', 1))

        return [fuel_bool, oxygen_bool, product_bool, temp_bool]
    return _get_output_fens
