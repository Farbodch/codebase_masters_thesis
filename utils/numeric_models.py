import numpy as np
from dolfin import *
from dolfin.function.expression import Expression

from os import path, makedirs
from .other_utils import *
from .math_utils import explicit_1D_Diff_fen as imported_diffuFen_expl
from .math_utils import gen1DDiffusionCoeff as imported_gen1DDC
from pathlib import Path

class SpecialDict(dict):
    """ 
    a class to handle the storage of "alpha" (or the random diffusion coefficient), which is defined 
    as a dolfin Expression. This storage is used to build the bilinear form, which then can be saved
    in the class property "self.a".
    """
    def __setitem__(self, key, value):
        # Customize saving logic based on the type of `value`
        if isinstance(value, Expression):
            # print(f"Storing special object: {key}")
            super().__setitem__(key, value)
        else:
            super().__setitem__(key, value)

#---------
# custom boundary condition definitions for FENICS, for the 2D combustion chamber

class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0) and on_boundary

# Define the varying Dirichlet boundary condition expression
class VaryingDirichletExpression(UserExpression):
    def __init__(self, domain_height, Y_F, Y_O, Y_P, T_0, T_i, **kwargs):
        super().__init__(**kwargs)
        self.domain_height = domain_height
        self.Y_F = Y_F
        self.Y_O = Y_O
        self.Y_P = Y_P
        self.T_0 = T_0
        self.T_i = T_i

    def eval(self, values, x):
        """Evaluation helper for the different Dirichlet boundary conditions at the left side of the spatial domain."""
        # Lower third of the boundary domain (\Gamma_3)
        if x[1] <= 0.33 * self.domain_height:
            values[0] = 0.0
            values[1] = 0.0
            values[2] = 0.0
            values[3] = self.T_0  # 300 kelvin at \Gamma_3 section of the left boundary

        # Upper third of the boundary domain (\Gamma_1)
        elif x[1] >= 0.66 * self.domain_height:
            values[0] = 0.0
            values[1] = 0.0
            values[2] = 0.0
            values[3] = self.T_0  # 300 kelvin at \Gamma_1 section of the left boundary

        # Middle third of the boundary domain (\Gamma_2)
        else:
            values[0] = self.Y_F
            values[1] = self.Y_O
            values[2] = self.Y_P
            values[3] = self.T_i

    def value_shape(self):
        return (4,)  # The expression is vector-valued with four components
#---------

class model:
    def __init__(self, 
                model_type: str = "ishigami", 
                vectSize: int = 3, 
                P: int = 3, 
                mean: float = 1, 
                std: float = 5, 
                meshInterval: int = 128,
                mesh_2D_dir: str = None,
                t_end_cdr: float = 0.5,
                num_steps_cdr: int = 1000,
                mesh_steps: float = 0.025,
                output_paraview: bool = False,
                paraview_data_dir: str = '',
                paraview_data_file_name: str = '',
                model_save_directory: str = '',
                model_save_name: str = '',
                figSaveDir: str = '',
                indicator_toggle: bool = False,
                indicator_constraint_val: float = 0.0,
                FEM_projection: bool = False,
                COMMENT=False,
                reset_exprim_data_toggle=True):
        
        """
        OUTDATED
        Initialize the Soboler class with a specified system type, vector size, and other parameters.
        
        Parameters:
        - model_type (str): The type of system, either 'ishigami', 'diffusion_1D' or 'diffusion_1D_scalar'.
        - vectSize (int): Size of the input vector.
        - P (int): Number of terms for diffusion coefficient.
        - mean (float): Mean value for diffusion coefficient.
        - std (float): Standard deviation for diffusion coefficient.
        - meshInterval (int): Number of intervals in the mesh for diffusion system.
        """
        self.model_save_directory = model_save_directory
        self.model_save_name = model_save_name

        self.valid_system_types = ['ishigami', 'ishigami_vect', 'diffusion_1D', 'diffusion_1D_both', 'diffusion_1D_explicit', 'toy_model_x1sqr_plus_x2sqr' , 'toy_model_vect']
        self.COMMENT = COMMENT
        self.model_type = model_type
        self.vectSize = vectSize
        self.sobolVals_clos = {}
        self.sobolVals_clos_aggr = {}

        self.sobolVals_total = {}
        self.sobolVals_main = {}

        self.numItersPerNAtIdx = {}

        # NOTe: in structure below, domain_index set to 0 for ishigami model.
        #structure:
        # {domain_index(tuple):
        # {experimentNumber_1(int): {'sobolVals_clos': {'N value_1(int)': {'sobolIndices_1': val(int),
        #                                                           'sobolIndices_2': val(int),
        #                                                            ...},
        #                                        'N value_2(int)': {'sobolIndices_1': val(int),
        #                                                           'sobolIndices_2': val(int),
        #                                                            ...},
        #                                        ...}, 
        #                          'sobolVals_clos_aggr: {'N value_1(int)': {'sobolIndices_1': val(int),
        #                                                           'sobolIndices_2': val(int),
        #                                                            ...},
        #                                        'N value_2(int)': {'sobolIndices_1': val(int),
        #                                                           'sobolIndices_2': val(int),
        #                                                            ...},
        #                                        ...},
        #                          'sobolVals_total: {'N value_1(int)': {'sobolIndices_1': val(int),
        #                                                           'sobolIndices_2': val(int),
        #                                                            ...},
        #                                        'N value_2(int)': {'sobolIndices_1': val(int),
        #                                                           'sobolIndices_2': val(int),
        #                                                            ...},
        #                                        ...},
        #  experimentNumber_2(int): {...         ...},
        # ...}
        # }
        if reset_exprim_data_toggle:
            self.exprimentDataDict = {}
            self.realizationDataDict = {}
            self.N_set = []

        self.alpha = None

        self.specifyX3 = False
        if len(figSaveDir) != 0:
            self.figSaveDir = figSaveDir
        else:
            self.figSaveDir = "figs"
        # For Ishigami system
        if self.model_type in ["ishigami", 'ishigami_vect']:
            self.a = 7
            self.b = 0.1
            self.x_3 = 0
            self.specifyX3 = False
            self.ishigami_indicator = False
            self.constraintVal = 0
        elif self.model_type == "diffusion_1D_explicit":
            self.P = P
            self.mean = mean
            self.std = std
            self.meshInterval = meshInterval
            self.mesh=np.linspace(0, 1, meshInterval+1)
            self.mesh_coords = self.mesh

        # For Diffusion system
        elif "diffusion_1D" in self.model_type:
            self.P = P
            self.mean = mean
            self.std = std
            self.meshInterval = meshInterval
            self.diffusion_indicator = indicator_toggle
            self.constraintVal = indicator_constraint_val
            self.projectOutputToCG = FEM_projection
            self.diffuFen_sobol_vect_len = meshInterval

            mesh = UnitIntervalMesh(self.meshInterval)
            self.mesh = mesh
            self.mesh_coords = mesh.coordinates().flatten()
            self.V = FunctionSpace(mesh, 'P', 1)
            u_D = Expression('0', degree=1)
            self.bc = DirichletBC(self.V, u_D, self.boundary)
            self.u = TrialFunction(self.V)
            self.v = TestFunction(self.V)
            self.f = Expression('1', degree=1)

            special_vars = SpecialDict()
            exec(self.getAlphaExecutable(), globals(), special_vars)
            self.alpha = special_vars['alpha']
            self.a = self.alpha*dot(grad(self.u),grad(self.v))*dx

        elif self.model_type == 'cdr':
            self.mesh_save_dir = 'data/CDR/mesh_save_dir'
            self.mesh_save_name = 'rectangle'

            self.mesh_2D_dir = mesh_2D_dir
            if mesh_2D_dir is None:       
                self.domain_width = 1.0
                self.domain_height = 0.5
                self.mesh_steps = 0.025
                print(f'No mesh file was loaded. Creating one with hight {self.domain_height} cm, width {self.domain_width} cm, and resolution of {self.mesh_steps} cm')
                generate_2D_mesh(domain_height=self.domain_height,
                                domain_width=self.domain_width,
                                mesh_steps=self.mesh_steps,
                                mesh_save_dir=self.mesh_save_dir,
                                mesh_save_name=self.mesh_save_name,
                                gen_xdmf=True)
                self.mesh_2D_dir = f'{self.mesh_save_dir}/{self.mesh_save_name}.xdmf'
            else:
                self.mesh_steps = mesh_steps

            self.dolf_mesh = Mesh()
            with XDMFFile(self.mesh_2D_dir) as xdmf:
                xdmf.read(self.dolf_mesh)

            # stoichiometric coefficients
            self.nu_F, self.nu_O, self.nu_P = 2, 1, 2

            # molecular weight of species [gram/mol]
            self.W_F, self.W_O, self.W_P = 2.016, 31.9, 18.0 

            # density of mixture [gr/cm^3]
            self.rho = 1.39e-3

            # universal gas constant [J/(mol.K)]
            self.R = 8.314472

            # heat of the reaction [kelvin]
            self.Q = 9800

            # diffusivity [cm^2/s]
            self.kappa = 2.0

            self.A = 5e11 # log-unif A = [5e11, 1.5e12]
            self.E = 1.5e3 # log unif E = [1.5e3, 9.5e3]
            self.T_i = 900 # unif T_i = [850, 1000] [kelvin] !TYPO in paper!
            self.T_0 = 300 # unif T_0 = [200, 400] [kelvin] !TYPO in paper!
            self.phi = 1.0 # unif phi = [0.5, 1.5] Fuel:oxidizer ratio of the premixed inflow

            # velocity field in x direction -> (u_x=50, u_y=0) [cm/s]
            self.u_x = 50.0
            self.u_y = 0.0
            self.U = Constant((self.u_x, self.u_y))
        
            # boundary values
            self.Y_F_at_boundary = 0.0282
            self.Y_O_at_boundary = 0.2259
            self.Y_P_at_boundary = 0

            # time-stepping params
            self.t_end = t_end_cdr
            self.num_steps = num_steps_cdr
            self.dt = self.t_end/self.num_steps

            self.P3 = FiniteElement("CG", triangle, 3)
            self.elements = MixedElement([self.P3, self.P3, self.P3, self.P3])
            self.V = FunctionSpace(self.dolf_mesh, self.elements)
            self.u = Function(self.V)
            self.Y_F, self.Y_O, self.Y_P, self.T = split(self.u)
            self.v_F, self.v_O, self.v_P, self.v_T = TestFunctions(self.V)
            self.du = TrialFunction(self.V)

            # ICs for Y_F, Y_O, Y_P, T
            self.u0 = Expression(('0', '0', '0', '300'), degree = 0)
            self.u_n = project(self.u0, self.V)
            self.Y_F_n, self.Y_O_n, self.Y_P_n, self.T_n = split(self.u_n)

            #Theta scheme for time-integration 0: Forward-Euler, 0.5: Crank-Nicholson, 1: Backward-Euler
            self.theta = 0.5
            self.Y_F_m = (1 - self.theta) * self.Y_F_n + self.theta * self.Y_F
            self.Y_O_m = (1 - self.theta) * self.Y_O_n + self.theta * self.Y_O
            self.Y_P_m = (1 - self.theta) * self.Y_P_n + self.theta * self.Y_P
            self.T_m = (1 - self.theta) * self.T_n + self.theta * self.T

            # self._build_cdr_system()

            self.solver_params = {
                'newton_solver': {
                    'linear_solver': 'gmres',  # Use GMRES iterative solver
                    'preconditioner': 'ilu',   # Use ILU preconditioner
                    'absolute_tolerance': 1e-6,
                    'relative_tolerance': 1e-6,
                    'maximum_iterations': 35,
                    'relaxation_parameter': 1.0,
                    'convergence_criterion': 'residual',
                    'krylov_solver': {
                        'monitor_convergence': False,
                        'nonzero_initial_guess': True
                    }
                }
            }
            self.mesh_coords = self.dolf_mesh.coordinates()         
            self.mesh_cells = self.dolf_mesh.cells()
            
            self.domain_width = self.mesh_coords[:, 0].max()
            self.domain_height = self.mesh_coords[:, 1].max()

            self.output_paraview = output_paraview
            self.paraview_data_dir = paraview_data_dir
            self.paraview_data_file_name = paraview_data_file_name
        elif self.model_type == 'toy_model_x1sqr_plus_x2sqr':
            self.a = 1
            self.b = 1
            self.specifyX3 = True
            self.choose_y1 = False
            self.choose_y2 = False
            self.indicator = False
            self.constraintVal = 0
        elif self.model_type == 'toy_model_vect':
            self.indicator = False
            self.constraintVal = False
        elif self.model_type == 'toy_1_vect':
            self.indicator = False
            self.constraintVal = False
        elif self.model_type == 'toy_2_vect':
            self.indicator = False
            self.constraintVal = False
        else:
            raise ValueError(f"{self.model_type} is an invalid system type! Choose from {self.valid_system_types}.")
    
    def get_mesh_data(self, print_only=True):
        numV = self.dolf_mesh.num_vertices()
        numE = self.dolf_mesh.num_edges()
        numCells = self.dolf_mesh.num_cells()
        numFaces = self.dolf_mesh.num_faces()
        
        print(f"Vertices: {numV}\nEdges: {numE}\nCells: {numCells}\nFaces: {numFaces}")
        if not print_only:
            return {'num_vertices': numV, 'num_edges': numE, 'num_cells': numCells, 'num_faces': numFaces}
    def save_curr_cdr_output_to_file(self):
        dir_to_save = 'data/fenics_functional_outputs'
        folder_name = f'h_{self.mesh_steps}__N_{self.num_steps}__t_end_{self.t_end}__A_{self.A}__E_{self.E}__Ti_{self.T_i}__T0_{self.T_0}__phi_{self.phi}'.replace('.', '_')
        curr_output_dir = path.join(dir_to_save, folder_name)
        makedirs(curr_output_dir, exist_ok=True)
        file_name_fuel = curr_output_dir + "/__fuel__" + folder_name + '.xdmf'
        file_name_oxygen = curr_output_dir + "/__oxygen__" + folder_name + '.xdmf'
        file_name_product = curr_output_dir + "/__product__" + folder_name + '.xdmf'
        file_name_temp = curr_output_dir + "/__temp__" + folder_name + '.xdmf'

        with XDMFFile(self.dolf_mesh.mpi_comm(), file_name_fuel) as xdmf_1:
            xdmf_1.write_checkpoint(self.fuel_field_t_now, 'fuel_field', 0, XDMFFile.Encoding.HDF5, append=False)
            # xdmf.write(self.fuel_field_t_now)
        with XDMFFile(self.dolf_mesh.mpi_comm(), file_name_oxygen) as xdmf_2:
            xdmf_2.write_checkpoint(self.oxyxen_field_t_now, 'oxygen_field', 0, XDMFFile.Encoding.HDF5, append=False)
            # xdmf.write(self.oxyxen_field_t_now)       
        with XDMFFile(self.dolf_mesh.mpi_comm(), file_name_product) as xdmf_3:
            xdmf_3.write_checkpoint(self.product_field_t_now, 'product_field', 0, XDMFFile.Encoding.HDF5, append=False)
            # xdmf.write(self.product_field_t_now)        
        with XDMFFile(self.dolf_mesh.mpi_comm(), file_name_temp) as xdmf_4:
            xdmf_4.write_checkpoint(self.temp_field_t_now, 'temp_field', 0, XDMFFile.Encoding.HDF5, append=False)
            # xdmf.write(self.temp_field_t_now)

    def load_curr_cdr_output_from_file(self):
        dir_to_save = 'data/fenics_functional_outputs'
        folder_name = f'h_{self.mesh_steps}__N_{self.num_steps}__t_end_{self.t_end}__A_{self.A}__E_{self.E}__Ti_{self.T_i}__T0_{self.T_0}__phi_{self.phi}'.replace('.', '_')
        curr_output_dir = path.join(dir_to_save, folder_name)
        file_name_fuel = curr_output_dir + "/__fuel__" + folder_name + '.xdmf'
        file_name_oxygen = curr_output_dir + "/__oxygen__" + folder_name + '.xdmf'
        file_name_product = curr_output_dir + "/__product__" + folder_name + '.xdmf'
        file_name_temp = curr_output_dir + "/__temp__" + folder_name + '.xdmf'
        # if not path.exists(file_name_fuel):

        #     print(file_name_fuel)
        # if not path.exists(file_name_oxygen):
        #     print('oxy good')
        # if not path.exists(file_name_product):
        #     print('prod good')
        # if not path.exists(file_name_temp):
        #     print('temp good')

        if path.exists(file_name_fuel) and path.exists(file_name_oxygen) and path.exists(file_name_product) and path.exists(file_name_temp):
            V_1 = FunctionSpace(self.dolf_mesh, 'CG', 3)
            V_2 = FunctionSpace(self.dolf_mesh, 'CG', 3)
            V_3 = FunctionSpace(self.dolf_mesh, 'CG', 3)
            V_4 = FunctionSpace(self.dolf_mesh, 'CG', 3)
            self.fuel_field_t_now = Function(V_1)
            self.oxyxen_field_t_now = Function(V_2)
            self.product_field_t_now = Function(V_3)
            self.temp_field_t_now = Function(V_4)
            
            with XDMFFile(self.dolf_mesh.mpi_comm(), file_name_fuel) as xdmf_1:
                xdmf_1.read_checkpoint(self.fuel_field_t_now, 'fuel_field', 0)
            with XDMFFile(self.dolf_mesh.mpi_comm(), file_name_oxygen) as xdmf_2:
                xdmf_2.read_checkpoint(self.oxyxen_field_t_now, 'oxygen_field', 0)
            with XDMFFile(self.dolf_mesh.mpi_comm(), file_name_product) as xdmf_3:
                xdmf_3.read_checkpoint(self.product_field_t_now, 'product_field', 0)
            with XDMFFile(self.dolf_mesh.mpi_comm(), file_name_temp) as xdmf_4:
                xdmf_4.read_checkpoint(self.temp_field_t_now, 'temp_field', 0)

            
        else:
            raise FileNotFoundError(f"Either of fuel, oxygen, product or temp in {folder_name} doesn't exist.")


    def _build_cdr_system(self, A, E, T_i, T_0, phi):
        self.A = A
        self.E = E
        self.T_i = T_i
        self.T_0 = T_0
        self.phi = phi

        # Instantiate the custom expression for the BCs
        self.u_D = VaryingDirichletExpression(
                        domain_height = self.domain_height,
                        Y_F = self.Y_F_at_boundary,
                        Y_O = self.Y_O_at_boundary,
                        Y_P = self.Y_P_at_boundary,
                        T_0 = self.T_0,
                        T_i = self.T_i,
                        degree = 1)
        
        # Apply the boundary condition on the left boundary 
        self.bc = DirichletBC(self.V, self.u_D, LeftBoundary())
        # Note: Neumann boundary condition handled naturally in the weak form

        self.F_Y_F = ((self.Y_F - self.Y_F_n) / self.dt) * self.v_F * dx + self.kappa * dot(grad(self.Y_F_m), grad(self.v_F)) * dx + self.v_F * dot(self.U, grad(self.Y_F_m)) * dx + self.v_F * self.nu_F * (self.W_F/self.rho) * (self.rho*self.Y_F_m/self.W_F)**self.nu_F * (self.rho*self.Y_O_m/self.W_O)**self.nu_O * self.A * exp(-self.E/(self.R*self.T_m)) * dx
        self.F_Y_O = ((self.Y_O - self.Y_O_n) / self.dt) * self.v_O * dx + self.kappa * dot(grad(self.Y_O_m), grad(self.v_O)) * dx + self.v_O * dot(self.U, grad(self.Y_O_m)) * dx + self.v_O * self.nu_O * (self.W_O/self.rho) * (self.rho*self.Y_F_m/self.W_F)**self.nu_F * (self.rho*self.Y_O_m/self.W_O)**self.nu_O * self.A * exp(-self.E/(self.R*self.T_m)) * dx
        self.F_Y_P = ((self.Y_P - self.Y_P_n) / self.dt) * self.v_P * dx + self.kappa * dot(grad(self.Y_P_m), grad(self.v_P)) * dx + self.v_P * dot(self.U, grad(self.Y_P_m)) * dx - self.v_P * self.nu_P * (self.W_P/self.rho) * (self.rho*self.Y_F_m/self.W_F)**self.nu_F * (self.rho*self.Y_O_m/self.W_O)**self.nu_O * self.A * exp(-self.E/(self.R*self.T_m)) * dx
        self.F_T   = ((self.T   - self.T_n)   / self.dt) * self.v_T * dx + self.kappa * dot(grad(self.T_m),   grad(self.v_T)) * dx + self.v_T * dot(self.U, grad(self.T_m))   * dx - self.v_T * self.nu_P * (self.W_P/self.rho) * (self.rho*self.Y_F_m/self.W_F)**self.nu_F * (self.rho*self.Y_O_m/self.W_O)**self.nu_O * self.A * exp(-self.E/(self.R*self.T_m)) * self.Q * dx

        self.F = self.F_Y_F + self.F_Y_O + self.F_Y_P + self.F_T

        self.J = derivative(self.F, self.u, self.du)

    def _setup_output_paraview_files(self):
        paraview_data_dir = self.paraview_data_dir
        paraview_data_file_name = self.paraview_data_file_name
        if paraview_data_dir == '':
            paraview_data_dir = 'data/CDR/paraview_data'
        if paraview_data_file_name == '':
            paraview_data_file_name += f'h_{self.mesh_steps}__N_{self.num_steps}__t_end_{self.t_end}__A_{self.A}__E_{self.E}__Ti_{self.T_i}__T0_{self.T_0}__phi_{self.phi}'.replace('.', '_')
        self.output_dir_paraview = path.join(paraview_data_dir, paraview_data_file_name)
        makedirs(self.output_dir_paraview, exist_ok=True)

        self.vtkfile_F = File(path.join(self.output_dir_paraview, 'fuel.pvd'), 'compressed')
        self.vtkfile_O = File(path.join(self.output_dir_paraview, 'oxygen.pvd'), 'compressed')
        self.vtkfile_P = File(path.join(self.output_dir_paraview, 'product.pvd'), 'compressed')
        self.vtkfile_T = File(path.join(self.output_dir_paraview, 'temperature.pvd'), 'compressed')
    
    def reset_model(self, reset_exprim_data_toggle=True):
        if 'cdr' in self.model_type:
            self.__init__(model_type = self.model_type, 
                    # vectSize = self.vectSize, 
                    # P = self.P, 
                    # mean = self.mean, 
                    # std = self.std, 
                    # meshInterval = self.meshInterval,
                    mesh_2D_dir = self.mesh_2D_dir,
                    t_end_cdr = self.t_end,
                    num_steps_cdr = self.num_steps,
                    mesh_steps = self.mesh_steps,
                    output_paraview = self.output_paraview,
                    paraview_data_dir = self.paraview_data_dir,
                    paraview_data_file_name = self.paraview_data_file_name,
                    figSaveDir = self.figSaveDir,
                    COMMENT = self.COMMENT,
                    reset_exprim_data_toggle=reset_exprim_data_toggle)
        elif 'diffusion' in self.model_type:
            self.__init__(model_type = self.model_type,
                        vectSize = self.vectSize, 
                        P = self.P, 
                        mean = self.mean, 
                        std = self.std, 
                        meshInterval = self.meshInterval,
                        indicator_toggle = self.diffusion_indicator,
                        indicator_constraint_val = self.constraintVal,
                        FEM_projection = self.projectOutputToCG,
                        model_save_directory = self.model_save_directory,
                        model_save_name = self.model_save_name,
                        figSaveDir = self.figSaveDir,
                        COMMENT = self.COMMENT,
                        reset_exprim_data_toggle=reset_exprim_data_toggle)
        else:
            self.__init__(model_type = self.model_type, 
                    vectSize = self.vectSize, 
                    # P = self.P, 
                    # mean = self.mean, 
                    # std = self.std, 
                    # meshInterval = self.meshInterval,
                    # mesh_2D_dir = self.mesh_2D_dir,
                    # t_end_cdr = self.t_end,
                    # num_steps_cdr = self.num_steps,
                    # mesh_steps = self.mesh_steps,
                    # output_paraview = self.output_paraview,
                    # paraview_data_dir = self.paraview_data_dir,
                    # paraview_data_file_name = self.paraview_data_file_name,
                    figSaveDir = self.figSaveDir,
                    COMMENT = self.COMMENT,
                    reset_exprim_data_toggle=reset_exprim_data_toggle)

    def save_my_model(self, file_name='',  save_directory='', set_new_save_directory=False, set_new_save_name=False):
        if save_directory == '':
            save_directory = self.model_save_directory
        if save_directory == '':
            raise ValueError('A save_directory was expected, none were entered!')
        if set_new_save_directory and save_directory != '':
            self.model_save_directory = save_directory

        if file_name == '':
            file_name = self.model_save_name
        if file_name == '':
            raise ValueError('A save_directory was expected, none were entered!')
        if set_new_save_name and file_name != '':
            self.model_save_name = file_name

        self._savestate_FEM()
        save_model(model=self, save_dir=save_directory, save_name=file_name)

        

    def get_cdr(self, coefficients=[5e11,1.5e3,900,300,1.0,0], num_steps=None, t_end=None, reset=False, get_interpolated=False, t=0):
        A = coefficients[0]
        E = coefficients[1]
        T_i = coefficients[2]
        T_0 = coefficients[3]
        phi = coefficients[4]

        set_log_active(False)
        if reset:
            self.reset_model()

        if num_steps is not None and t_end is not None:
            self.num_steps = num_steps
            self.t_end = t_end
            self.dt = t_end/num_steps

        self._build_cdr_system(A=A, E=E, T_i=T_i, T_0=T_0, phi=phi)
        solver_params = self.solver_params
        dt = self.dt
        F = self.F
        u = self.u 
        bc = self.bc
        J = self.J
        u_n = self.u_n

        output_paraview = self.output_paraview
        if output_paraview:
            self._setup_output_paraview_files()

        for _ in range(self.num_steps):
            t += dt
            # solve the variational formulation
            problem = NonlinearVariationalProblem(F, u, bc, J=J)
            solver = NonlinearVariationalSolver(problem)
            solver.parameters.update(solver_params)
            solver.solve()
            u_n.assign(u)
            if output_paraview:
                u1, u2, u3, u4 = u.split() #u1 Fuel, u2 Oxygen, u3 Product, u4 Temperature 
                self.vtkfile_F << (u1, t)
                self.vtkfile_O << (u2, t)
                self.vtkfile_P << (u3, t)
                self.vtkfile_T << (u4, t)

        u1, u2, u3, u4 = u.split()
        self.fuel_field_t_now = u1
        self.oxyxen_field_t_now = u2
        self.product_field_t_now = u3
        self.temp_field_t_now = u4

        if get_interpolated:
            V_scalar = FunctionSpace(self.dolf_mesh, 'CG', 3)
            fuel_interpolated = interpolate(u1, V_scalar)
            oxygen_interpolated = interpolate(u2, V_scalar)
            product_interpolated = interpolate(u3, V_scalar)
            temperature_interpolated = interpolate(u4, V_scalar)

            self.output_t_now_interpolated = {'fuel':fuel_interpolated, 'oxygen':oxygen_interpolated, 'product': product_interpolated, 'temp': temperature_interpolated}

        #     return [fuel_interpolated, oxygen_interpolated, product_interpolated, temperature_interpolated]
        # else:
        #     return [u1, u2, u3, u4]
    
    def getAbsDiffs_ishi(self, N_set: list[int]=[], get_diffs=True):
        """Calculate absolute differences for Ishigami Sobol indices."""
        expectedSobols = {"001": 0.3139, "010": 0.4424, "100": 0.0, "011": 0.7563, "101": 0.5567, "110": 0.4426, "111": 1.0}
        absDiffs = {}

        #N_set here is used to selectively pick for which N values in N_set 
        #should this difference be calculated
        if len(N_set) == 0:
            N_set = self.N_set
        for N_val, sobols in self.sobolVals_clos.items():
            if int(N_val) not in N_set:
                continue
            for key, expected in expectedSobols.items():
                if key not in absDiffs:
                    absDiffs[key] = []
                if get_diffs:
                    absDiffs[key].append(np.abs(expected - sobols.get(key, 0)))
                else:
                    absDiffs[key].append(np.array(sobols.get(key, 0)))
                

        return absDiffs

    @staticmethod
    def boundary(x, on_boundary):
        """Define the boundary condition for diffusion problem."""
        return on_boundary

    def getDiffCoeff(self, P):
        """Return a string representing the diffusion coefficient expression."""
        expr = 'mean'
        for i in range(P):
            k = i + 1
            expr += f'+(std/(pow({k}, 2)*(pow(pi, 2))))*cos(pi*{k}*x[0])*u{k}'
        return expr

    def sampleInput(self):
        """Sample input vector based on system type."""
        if self.model_type in ["ishigami", 'ishigami_vect']:
            return np.random.uniform(-np.pi, np.pi, size=self.vectSize)
        elif "diffusion_1D" in self.model_type:
            return np.random.uniform(-1, 1, size=self.vectSize)
        elif self.model_type in ['toy_model_x1sqr_plus_x2sqr', 'toy_model_vect']:
            return np.random.uniform(-1, 1, size=self.vectSize)
        elif self.model_type in ['toy_1_vect', 'toy_2_vect']:
            return np.random.uniform(0, 1, size=self.vectSize)

    def getAlphaExecutable(self):
        """Generate the executable string to define alpha for diffusion."""
        P = self.P
        mean = self.mean
        std = self.std
        
        U = np.random.uniform(-1,1, P)
        
        alphaStr = f'alpha = Expression(\'{self.getDiffCoeff(P)}\', degree=1, mean={mean}, std={std}'
        for i in range(P):
            alphaStr += f', u{i + 1}={U[i]}'
        alphaStr += ')'
        if self.COMMENT:
            print(U)
            print(alphaStr)
        return alphaStr
    
    def set_uniform_1D_mesh(self, interval):
        self.ishi_interval = interval
        self.mesh_coords = gen_uniform_1d_mesh_from_interval_and_resolution(domain=interval, mesh_resolution=self.meshInterval)

    
    def _toy_model_vect_fen(self, x_1, x_2, x_3):
        """..."""
        return x_1 + x_1*x_2 + x_1*x_3
    
    def _toy_1_fen(self, x_1, x_2, x_3, x_4):
        """..."""
        return 4*x_4+2*x_3+x_2+0.5*x_1
    
    def _toy_2_fen(self, x_1, x_2, x_3, x_4):
        """..."""
        return 4*x_4+2*x_3+4*x_2+0.5*x_1
    

    def toy_model_vect(self, inputVect):
        """..."""
        if 'vect' not in self.model_type and not self.specifyX3:
            x_1, x_2, x_3 = inputVect
        elif self.specifyX3:
            x_1, x_2 = inputVect
            x_3 = self.x_3
        else:
            x_1, x_2 = inputVect

        if 'vect' in self.model_type:
            x_3_list = self.mesh_coords
            if self.indicator:
                outputs = [int(self._toy_model_vect_fen(x_1=x_1, x_2=x_2, x_3=x_3) <= self.constraintVal) for x_3 in x_3_list]
            else:
                outputs = [self._toy_model_vect_fen(x_1=x_1, x_2=x_2, x_3=x_3) for x_3 in x_3_list]
            return np.array(outputs)
        else:
            output = self._toy_model_vect_fen(x_1=x_1, x_2=x_2, x_3=x_3)

        if self.indicator:
            return int(output <= self.constraintVal)
        else:
            return np.array(output)

    def toy_1(self, inputVect):
        """..."""
        if 'vect' not in self.model_type and not self.specifyX3:
            x_, x_1, x_2, x_3 = inputVect
        elif self.specifyX3:
            x_1, x_2, x_3 = inputVect
            x_ = self.x_3
        else:
            x_1, x_2, x_3 = inputVect

        if 'vect' in self.model_type:
            x__list = self.mesh_coords
            if self.indicator:
                outputs = [int(self._toy_1_fen(x_1=x_, x_2=x_1, x_3=x_2, x_4=x_3) <= self.constraintVal) for x_ in x__list]
            else:
                outputs = [self._toy_1_fen(x_1=x_, x_2=x_1, x_3=x_2, x_4=x_3) for x_3 in x__list]
            return np.array(outputs)
        else:
            output = self._toy_1_fen(x_1=x_, x_2=x_1, x_3=x_2, x_4=x_3)

        if self.indicator:
            return int(output <= self.constraintVal)
        else:
            return np.array(output)
    
    def toy_2(self, inputVect):
        """..."""
        if 'vect' not in self.model_type and not self.specifyX3:
            x_, x_1, x_2, x_3 = inputVect
        elif self.specifyX3:
            x_1, x_2, x_3 = inputVect
            x_ = self.x_3
        else:
            x_1, x_2, x_3 = inputVect

        if 'vect' in self.model_type:
            x__list = self.mesh_coords
            if self.indicator:
                outputs = [int(self._toy_2_fen(x_1=x_, x_2=x_1, x_3=x_2, x_4=x_3) <= self.constraintVal) for x_ in x__list]
            else:
                outputs = [self._toy_2_fen(x_1=x_, x_2=x_1, x_3=x_2, x_4=x_3) for x_3 in x__list]
            return np.array(outputs)
        else:
            output = self._toy_2_fen(x_1=x_, x_2=x_1, x_3=x_2, x_4=x_3)

        if self.indicator:
            return int(output <= self.constraintVal)
        else:
            return np.array(output)

        
    def _ishigami(self, x_1, x_2, x_3):
        """Define the Ishigami function."""
        return np.sin(x_1) + self.a * np.sin(x_2)*np.sin(x_2) + self.b * x_3*x_3*x_3*x_3 * np.sin(x_1)

    def toy_model_x1sqr_plus_x2sqr(self, inputVect):
        """..."""
        x_1, x_2 = inputVect
        if self.choose_y1:
            output = self.a*x_1*np.sin(np.pi*x_2)
        elif self.choose_y2:
            output = self.a*x_1*np.cos(np.pi*x_2)
        else:
            output = self.a*x_1*np.sin(np.pi*x_2)*self.a*x_1*np.sin(np.pi*x_2) + self.a*x_1*np.cos(np.pi*x_2)*self.a*x_1*np.cos(np.pi*x_2)   
        if self.indicator:
            return int(output <= self.constraintVal)
        else: 
            return output

    def ishiFen(self, inputVect):
        if 'vect' not in self.model_type and not self.specifyX3:
            x_1, x_2, x_3 = inputVect
        elif self.specifyX3:
            x_1, x_2 = inputVect
            x_3 = self.x_3
        else:
            x_1, x_2 = inputVect
        # if self.specifyX3:
        #     x_3 = self.x_3
        if 'vect' in self.model_type:
            x_3_list = self.mesh_coords
            if self.ishigami_indicator:
                # print(x_3_list)
                # print([self._ishigami(x_1=x_1, x_2=x_2, x_3=x_3) for x_3 in x_3_list])
                outputs = [int(self._ishigami(x_1=x_1, x_2=x_2, x_3=x_3) <= self.constraintVal) for x_3 in x_3_list]
                # print(outputs)
            else:
                outputs = [self._ishigami(x_1=x_1, x_2=x_2, x_3=x_3) for x_3 in x_3_list]
            return np.array(outputs)
        else:
            output = self._ishigami(x_1=x_1, x_2=x_2, x_3=x_3)

        if self.ishigami_indicator:
            return int(output <= self.constraintVal)
        else:
            return np.array(output)
    
    def get_currDiffuCoeffsInMesh(self):
        alpha_in_mesh = interpolate(self.alpha, self.V)
        alpha_in_mesh = alpha_in_mesh.vector().get_local()

        return alpha_in_mesh
    
    def get_difference_bw_alpha_explAndInMesh(self, coefficients):
        alpha_in_mesh = self.get_currDiffuCoeffsInMesh()
        alpha_from_expl_arr = []
        difference_bw_explAndInMesh = 0
        for idx, x in enumerate(self.mesh_coords):
            alpha_from_expl = imported_gen1DDC(x=x, withGenXi=False, explicitXiList=coefficients)
            alpha_from_expl_arr.append(alpha_from_expl)
            difference_bw_explAndInMesh += np.abs(alpha_from_expl-alpha_in_mesh[idx])
        return (difference_bw_explAndInMesh)
    
    def diffuFen(self, coefficients, scalarDiffuIdx=0, reset=False, indicesToEvalProjAt=np.array([0])):
        if reset:
            self.reset_model()
        set_log_active(False)
        v = self.v
        f = self.f
        bc = self.bc
        V = self.V
        a = self.a 
        L = f*v*dx

        """Solve the Diffusion problem with updated alpha coefficients and return the result."""
        for i in range(self.P):
            if self.COMMENT:
                print(f'u{i+1}: {coefficients[i]}')
            setattr(self.alpha, f'u{i+1}', coefficients[i])
        u = Function(V)
        solve(a == L, u, bc)
        self.diffuFen_curr_u = u

        if self.projectOutputToCG:
            mesh = self.mesh
            projTargetSpace = FunctionSpace(mesh, 'CG', 1)
            u_projected = project(u, projTargetSpace)
            y = np.array([u_projected(x) for x in indicesToEvalProjAt])
        else:
            y = u.vector().get_local()
        # if self.model_type == 'diffusion_1D_scalar':   
        #     return y[scalarDiffuIdx]
        
        # #return the whole "image" (of the PDE "mapping")
        # else:
        if self.diffusion_indicator:
            return (y<=self.constraintVal).astype(int)
        return y

    def _savestate_FEM(self):
        path_to_save_FEM_state = f"{self.model_save_directory}/xdmf"
        makedirs(path_to_save_FEM_state, exist_ok=True)
        if path_to_save_FEM_state[-1] != "/":
            path_to_save_FEM_state += '/'
        path_to_save_FEM_state += self.model_save_name
        with XDMFFile(f"{path_to_save_FEM_state}.xdmf") as xdmf:
            u = self.diffuFen_curr_u
            u.rename("u", "")
            # xdmf.parameters["functions_share_mesh"] = True
            # xdmf.parameters["flush_output"] = True
            xdmf.write_checkpoint(u, 'u')

    def __setstate__(self, state):
        self.__dict__.update(state)
        if 'diffusion_1D' in self.model_type:
            self.reset_model(reset_exprim_data_toggle=False)
            
            if self.model_save_directory[-1] != '/':
                self.model_save_directory+='/'
            path_to_load_FEM_state = f"{self.model_save_directory}xdmf"
            if path_to_load_FEM_state[-1] != "/":
                path_to_load_FEM_state += '/'

            p = Path(path_to_load_FEM_state)
            #maybe make this a bit more robuts?
            assert(p.exists()),'Path error. Possible cause: the path when model was saved and current path are incompatibe.'

            path_to_load_FEM_state += self.model_save_name
            
            V = self.V
            u = Function(V)
            with XDMFFile(f"{path_to_load_FEM_state}.xdmf") as xdmf:
                xdmf.read_checkpoint(u, 'u')
            self.diffuFen_curr_u = u
            # load diffuFen_curr_u

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'diffusion_1D' in self.model_type:
            self_params_to_pop = ("mesh", "u", "V", "bc", "v", "f", "alpha", "a", "diffuFen_curr_u")
            for k in self_params_to_pop:
                state.pop(k, None)
        return state        

    def diffuFen_expl(self, coefficients, scalarDiffuIdx=0.5):
        u_x, _ = imported_diffuFen_expl(x=scalarDiffuIdx, 
                                        P=self.P, 
                                        mu=self.mean, 
                                        sigma=self.std, 
                                        withGenXi=False,
                                        passedXiList=coefficients)
        return u_x

    # ishigami', 'diffusion_1D' or 'diffusion_1D_scalar'
    def getModelOutput(self, input, at_x=0.5):
        if self.model_type in ["ishigami", 'ishigami_vect']:
            return self.ishiFen(inputVect=input)
        elif self.model_type == 'diffusion_1D_explicit':
            return self.diffuFen_expl(coefficients=input, scalarDiffuIdx=at_x)
        elif "diffusion_1D" in self.model_type:
            return self.diffuFen(coefficients=input)