import pytest
import utils.numeric_models as nm
import numpy as np

#-------------------------------
# fixtures
#-------------------------------
@pytest.fixture
def ishigami_model():
    model = nm.model(model_type="ishigami", vectSize=3)
    return model

@pytest.fixture
def ishigami_vect_model():
    model = nm.model(model_type="ishigami_vect", vectSize=2)
    return model

@pytest.fixture
def diffusion_1D_model():
    model = nm.model(model_type="diffusion_1D")
    return model

@pytest.fixture
def diffusion_1D_both():
    model = nm.model(model_type="diffusion_1D_both")
    return model

@pytest.fixture
def diffusion_1D_explicit():
    model = nm.model(model_type="diffusion_1D_explicit")
    return model


#-------------------------------
# test_nm_general
#-------------------------------

#-------------------------------
# test_nm_ishigami
#-------------------------------
def test_ishigami_nm_variable_check_a_default(ishigami_model):
    assert ishigami_model.a == 7

def test_ishigami_nm_variable_check_a_10(ishigami_model):
    ishigami_model.a = 10
    assert ishigami_model.a == 10

def test_ishigami_nm_variable_check_b_default(ishigami_model):
    assert ishigami_model.b == 0.1

def test_ishigami_nm_variable_check_b_5(ishigami_model):
    ishigami_model.b = 5
    assert ishigami_model.b == 5

def test_ishigami_nm_variable_check_specifyX3_default(ishigami_model):
    assert ishigami_model.specifyX3 == False

def test_ishigami_nm_variable_check_ishigami_indicator_default(ishigami_model):
    assert ishigami_model.ishigami_indicator == False

def test_ishigami_nm_variable_check_constraintVal_default(ishigami_model):
    assert ishigami_model.constraintVal == 0

def test_ishigami_nm_x1_0_x2_0_x3_0(ishigami_model):
    assert ishigami_model._ishigami(x_1=0,x_2=0,x_3=0) == 0

def test_ishigami_nm_x1_pi_x2_pi_x3_pi(ishigami_model):
    assert ishigami_model._ishigami(x_1=np.pi,x_2=np.pi,x_3=np.pi) <= 1e-14

def test_ishigami_nm_x1_piDiv2_x2_piDiv2_x3_piDiv2(ishigami_model):
    assert ishigami_model._ishigami(x_1=np.pi/2,x_2=np.pi/2,x_3=1) == 8.1

#-------------------------------
# test_nm_ishigami_vect
#-------------------------------

def test_ishigami_vect_nm_set_ishi_mesh_mesh_coords_type(ishigami_vect_model):
    ishigami_vect_model.meshInterval = 2
    ishigami_vect_model.set_ishi_mesh(interval=[-1,1])
    assert type(ishigami_vect_model.mesh_coords) == np.ndarray

def test_ishigami_vect_nm_set_ishi_mesh_mesh_coords_size(ishigami_vect_model):
    ishigami_vect_model.meshInterval = 128
    ishigami_vect_model.set_ishi_mesh(interval=[-1,1])
    assert len(ishigami_vect_model.mesh_coords) == ishigami_vect_model.meshInterval+1

def test_ishigami_vect_nm_set_ishi_mesh_values_meshInterval_4_intervals_0_to_1(ishigami_vect_model):
    ishigami_vect_model.meshInterval = 4
    ishigami_vect_model.set_ishi_mesh(interval=[0,1])
    assert np.array_equal(ishigami_vect_model.mesh_coords, np.array([0., 0.25, 0.5, 0.75, 1.]))

def test_ishigami_vect_nm_set_ishi_mesh_values_meshInterval_4_intervals_neg1_to_1(ishigami_vect_model):
    ishigami_vect_model.meshInterval = 4
    ishigami_vect_model.set_ishi_mesh(interval=[-1,1])
    assert np.array_equal(ishigami_vect_model.mesh_coords, np.array([-1., -0.5, 0., 0.5, 1.]))

def test_ishigami_vect_nm_ishiFen_output_meshInterval_8_intervals_negPi_to_Pi_x1_0_x2_0(ishigami_vect_model):
    ishigami_vect_model.meshInterval = 8
    ishigami_vect_model.set_ishi_mesh(interval=[-np.pi, np.pi])
    output = ishigami_vect_model.ishiFen(inputVect=[0,0])
    expected_output = np.zeros(shape=ishigami_vect_model.meshInterval+1)
    assert np.array_equal(output, expected_output)

def test_ishigami_vect_nm_ishiFen_output_meshInterval_8_intervals_negPi_to_Pi_x1_pi_x2_piDiv2(ishigami_vect_model):
    ishigami_vect_model.meshInterval = 8
    ishigami_vect_model.set_ishi_mesh(interval=[-np.pi, np.pi])
    output = ishigami_vect_model.ishiFen(inputVect=[np.pi,np.pi/2])
    expected_output = np.ones(shape=ishigami_vect_model.meshInterval+1)*ishigami_vect_model.a
    difference_array = np.round(output-expected_output, decimals=14)
    zero_array = np.zeros(shape=ishigami_vect_model.meshInterval+1)
    assert np.array_equal(difference_array, zero_array)

def test_ishigami_vect_nm_ishiFen_output_meshInterval_8_intervals_negPi_to_Pi_x1_0_x2_1(ishigami_vect_model):
    ishigami_vect_model.meshInterval = 8
    ishigami_vect_model.set_ishi_mesh(interval=[-np.pi, np.pi])
    output = ishigami_vect_model.ishiFen(inputVect=[0,1])
    expected_output = np.array([4.95651393, 4.95651393, 4.95651393, 4.95651393, 4.95651393, 4.95651393, 4.95651393, 4.95651393, 4.95651393])
    difference_array = np.round(output-expected_output, decimals=8)
    zero_array = np.zeros(shape=ishigami_vect_model.meshInterval+1)
    assert np.array_equal(zero_array, difference_array)

def test_ishigami_vect_nm_ishiFen_output_indicator_10_meshInterval_8_intervals_negPi_to_Pi_x1_pi_x2_piDiv2(ishigami_vect_model):
    ishigami_vect_model.meshInterval = 8
    ishigami_vect_model.ishigami_indicator = True
    ishigami_vect_model.constraintVal = 10
    ishigami_vect_model.set_ishi_mesh(interval=[-np.pi, np.pi])
    output = ishigami_vect_model.ishiFen(inputVect=[np.pi,np.pi/2])
    expected_output = np.ones(shape=ishigami_vect_model.meshInterval+1, dtype=int)
    assert np.array_equal(output, expected_output)

def test_ishigami_vect_nm_ishiFen_output_indicator_5_meshInterval_8_intervals_negPi_to_Pi_x1_pi_x2_piDiv2(ishigami_vect_model):
    ishigami_vect_model.meshInterval = 8
    ishigami_vect_model.ishigami_indicator = True
    ishigami_vect_model.constraintVal = 5
    ishigami_vect_model.set_ishi_mesh(interval=[-np.pi, np.pi])
    output = ishigami_vect_model.ishiFen(inputVect=[np.pi,np.pi/2])
    expected_output = np.zeros(shape=ishigami_vect_model.meshInterval+1, dtype=int)
    assert np.array_equal(output, expected_output)

def test_ishigami_vect_nm_ishiFen_output_indicator_10_meshInterval_8_intervals_negPi_to_Pi_x1_1_x2_1(ishigami_vect_model):
    ishigami_vect_model.meshInterval = 8
    ishigami_vect_model.ishigami_indicator = True
    ishigami_vect_model.constraintVal = 10
    ishigami_vect_model.set_ishi_mesh(interval=[-np.pi, np.pi])
    output = ishigami_vect_model.ishiFen(inputVect=[1,1])
    expected_output = np.array([0, 1, 1, 1, 1, 1, 1, 1, 0])
    assert np.array_equal(output, expected_output)

def test_ishigami_vect_nm_ishiFen_output_indicator_8_meshInterval_8_intervals_negPi_to_Pi_x1_1_x2_1(ishigami_vect_model):
    ishigami_vect_model.meshInterval = 8
    ishigami_vect_model.ishigami_indicator = True
    ishigami_vect_model.constraintVal = 8
    ishigami_vect_model.set_ishi_mesh(interval=[-np.pi, np.pi])
    output = ishigami_vect_model.ishiFen(inputVect=[1,1])
    expected_output = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0])
    assert np.array_equal(output, expected_output)
    
#-------------------------------
# test_nm_1D_diffusion_FEM
#-------------------------------

#-------------------------------
# test_nm_1D_diffusion_explicit (NEED math_utils TESTS first!)
#-------------------------------

#-------------------------------
# test_nm_cdr
#-------------------------------