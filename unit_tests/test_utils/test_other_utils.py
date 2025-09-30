import pytest
# unit_testing -> DONE.
from utils.other_utils import flipStr, directBinStrSum, getIndexSuperset, getIndexSubsets, getIndexComplement, getSingletonIndexAsInt
# unit_testing -> xXx NOT DONE xXx!
from utils.other_utils import save_model, load_model, get_x_boundary_indices_on_1D_FEM_mesh, generate_xdmf_from_dot_msh, generate_2D_mesh, gen_uniform_1d_mesh_from_interval_and_resolution

#-------------------------------
# test_flipStr
#-------------------------------
def test_flipStr_flip_input():
    test_str = '123'
    assert flipStr(test_str) == '321'
def test_flipStr_symmetric_input():
    test_str = '101'
    assert flipStr(test_str) == '101'
def test_flipStr_flip_one_hot_input():
    test_str = '1010'
    assert flipStr(test_str) == '0101'
def test_flipStr_wrong_input_type():
    test_str = 101
    with pytest.raises(ValueError):
        flipStr(test_str)

#-------------------------------
# test_directBinStrSum
#-------------------------------
def test_directBinStrSum_regular_sum_101():
    test_str = "101"
    assert directBinStrSum(test_str) == 2
def test_directBinStrSum_regular_sum_11111():
    test_str = "11111"
    assert directBinStrSum(test_str) == 5
def test_directBinStrSum_wrong_input_1234():
    test_str = "1234"
    with pytest.raises(ValueError):
        directBinStrSum(test_str)
def test_directBinStrSum_wrong_input_non_string():
    test_str = 1234
    with pytest.raises(ValueError):
        directBinStrSum(test_str)

#-------------------------------
# test_getIndexSuperset
#-------------------------------

def test_getIndexSuperset_vectSize_2_all_exclude_zero():
    test_vectSize = 2
    assert getIndexSuperset(vectSize=test_vectSize) == ['01', '10', '11']
def test_getIndexSuperset_vectSize_2_all_include_zero():
    test_vectSize = 2
    assert getIndexSuperset(vectSize=test_vectSize, include_zero=True) == ['00','01', '10', '11']
def test_getIndexSuperset_vectSize_2_only_singletons_exclude_zero():
    test_vectSize = 2
    assert getIndexSuperset(vectSize=test_vectSize, only_singletons=True) == ['01', '10']
def test_getIndexSuperset_vectSize_2_only_singletons_include_zero():
    test_vectSize = 2
    assert getIndexSuperset(vectSize=test_vectSize, only_singletons=True, include_zero=True) == ['00', '01', '10']
def test_getIndexSuperset_vectSize_3_all_exclude_zero():
    test_vectSize = 3
    assert getIndexSuperset(vectSize=test_vectSize) == ['001', '010', '011', '100', '101', '110', '111']
def test_getIndexSuperset_vectSize_3_all_include_zero():
    test_vectSize = 3
    assert getIndexSuperset(vectSize=test_vectSize, include_zero=True) == ['000', '001', '010', '011', '100', '101', '110', '111']
def test_getIndexSuperset_vectSize_3_only_singletons_exclude_zero():
    test_vectSize = 3
    assert getIndexSuperset(vectSize=test_vectSize, only_singletons=True) == ['001', '010', '100']
def test_getIndexSuperset_vectSize_3_only_singletons_include_zero():
    test_vectSize = 3
    assert getIndexSuperset(vectSize=test_vectSize, only_singletons=True, include_zero=True) == ['000', '001', '010', '100']
def test_getIndexSuperset_wrong_input_type_vectSize():
    test_vectSize = '3'
    with pytest.raises(ValueError):
        getIndexSuperset(vectSize=test_vectSize)
def test_getIndexSuperset_wrong_input_value_vectSize():
    test_vectSize = -1
    with pytest.raises(ValueError):
        getIndexSuperset(vectSize=test_vectSize)
def test_getIndexSuperset_wrong_input_type_include_zero():
    test_vectSize = 2
    test_include_zero = 'true'
    with pytest.raises(ValueError):
        getIndexSuperset(vectSize=test_vectSize, include_zero=test_include_zero)
def test_getIndexSuperset_wrong_input_type_only_singletons():
    test_vectSize = 2
    test_only_singletons = 'true'
    with pytest.raises(ValueError):
        getIndexSuperset(vectSize=test_vectSize, only_singletons=test_only_singletons)

#-------------------------------
# test_getIndexSubsets
#-------------------------------
def test_getIndexSubsets_indexStr_11():
    test_indexStr = '11'
    assert getIndexSubsets(indexStr=test_indexStr) == ['00', '10', '01', '11']
def test_getIndexSubsets_indexStr_101():
    test_indexStr = '101'
    assert getIndexSubsets(indexStr=test_indexStr) == ['000', '100', '001', '101']
def test_getIndexSubsets_wrong_input_type_indexStr():
    test_indexStr = 101
    with pytest.raises(ValueError):
        getIndexSubsets(indexStr=test_indexStr)
def test_getIndexSubsets_wrong_input_value_indexStr():
    test_indexStr = '1234'
    with pytest.raises(ValueError):
        getIndexSubsets(indexStr=test_indexStr)

#-------------------------------
# test_getIndexComplement
#-------------------------------
def test_getIndexComplement_idxBinStr_001():
    test_idxBinStr = '001'
    assert getIndexComplement(idxBinString=test_idxBinStr) == '110'
def test_getIndexComplement_idxBinStr_01():
    test_idxBinStr = '01'
    assert getIndexComplement(idxBinString=test_idxBinStr) == '10'
def test_getIndexComplement_idxBinStr_0():
    test_idxBinStr = '0'
    assert getIndexComplement(idxBinString=test_idxBinStr) == '1'
def test_getIndexComplement_idxBinStr_1():
    test_idxBinStr = '1'
    assert getIndexComplement(idxBinString=test_idxBinStr) == '0'
def test_getIndexComplement_idxBinStr_101():
    test_idxBinStr = '101'
    assert getIndexComplement(idxBinString=test_idxBinStr) == '010'
def test_getIndexComplement_idxBinStr_11101():
    test_idxBinStr = '11101'
    assert getIndexComplement(idxBinString=test_idxBinStr) == '00010'
def test_getIndexComplement_idxBinStr_11111():
    test_idxBinStr = '11111'
    assert getIndexComplement(idxBinString=test_idxBinStr) == '00000'
def test_getIndexComplement_idxBinStr_0000():
    test_idxBinStr = '0000'
    assert getIndexComplement(idxBinString=test_idxBinStr) == '1111'
def test_getIndexComplement_wrong_input_type_idxBinString():
    test_idxBinStr = 101
    with pytest.raises(ValueError):
        getIndexComplement(idxBinString=test_idxBinStr)
def test_getIndexComplement_wrong_input_value_idxBinString():
    test_idxBinStr = '1234'
    with pytest.raises(ValueError):
        getIndexComplement(idxBinString=test_idxBinStr)

#-------------------------------
# test_getSingletonIndexAsInt
#-------------------------------
def test_getSingletonIndexAsInt_idxBinStr_1():
    test_idxBinStr = '1'
    assert getSingletonIndexAsInt(idxBinString=test_idxBinStr) == 0
def test_getSingletonIndexAsInt_idxBinStr_100():
    test_idxBinStr = '100'
    assert getSingletonIndexAsInt(idxBinString=test_idxBinStr) == 0
def test_getSingletonIndexAsInt_idxBinStr_010():
    test_idxBinStr = '010'
    assert getSingletonIndexAsInt(idxBinString=test_idxBinStr) == 1
def test_getSingletonIndexAsInt_idxBinStr_010000000000():
    test_idxBinStr = '010000000000'
    assert getSingletonIndexAsInt(idxBinString=test_idxBinStr) == 1
def test_getSingletonIndexAsInt_idxBinStr_001():
    test_idxBinStr = '001'
    assert getSingletonIndexAsInt(idxBinString=test_idxBinStr) == 2 
def test_getSingletonIndexAsInt_idxBinStr_000001000():
    test_idxBinStr = '000001000'
    assert getSingletonIndexAsInt(idxBinString=test_idxBinStr) == 5
def test_getSingletonIndexAsInt_wrong_input_type_idxBinString():
    test_idxBinStr = 101
    with pytest.raises(ValueError):
        getSingletonIndexAsInt(idxBinString=test_idxBinStr)
def test_getSingletonIndexAsInt_wrong_input_value_idxBinString_1234():
    test_idxBinStr = '1234'
    with pytest.raises(ValueError):
        getSingletonIndexAsInt(idxBinString=test_idxBinStr)
def test_getSingletonIndexAsInt_wrong_input_value_idxBinString_011():
    test_idxBinStr = '011'
    with pytest.raises(ValueError):
        getSingletonIndexAsInt(idxBinString=test_idxBinStr)