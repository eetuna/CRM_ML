import sys, os
sys.path.insert(0, '/workspaces/catheter/CRM_ML')
os.chdir('/workspaces/catheter/CRM_ML')
import torch, numpy as np, crm_torch_ext
from crm_torch_ext import _crm_torch_ext
print(f"Loaded _crm_torch_ext from: {_crm_torch_ext.__file__}")
from crm_ml_rl.wrappers import crm_python

_crm_torch_ext.initialize_params(
    "data/catheter_params/CatheterParameterSet_1_dyn.txt",
    "data/catheter_params/CatheterSpatialConfiguration_1.txt")
_crm_torch_ext.set_timestep(0.02)
_crm_torch_ext.set_integrator("abm4")
_crm_torch_ext.set_integration_step_size(0.1)
_crm_torch_ext.set_damping([12.18, 12.18, 284.43, 0.0305, 0.0305, 0.00503])

dyn = crm_python.CRMDynamics()
dyn.load_parameters(
    "data/catheter_params/CatheterParameterSet_1_dyn.txt",
    "data/catheter_params/CatheterSpatialConfiguration_1.txt")
dyn.set_damping(np.array([12.18, 12.18, 284.43, 0.0305, 0.0305, 0.00503]))
dyn.dt = 0.02
dyn.integration_step_size = 0.1
dyn.set_integrator("abm4")
dyn.initialize_from_kinematics(np.array([0.1, 0.0, 0.0]), 94.3)
seed = dyn.get_seed_state()

currents = torch.tensor([0.1, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
insertion = torch.tensor([94.3], dtype=torch.float64)
seed_v = torch.from_numpy(seed['v'])
seed_w = torch.from_numpy(seed['w'])
seed_p = torch.from_numpy(seed['p'])
seed_R = torch.from_numpy(seed['R'])
seed_xf = torch.from_numpy(seed['xf'])
seed_mL = torch.from_numpy(seed['mL'])
seed_nL = torch.from_numpy(seed['nL'])

print("Forward...")
output = crm_torch_ext.crm_step(currents, insertion, seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL)
print(f"Output: {output}")

print("\nBackward...")
output[0].backward()
print(f"Currents grad (from output[0]): {currents.grad}")

# Compare with Python
result = dyn.linearize_full_seed_action_from_seed_implicit(
    currents=np.array([0.1, 0.0, 0.0]), insertion_length=94.3,
    v_in=seed['v'], w_in=seed['w'], p_in=seed['p'], R_in=seed['R'],
    xf_in=seed['xf'], mL_in=seed['mL'], nL_in=seed['nL'],
    return_debug=True)

print(f"\nConverged BVP solution comparison:")
cpp_mL = np.array([0.00179876, 0.0145056, 0.296439]) # From previous log
print(f"Python mL_star: {result['base']['next_mL'][0]}")
print(f"C++ mL_star:    {cpp_mL}")

print(f"\nB matrix comparison (first row):")
print(f"Python B[0, :]: {result['B'][0, :]}")
# In C++, currents grad is B^T @ grad_output. 
# For output[0].backward(), grad_output = [1,0,0,0,0,0], so currents.grad = B[0, 0:3]
print(f"C++ B[0, 0:3]:  {currents.grad.detach().numpy()}")

# Check full B matrix first row
print(f"\nFull B matrix comparison (first row, including insertion):")
print(f"Python: {result['B'][0, :]} (grad_ins={result['grad_insertion'][0]})")

