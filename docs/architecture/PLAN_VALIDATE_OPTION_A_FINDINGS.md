# Detailed Plan: Validate Option A "Dual Mode" Hypothesis

**Status:** READY FOR EXECUTION
**Target:** `crm_python` (Python Wrapper)
**Objective:** rigorous verification that the Python Wrapper behaves differently depending on which API is called, proving that "Statelessness causes Failure" within the Option A codebase itself.

---

## 1. Static Analysis (Code Audit)

**Goal:** Confirm the implementation difference in `crm_ml_rl/wrappers/crm_bindings.cpp`.

### Task 1.1: `step()` Analysis (The Success Path)
*   **Target:** `CRMDynamicsWrapper::step()`
*   **Search Pattern:** `ABM4`
*   **Verify:**
    *   Does it access private member variables `this->xdot_nm1`, `this->xdot_nm2`, `this->xdot_nm3`?
    *   Does it call `ABM4_coildyn`?
    *   **Evidence:** Copy the exact lines showing state access.

### Task 1.2: `step_from_seed()` Analysis (The Failure Path)
*   **Target:** `CRMDynamicsWrapper::step_from_seed()`
*   **Search Pattern:** `warning` / `consecutive`
*   **Verify:**
    *   Does it accept `xdot` history as arguments? (Expectation: No).
    *   Does it contain the warning "Consecutive stepping is NOT RELIABLE"?
    *   **Evidence:** Copy the warning text.

---

## 2. Dynamic Analysis (The Experiment)

**Goal:** Force Option A to fail by using it "statelessly".

### Task 2.1: Create Experiment Script
**File:** `scripts/verify_option_a_stateless_failure.py`

**Algorithm:**
```python
def run_experiment():
    # Setup
    wrapper = CRMDynamics()
    wrapper.initialize(...)
    
    # 1. Control Group (Stateful)
    print("Running Stateful Control...")
    try:
        for u in trajectory:
            wrapper.step(u) # Uses internal history
        print("Control: SUCCESS")
    except Exception as e:
        print(f"Control: FAILED ({e})")

    # 2. Experimental Group (Stateless)
    print("Running Stateless Experiment...")
    state = wrapper.get_seed_state() # Start fresh
    try:
        for u in trajectory:
            # Manually feed output back to input
            # BUT do not pass history (because API doesn't support it)
            out = wrapper.step_from_seed(u, ..., state['v'], state['w']...)
            
            if out['localmin'] == 3:
                raise RuntimeError("Convergence Failure (localmin=3)")
                
            # Update seed for next step
            state = extract_state(out)
            
        print("Experiment: SUCCESS (Unexpected)")
    except RuntimeError as e:
        print(f"Experiment: CONFIRMED FAILURE ({e})")
```

---

## 3. Success Criteria

The hypothesis is **CONFIRMED** if and only if:
1.  **Static Analysis** shows `step()` uses history and `step_from_seed()` does not.
2.  **Dynamic Analysis** shows the Control Group succeeds and the Experimental Group fails on the *exact same trajectory* ($94.3\text{mm}$).

This will definitively prove that **Statelessness = Convergence Failure** for this system.
