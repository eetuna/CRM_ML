# Task-Based Implementation Plan: End-to-End Differentiable Physics Model

This document outlines the tasks required to transition from the current hybrid, residual-based ML approach to a fully end-to-end differentiable model. This will involve re-implementing the core C++ physics simulation in a framework that supports automatic differentiation (autodiff) and integrating it with the existing PyTorch training pipeline.

## Phase 1: Research and Proof of Concept

**Goal:** De-risk the project by evaluating technologies and validating the feasibility of a differentiable physics model.

*   **Task 1.1: Evaluate Differentiable Frameworks**
    *   **Sub-task:** Research C++ autodiff libraries (e.g., `autodiff`, `adept`, `stan-math`).
    *   **Sub-task:** Research pure Python/PyTorch implementations for ordinary differential equations (ODEs), such as `torchdiffeq`.
    *   **Sub-task:** Create a trade-study document comparing performance, ease of integration with Pybind11, learning curve, and community support.
    *   **Deliverable:** A decision on the target framework.

*   **Task 1.2: Prototype a Core Physics Component**
    *   **Sub-task:** Select a small, self-contained but critical part of the physics code (e.g., forward kinematics for a single catheter segment).
    *   **Sub-task:** Re-implement this component in the chosen framework.
    *   **Deliverable:** A functional, differentiable prototype of the selected component.

*   **Task 1.3: Verify Gradient Accuracy**
    *   **Sub-task:** Write unit tests to compare the gradients computed by the new autodiff implementation against a numerical finite-difference check.
    *   **Sub-task:** If available, validate against the analytical Jacobian from the original C++ code.
    *   **Deliverable:** A suite of tests confirming the accuracy and stability of the computed gradients.

*   **Task 1.4: Benchmark Performance**
    *   **Sub-task:** Profile the execution speed of the original C++ component.
    *   **Sub-task:** Profile the forward and backward pass of the new differentiable prototype.
    *   **Deliverable:** A performance report comparing the two implementations and identifying potential bottlenecks.

## Phase 2: Full Implementation of Differentiable Physics Model

**Goal:** Translate the entire C++ physics simulation into the chosen differentiable framework.

*   **Task 2.1: Re-implement Core Data Structures and Operations**
    *   **Sub-task:** Translate the `CRM_CatheterClass.cpp` and `CRM_StateVector_Definitions.hpp` into the new framework.
    *   **Sub-task:** Re-implement the matrix and vector operations from `CRM_MatrixOperations.hpp` using differentiable functions.
    *   **Deliverable:** Differentiable versions of the core data structures.

*   **Task 2.2: Re-implement Forward Kinematics (FK)**
    *   **Sub-task:** Translate the `CRM_ForwardKinematics.cpp` logic.
    *   **Sub-task:** Ensure the numerical integration scheme (e.g., Runge-Kutta) used for the FK is implemented in a way that supports differentiation. This is a critical step.
    *   **Deliverable:** A fully differentiable Forward Kinematics model.

*   **Task 2.3: Re-implement Catheter Dynamics (DYN)**
    *   **Sub-task:** Translate the `CRMDYN.hpp` and `CoilDynamics_Defs.cpp` code.
    *   **Sub-task:** Pay special attention to the numerical integration and solver logic to ensure it's differentiable.
    *   **Deliverable:** A fully differentiable dynamics model that can predict the next state.

## Phase 3: PyTorch Integration

**Goal:** Expose the new differentiable model to Python and make it usable as a standard PyTorch module.

*   **Task 3.1: Develop PyTorch Binding Layer**
    *   **Sub-task:** Create a custom PyTorch `torch.autograd.Function`. This is the bridge between PyTorch and the custom differentiable model.
    *   **Sub-task:** The `forward` method of this function will call the new differentiable physics model.
    *   **Sub-task:** The `backward` method will accept the incoming gradients (from the loss function) and compute the Vector-Jacobian Product (VJP) by calling the backward pass of the differentiable physics model.
    *   **Deliverable:** A Python file containing the `autograd.Function` for the physics model.

*   **Task 3.2: Create a `nn.Module` Wrapper**
    *   **Sub-task:** Create a `torch.nn.Module` that wraps the `autograd.Function`.
    *   **Sub-task:** This will make the differentiable physics model behave like a standard PyTorch layer, which can be seamlessly included in `nn.Sequential` or other model architectures.
    *   **Deliverable:** A PyTorch module representing the differentiable physics model.

## Phase 4: Training, Evaluation, and Advanced Modeling

**Goal:** Leverage the new end-to-end model to improve performance and explore new research directions.

*   **Task 4.1: Update Training and Model Architectures**
    *   **Sub-task:** Modify the `ResidualDynamicsModel` to be a true hybrid model where the ML component and the physics component are combined within the forward pass.
    *   **Sub-task:** Update the training script (`train_dynamics.py`) to use this new hybrid model. The `loss.backward()` call will now automatically propagate gradients through both the ML and physics components.
    *   **Deliverable:** Updated model and training scripts.

*   **Task 4.2: Train and Evaluate New Models**
    *   **Sub-task:** Train several end-to-end hybrid models.
    *   **Sub-task:** Rigorously compare their performance (accuracy, sample efficiency, convergence) against the original residual models and a purely data-driven `FullDynamicsModel`.
    *   **Deliverable:** A comprehensive report and updated trained models.

*   **Task 4.3: Explore Advanced Techniques (Stretch Goals)**
    *   **Sub-task:** Implement a Physics-Informed Neural Network (PINN) by adding a loss term that penalizes violations of the system's governing ODEs.
    *   **Sub-task:** Use the differentiable model for gradient-based trajectory optimization.
    *   **Deliverable:** Experimental results from these advanced techniques.
