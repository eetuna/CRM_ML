# Analysis: Backward Pass Strategy (RK4 vs. ABM4)

**Date:** December 28, 2025
**Subject:** Investigation into why the Option A Backward Pass uses stateless integration despite having access to the stateful ABM4 engine.

---

## 1. The Observation

*   **Forward Pass (`step`):** Uses **Stateful ABM4**. It consumes and updates the history buffer ($x'_{t-1 \dots t-3}$) to achieve high convergence on stiff problems.
*   **Backward Pass (`linearize`):** Uses **Stateless RK4**. It effectively resets the integrator for the linearization step, ignoring the history buffer.
*   **The Artifact:** The AD codebase (`CRMDYN_..._autodiff_eigen.hpp`) *contains* a fully templated ABM4 implementation, yet it is bypassed during the standard backward pass.

## 2. The Rationale: Implicit Differentiation vs. BPTT

The decision to avoid ABM4 in the backward pass is driven by the choice of differentiation strategy.

### Strategy A: Backpropagation Through Time (BPTT)
*   **Mechanism:** Differentiate through the integrator steps. $\nabla x_t$ depends on $\nabla x_{t-1}, \nabla x_{t-2}, \dots$.
*   ** Requirement:** Must use the *exact same integrator* (ABM4) for forward and backward to capture the dependency chain.
*   **Cost:** Memory grows linearly with time steps. Gradient calculation becomes $O(T^2)$ or $O(T)$ with massive storage.
*   **Stability:** Differentiating through a stiff BVP solver's history is often numerically unstable.

### Strategy B: Implicit Function Theorem (IFT) - **Current Choice**
*   **Mechanism:** Assume the forward pass found an equilibrium $F(x^*, u) = 0$. Calculate gradients by solving the linear system at that point: $\frac{dx}{du} = -(\frac{\partial F}{\partial x})^{-1} \frac{\partial F}{\partial u}$.
*   **Independence:** The *path* taken to find $x^*$ (ABM4 vs RK4) is irrelevant to the gradient at $x^*$, provided $x^*$ is correct.
*   **Benefit:** The backward pass is **local**. It depends only on the current state, not the history.
*   **Implementation:** Since the gradient is local, we can use a simpler, stateless integrator (RK4) or just the residual function to compute the Jacobians $\partial F/\partial x$ and $\partial F/\partial u$.

## 3. Why is the ABM4 AD code there?

The presence of `ABM4` in the AD headers suggests:
1.  **Completeness:** The author ported the full physics library to AD templates.
2.  **Future Option:** It allows for "Forward-Mode AD" (Sensitivity Analysis) where propagating history is cheaper (just wider numbers).
3.  **Validation:** It enables checking if the "Path Independent" assumption of IFT holds.

## 4. Implication for Option C

To match Option A's success:
1.  **Forward:** Must use **ABM4** (Stateful) to find the correct $x^*$.
2.  **Backward:** Should continue using **IFT** (Stateless) on that $x^*$. 
    *   **Action:** We must implement a "Gradient Stop" on the history tensor. We pass history to the forward function for *value* calculation, but we block gradients from flowing back into it.

**Conclusion:** Option C should use ABM4 for physics (Forward) but treat the history as a constant for gradients (Backward).
