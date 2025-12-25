import argparse
import importlib.util
import json
import os
import sys

import numpy as np


def load_crm_python(so_path: str):
    # Extension init symbol is PyInit_crm_python, so the module name must be "crm_python".
    spec = importlib.util.spec_from_file_location("crm_python", so_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load extension spec from {so_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["crm_python"] = module
    spec.loader.exec_module(module)
    return module


def setup_dyn(mod, insertion: float):
    dyn = mod.CRMDynamics()
    ok = dyn.load_parameters(
        "data/catheter_params/CatheterParameterSet_1_dyn.txt",
        "data/catheter_params/CatheterSpatialConfiguration_1.txt",
    )
    if not ok:
        raise RuntimeError("Failed to load parameters")

    dyn.set_damping(
        np.array(
            [
                12.1761626666366,
                12.1761626666366,
                284.429938756989,
                0.0304776127617393,
                0.0304776127617393,
                0.00502712804532508,
            ],
            dtype=np.float64,
        )
    )
    dyn.dt = 0.05
    dyn.integration_step_size = 0.01
    dyn.initialize_from_kinematics([0.0, 0.0, 0.2], insertion)
    return dyn


def tip_state6(step_out):
    tip_pos = np.asarray(step_out["tip_position"], dtype=np.float64).reshape(-1)
    tip_vel = np.asarray(step_out["tip_velocity"], dtype=np.float64).reshape(-1)
    return np.concatenate([tip_pos, tip_vel], axis=0)


def seed_arrays(seed_npz: str):
    data = np.load(seed_npz)
    return (
        data["v"].astype(np.float64),
        data["w"].astype(np.float64),
        data["p"].astype(np.float64),
        data["R"].astype(np.float64),
        data["xf"].astype(np.float64),
        data["mL"].astype(np.float64),
        data["nL"].astype(np.float64),
    )


def perturb_seed(v, w, p, R, xf, mL, nL, rng: np.random.Generator, sigma: float):
    return (
        v + rng.normal(scale=sigma, size=v.shape),
        w + rng.normal(scale=sigma, size=w.shape),
        p + rng.normal(scale=sigma, size=p.shape),
        R + rng.normal(scale=sigma, size=R.shape),
        xf + rng.normal(scale=sigma, size=xf.shape),
        mL + rng.normal(scale=sigma, size=mL.shape),
        nL + rng.normal(scale=sigma, size=nL.shape),
    )


def directional_check(dyn, currents, insertion, base_seed, A, B, y0, rng, n_dirs=8, du_sigma=1e-4, seed_sigma=1e-4):
    v, w, p, R, xf, mL, nL = base_seed
    errs = []
    seed0_flat = np.concatenate(
        [v.reshape(-1), w.reshape(-1), p.reshape(-1), R.reshape(-1), xf.reshape(-1), mL.reshape(-1), nL.reshape(-1)],
        axis=0,
    )
    for _ in range(n_dirs):
        du = rng.normal(scale=du_sigma, size=(3,))
        dv, dw, dp, dR, dxf, dmL, dnL = perturb_seed(v, w, p, R, xf, mL, nL, rng, seed_sigma)
        step = dyn.step_from_seed(currents + du, insertion, dv, dw, dp, dR, dxf, dmL, dnL)
        if not bool(step["converged"]):
            continue
        y_true = tip_state6(step)

        seed_flat = np.concatenate(
            [
                dv.reshape(-1),
                dw.reshape(-1),
                dp.reshape(-1),
                dR.reshape(-1),
                dxf.reshape(-1),
                dmL.reshape(-1),
                dnL.reshape(-1),
            ],
            axis=0,
        )
        dseed = seed_flat - seed0_flat

        y_pred = y0 + B @ du.reshape(3, 1) + A @ dseed.reshape(-1, 1)
        y_pred = y_pred.reshape(-1)
        denom = np.linalg.norm(y_true) + 1e-9
        errs.append(float(np.linalg.norm(y_pred - y_true) / denom))
    if not errs:
        return float("nan")
    return float(np.mean(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", required=True, help="Path to crm_python*.so")
    ap.add_argument("--seed-npz", required=True, help="NPZ file with v,w,p,R,xf,mL,nL arrays")
    ap.add_argument("--currents-npz", required=True, help="NPZ file with currents array (N,3)")
    ap.add_argument("--out-jsonl", required=True, help="Output JSONL path")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--dirs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--insertion", type=float, default=50.0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    mod = load_crm_python(args.so)
    dyn = setup_dyn(mod, args.insertion)
    base_seed = seed_arrays(args.seed_npz)

    currents_all = np.load(args.currents_npz)["currents"].astype(np.float64)
    results = []

    for idx, currents in enumerate(currents_all):
        if len(results) >= args.trials:
            break

        v, w, p, R, xf, mL, nL = base_seed
        step0 = dyn.step_from_seed(currents, args.insertion, v, w, p, R, xf, mL, nL)
        if not bool(step0["converged"]):
            continue
        y0 = tip_state6(step0).reshape(6, 1)

        # Baseline B via cheap FD on currents only (original dynamics path).
        out_b = dyn.linearize_action_from_seed(currents, args.insertion, v, w, p, R, xf, mL, nL, 1e-4)
        B_fd = np.asarray(out_b["B"], dtype=np.float64)

        # Implicit linearization; try return_debug if available.
        have_debug = False
        out_imp = None
        try:
            out_imp = dyn.linearize_full_seed_action_from_seed_implicit(
                currents, args.insertion, v, w, p, R, xf, mL, nL, 1e-4, 1e-5, 1e-5, 1e-5, True
            )
            have_debug = True
        except TypeError:
            out_imp = dyn.linearize_full_seed_action_from_seed_implicit(
                currents, args.insertion, v, w, p, R, xf, mL, nL, 1e-4, 1e-5, 1e-5, 1e-5
            )

        base = out_imp["base"]
        if not bool(base["converged"]):
            continue

        A_imp = np.asarray(out_imp["A"], dtype=np.float64)
        B_imp = np.asarray(out_imp["B"], dtype=np.float64)
        y0_imp = tip_state6(base).reshape(6, 1)

        # Current-only check for B_fd (no seed perturbation).
        b_errs = []
        for _ in range(args.dirs):
            du = rng.normal(scale=1e-4, size=(3,))
            step = dyn.step_from_seed(currents + du, args.insertion, v, w, p, R, xf, mL, nL)
            if not bool(step["converged"]):
                continue
            y_true = tip_state6(step).reshape(6, 1)
            y_pred = y0 + B_fd @ du.reshape(3, 1)
            b_errs.append(float(np.linalg.norm(y_pred - y_true) / (np.linalg.norm(y_true) + 1e-9)))
        b_err = float(np.mean(b_errs)) if b_errs else float("nan")

        # Full (currents+seed) check: compare implicit linearization against central-difference directional baseline.
        dir_imp_errs = []
        dir_fd_errs = []
        seed0_flat = np.concatenate(
            [v.reshape(-1), w.reshape(-1), p.reshape(-1), R.reshape(-1), xf.reshape(-1), mL.reshape(-1), nL.reshape(-1)],
            axis=0,
        )
        for _ in range(args.dirs):
            du = rng.normal(scale=1e-4, size=(3,))
            dv, dw, dp, dR, dxf, dmL, dnL = perturb_seed(v, w, p, R, xf, mL, nL, rng, 1e-4)

            # y_true at +delta
            step_p = dyn.step_from_seed(currents + du, args.insertion, dv, dw, dp, dR, dxf, dmL, dnL)
            if not bool(step_p["converged"]):
                continue
            y_true = tip_state6(step_p).reshape(6, 1)

            # y_minus at -delta
            dv_m, dw_m, dp_m, dR_m, dxf_m, dmL_m, dnL_m = (
                2 * v - dv,
                2 * w - dw,
                2 * p - dp,
                2 * R - dR,
                2 * xf - dxf,
                2 * mL - dmL,
                2 * nL - dnL,
            )
            step_m = dyn.step_from_seed(currents - du, args.insertion, dv_m, dw_m, dp_m, dR_m, dxf_m, dmL_m, dnL_m)
            if not bool(step_m["converged"]):
                continue
            y_minus = tip_state6(step_m).reshape(6, 1)

            # Directional FD prediction: y0 + (y_plus - y_minus)/2
            y_pred_dirfd = y0 + (y_true - y_minus) * 0.5
            dir_fd_errs.append(float(np.linalg.norm(y_pred_dirfd - y_true) / (np.linalg.norm(y_true) + 1e-9)))

            seed_flat = np.concatenate(
                [dv.reshape(-1), dw.reshape(-1), dp.reshape(-1), dR.reshape(-1), dxf.reshape(-1), dmL.reshape(-1), dnL.reshape(-1)],
                axis=0,
            )
            dseed = (seed_flat - seed0_flat).reshape(-1, 1)
            y_pred_imp = y0_imp + B_imp @ du.reshape(3, 1) + A_imp @ dseed
            dir_imp_errs.append(float(np.linalg.norm(y_pred_imp - y_true) / (np.linalg.norm(y_true) + 1e-9)))

        dir_fd = float(np.mean(dir_fd_errs)) if dir_fd_errs else float("nan")
        dir_imp = float(np.mean(dir_imp_errs)) if dir_imp_errs else float("nan")

        row = {
            "idx": idx,
            "currents": currents.tolist(),
            "B_fd": B_fd.tolist(),
            "A_imp": A_imp.tolist(),
            "B_imp": B_imp.tolist(),
            "curr_only_err_Bfd": b_err,
            "dir_err_dirfd": dir_fd,
            "dir_err_imp": dir_imp,
        }
        if have_debug:
            row["have_ad_jxx"] = bool(out_imp.get("have_ad_jxx", False))
            row["residual_norm"] = float(out_imp.get("residual_norm", float("nan")))
        results.append(row)

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row))
            f.write("\n")

    print(f"Wrote {len(results)} trials to {args.out_jsonl}")


if __name__ == "__main__":
    main()
