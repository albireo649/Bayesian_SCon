# Project Summary
This is a Bayesian Optimization framework for COMSOL magnet design (Bayesian_SCon). It optimizes permanent magnet configurations in a 3D electromagnetic system.

## Key Components:

* COMSOL Interface (comsol_interface.py): Integrates with COMSOL FEA software to run electromagnetic simulations on magnet geometries

* Bayesian Optimization (bayes_opt.py): Uses PyTorch/BoTorch with Gaussian processes and constrained acquisition functions to find optimal magnet designs

* Configuration (config.py): Defines optimization parameters (magnet heights, widths, offset positioning) with constraints

* Main Pipeline (main_file.py): Orchestrates the simulation-optimization loop

* Objective: Maximize the force (Fx) in the magnetic assembly by optimizing magnet dimensions (M1_H, M1_W, M2_H, M2_W) and their positioning (offset), subject to geometric constraints.