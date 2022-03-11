# Scripts

* **numba_interp.py**: Code for bilinear interpolation, necessary for the implementation of the semi-lagrangian algorithm, used for considering advection terms in PDEs.
* **logistic_SL.py**: Code for solving a single-species advection-reaction-diffusion equation with a logistic reaction via finite differences with a fourth order runge-kutta for reaction and diffusion and a semi-lagrangian integration scheme for advection.
* **adv_reac_diff_HG.py**: Code for reproducing the results of [Hernández-García and López (2004)](https://doi.org/10.1016/j.ecocom.2004.05.002), studying a phytoplankton-zooplankton system under a chaotic flow.
* **pv_gillespie.py**: Individual based simulations of a singles-species spatial birth-death-competition system under a point-vortex flow.
* **pcf_analisys**: Generating plots for the individual based simulation using the pair correlation function.