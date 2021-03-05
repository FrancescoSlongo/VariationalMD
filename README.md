# VariationalMD
Depository for the code of my Master thesis. Here there is a guide of where are the relevant files.

#######
 CODE
#######

2D_ToyModel_1: jupyter-notebook with the code to perform the Ratchet-and-Pawl Molecular Dynamics. The file contains several tests for the code that I did while writing it. At the beginning I plot the potential, the Boltzmann probability and the forces acting on the system.

2D_ToyModel_2: jupyter-notebook with only the relevant code of 2D_ToyModel_1. It contains an integrator for the overdamped Langevin equation, an integrator with a constant biasing force and the code that I use for the Ratchet&Pawl algorithm. Every function here can be found in 2D_ToyModel_1, but scattered around.

2D_VariationalMD: jupyter-notebook where I perform the variational method. The file contains the iterative algorithm to find the good reaction coordinate.

Optimization: jupyter-notebook where I tested how to implement correctly the minimization functions of scipy for some case similar to the one I am studying.

######
IMAGES
######

preliminary_work: folder with all the images that I can get from the codes "2D_ToyModel_1" and "2D_ToyModel_2". Inside there are the following folders:
@potential: relevant plots of the potential I am studying.
@simple_langevin: plots of the trajectories and the sampling of the transition region using the simple Langevin integrator. There are two cases: when I start in the intermediate state and when I start in the reactant state. In the latter there is the case with temperature KbT = 1.5, in order to avoid to be stacked in the reactant state.
@steered_langevin: plots of the trajectories or the sampling of the transition region using the Langevin integrator with a constant bias force towards the product.
@ratchet: plots of the trajectories or the sampling of the transition region using the Ratchet&Pawl algorithm using the distance between states as reaction coordinate.

variationalMD: folder with all the images that I get from the code "2D_VariationalMD". Inside there are the following folders:
@Langevin: plots of the trajectories and the sampling of the transition obtained with Ratchet&Pawl algorithm using different reaction coordinates. If the plot finishes with "l", "cU" or "d" it means that I used a linear, circular on the upper plane or distance based committor respectively. If I used a mix of them, in the title I specify the coefficients of the linear combination.
@SelfConsistent: folder with all the results of the iterative procedure. The folder linear contains the result when I use a linear combination with linear coefficients, while quadratic contains the results with squared coefficients. It contains the evolution of the coefficients (plots called coeff), with the relative constraint violation (plots called constr). If the plot finishes with "l", "cU" or "d" it means that I used a linear, circular on the upper plane or distance based committor respectively. In the folder there are also the trajectories and the sampling of the transition region using the coefficients that I get from the iterative procedure.
