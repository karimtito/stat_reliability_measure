# Experimental design: comparison of Statistical Reliability methods for Neural Networks

## 1. Goal 

The goal is to benchmark different methods of Statistical Reliability Engineering and of Rare Event Simulat in the context of Neural Networks classifiers.

The methods to compare include:

- [x] Crude Monte Carlo (MC)
- [x] Multiple Importance Splitting (MLS)
- [x] Hamiltonian within Sequential Monte Carlo (H_SMC)
- [x] Metropolis Adjusted Langevin Algorithm within Sequential Monte Carlo (MALA_SMC)
- [x] First Order Reliability Method (FORM)
- [x] Hybrid (Gradient-Informed) Multiple Importance Splitting (HMLS)
- [x] Importance Sampling  (IS)
- [x] Cross-Entropy Adaptive Importance Sampling (CE-AIS)
- [x] Line Sampling methods (LS)

## 2. Scope of experiments

We evaluate each estimation method on 3 test cases:

- [ ] A linear toy model under Gaussian perturbations, see [`exp_1`](./exp_1) 
- [ ] Models trained on MNIST dataset under uniform noise perturbations, see [`exp_2`](./exp_2) 
- [ ] Models trained on ImageNet dataset under uniform noise perturbations, see [`exp_3`](./exp_3) 

## 3. Rare Event Threshold

We study Rare Events in the case of Neural Networks misclassifications. However 'Rare' is a rather ambiguous term. Therefore on the toy model and and the MNIST models we study the probability level threshold such that a Crude Monte Carlo simulation become less efficient than Rare Event Simulation algorithms.
This corresponds to the experiments scripts [`exp_1_high_low_prob.py`](./exp_1_high_to_low_prob.py).
