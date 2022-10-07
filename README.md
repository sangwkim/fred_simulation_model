# fred_simulation_model

This repository contains the source code to train a simulation model for the FrED (Fiber Extrusion Device) used in the paper [Dynamic Control of a Fiber Manufacturing Process Using Deep Reinforcement Learning](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9437478). This also includes a code that simulates a simple proportional-integral control on the learned model.

## Prerequisites

Use the below commands to install required dependencies.

```bash
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install pkbar
```

train_fred_010.py: trains an RNN for the dynamic simulation model for FrED

pi_simul_010.py: simulates a PI feedback controller on the learned FrED model
