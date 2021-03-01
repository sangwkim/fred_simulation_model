# fred_simulation_model

## Prerequisites

Use the below commands to install required dependencies. (assumes that you use anaconda environment)

```bash
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install pkbar
```

train_fred_010.py: trains an RNN for the dynamic simulation model for FrED

pi_simul_010.py: simulates a PI feedback controller on the learned FrED model
