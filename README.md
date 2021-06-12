# PDEN

The code for **Progressive Domain Expansion Network for Single Domain Generalization**.

### Dataset

- Download the dataset from [Baidu Cloud Disk](https://pan.baidu.com/s/1kti4P_jEFKKbeqI1g-sP_Q). You will need a code when download the dataset files from Baidu Cloud Dist and the code is b717. 
- Place the dataset files in the path `./data/`

# Train
- First, run the command `sh run_base.sh 0 0`. The command will train the base network.
- Then, run the command `sh run_my.sh 0 0`. The command will train the PDEN model.

