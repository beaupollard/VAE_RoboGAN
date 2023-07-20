# VAE_RoboGAN
## Building (Linux, Mac OS)

1. Set up the modified \[[RoboGrammar](https://github.com/JiahengHu/RoboGrammar.git)\] repo following the instructions.


2. Install required python packages for VAE_RoboGAN
* pip3 install -r requirements.txt

## Running Examples
### To collect training data for VAE
`cd robot_utils`; 
`python3 collect_data.py -i500000 --grammar_file {PATH_TO_ROBOGRAMMAR}/data/designs/grammar_apr30.dot`

### To train Graph VAE for design encoding,
`python3 vae_train.py --save_dir sum_ls28_pred20 --data_dir new_train_data_loc_prune --gamma 20 
`

### To collect the data for training the reward Net 
`python3 collect_dis_data.py --data_dir new_train_data_loc_prune --save_dir 0to1k_data --task 0
`

Note task 0 to 11 represent nine different environment tasks ID. 

| Task ID  		| Task Name 					|
| ------------- | ------------------------------|
| 0  			| FlatTerrainTask  				|
| 1  			| RidgedTerrainTask  			|
| 2  			| GapTerrainTask  				|
| 3  			| CustomizedWallTerrainTask1  	|
| 4  			| CustomizedWallTerrainTask2  	|
| 5  			| CustomizedSteppedTerrainTask2 |
| 6  			| CustomizedBiModalTerrainTask1 |
| 7  			| CustomizedBiModalTerrainTask2 |
| 8  			| HillTerrainTask  				|
| 9  			| SteppedTerrainTask  			|
| 10  			| CustomizedSteppedTerrainTask1 |
| 11  			| CustomizedBiModalTerrainTask3 |

The Tasks are defined in the [a relative link] robot_utils/tasks.py

### To View the video of the enviroments and the robots
`python3 view_generated_design.py 
`

### To train VAE_roboGAN
`python3 gan_new_rw_dis_estonly.py
`
Note this file use the one hot vector to encode the environment