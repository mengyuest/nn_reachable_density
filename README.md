# Reachable States Density Estimation

[![Conference](https://img.shields.io/badge/CoRL2021-Accepted-success)](https://sites.google.com/robot-learning.org/corl2021)
   
[![Arxiv](http://img.shields.io/badge/arxiv-cs:2109.06728-B31B1B.svg)](https://arxiv.org/pdf/2109.06728.pdf)

The official code for "Learning Density Distribution of Reachable States for Autonomous Systems" (CoRL2021 https://openreview.net/pdf?id=sWBqOL5Nh4P)


This code base contains seven parts:

0. Installation requirements
1. Collect simulation data
2. Train neural network
3. Reproduce the result in Table.1
4. Reproduce the result in Figure 2 & 3
5. Reproduce the result in Figure 4
6. RPM computations


```shell
##########################################
## 0. Installation requirements         ##
##########################################
# Works under Ubuntu 18.04 & 20.04
# Needs to have a GPU computing environment
# Recommend RTX 2080Ti

# A full conda package environment can be
# found out in "environment.yml" in the
# same directory

# Download the data from https://drive.google.com/file/d/1qL0E6U1PScEDyOO25gNI8BEuXZLfCzpf/view?usp=sharing
# and unzip it to `./`

## Install Julia 1.4.2



##########################################
## 1. Collect the simulation data       ##
##########################################
# Taking pendulum as an example
python collect_data.py --exp_mode pend --exp_name 10k --dt 0.02 --nt 50 --num_samples 10000 --s_mins -2.1 -5.5 -2.0 -2.0 --s_maxs 2.1 5.5 2.0 2.0 --x_min -6.28 --x_max 6.28 --y_min -11.0 --y_max 11.0 --nx 161 --ny 161 --viz_log_density

# The result can be found in `cache/g****-******_pend_10k`



##########################################
## 2. Train neural network              ##
##########################################
# Taking the previous trajdata to train network
# (replace the XXXXXXXXXX to the previous data dir you have, e.g. g****-******_pend_10k)
python train_nn.py --gpus 0 --exp_mode pend --exp_name lr5k_e200k_norm --hiddens 32 32 --num_epochs 500000 --beta 0.5 --lr 5000 --train_data_path XXXXXXXXXX --wrap_log1 --dyna_weight 1.0 --less_t --train_dyna_only --log_density --t_struct --change_after 200000 --new_show_stat --new_dyna_weight 0.25 --new_lr 2000 --normalize

# The result can be found in `cache/g****-******_pend_lr5k_e200k_norm`



##########################################
## 3. Reproduce the result in Table.1   ##
##########################################
# Run all the commands in this section
# to reproduce result in Table. 1

#(VDP) (Ran)
python exp1_dens.py --gpus 1 --exp_mode vdp --train_data_path data/vdp_data.npz --model_path data/models/vdp_model.ckpt --ep_bandwidth 0.100 --s_mins -2.5 -2.5 --s_maxs 2.5 2.5 --train_ratio 0.5 --real_histogram

#(dint) (Ran)
python exp1_dens.py --gpus 1 --exp_mode dint --train_data_path data/dint_data.npz --model_path data/models/dint_model.ckpt --ep_bandwidth 0.100 --s_mins -0.5 -1.0 --s_maxs 4.0 1.0 --train_ratio 0.5 --real_histogram --max_n_trajs 100000

# (kop)
python exp1_dens.py --gpus 0 --exp_mode kop --train_data_path data/kop_data.npz --model_path data/models/kop_model.ckpt --ep_bandwidth 0.100 --s_mins 0.0 -2.0 -2.0 --s_maxs 2.0 2.0 2.0 --train_ratio 0.5 --real_histogram

#(Pend)(Ran)
python exp1_dens.py --gpus 1 --exp_mode pend --train_data_path data/pend_data.npz --model_path data/models/pend_model.ckpt --ep_bandwidth 0.100 --s_mins -2.1 -5.5 -2.0 -2.0 --s_maxs 2.1 5.5 2.0 2.0 --train_ratio 0.5 --real_histogram

#(Robot) (Ran)
python exp1_dens.py --gpus 1 --exp_mode robot --train_data_path data/robot_data.npz --model_path data/models/robot_model.ckpt --ep_bandwidth 0.100 --s_mins -1.8 -1.8 0.0 1.0 --s_maxs -1.2 -1.2 1.57 1.5 --train_ratio 0.5 --real_histogram

#(Car)(Ran)
python exp1_dens.py --gpus 1 --exp_mode car --train_data_path data/car_data.npz --model_path /data/models/car_model.ckpt --ep_bandwidth 0.100 --s_mins -2.1 -2.1 -0.1 0.0 --s_maxs 2.1 2.1 0.1 0.5 --train_ratio 0.5 --real_histogram

#(Quad)
python exp1_dens.py --gpus 1 --exp_mode quad --train_data_path data/quad_data.npz --model_path data/models/quad_model.ckpt --ep_bandwidth 0.100 --s_mins 4.6500 4.6500 2.9500 0.9400 -0.0500 -0.4999 --s_maxs 4.7500 4.7500 3.0500 0.9600 0.0500 0.4997 --train_ratio 0.5 --real_histogram --bins 16

#(acc) (Ran)
python exp1_dens.py --gpus 1 --exp_mode acc --train_data_path data/acc_data.npz --model_path data/models/acc_model.ckpt --ep_bandwidth 0.100 --s_mins 59.0 26.0 -0.01 30.0 -0.01 -10.1 -2.0 --s_maxs 62.0 30.0 0.01 30.5 0.01 -9.9 2.0 --train_ratio 0.5 --real_histogram --bins 5

#(gcas) (Ran)
python exp1_dens.py --gpus 0 --exp_mode gcas --train_data_path data/gcas_data.npz --model_path data/models/gcas_model.ckpt --ep_bandwidth 0.25 --s_mins 560.0040 0.0750 -0.0100 -0.1000 -0.9996 -0.0100 -0.0100 -0.0100 -0.0100 -0.0100 -0.0100 1150.0158 0.0002 --s_maxs 599.9902 0.1000 0.0100 -0.0750 -0.5002 0.0100 0.0100 0.0100 0.0100 0.0100 0.0100 1199.9896 0.9994 --train_ratio 0.5 --real_histogram --bins 2


#(toon) (Ran)
python exp1_dens.py --gpus 1 --exp_mode toon --train_data_path data/toon_data.npz --model_path data/models/toon_model.ckpt --ep_bandwidth 0.10 --s_mins -0.1000 -0.1000 -0.0999 -0.1000 -0.0999 -0.1000 -0.1000 -0.1000 -0.1000 -0.1000 -0.1000 -0.1000 -0.1000 -0.1000 -0.1000 -0.0100 --s_maxs 0.1000 0.1000 0.1000 0.0999 0.1000 0.1000 0.1000 0.1000 0.1000 0.1000 0.1000 0.1000 0.1000 0.1000 0.1000 0.0100 --train_ratio 0.5 --real_histogram --bins 1 --max_n_trajs 100000 --skip_histogram

# and the result will be shown in the terminal



###############################################
## 4. Reproduce the result in Figure 2 & 3   ##
###############################################
cd plots
python exp4_merge.py --path ../data/final_data_vdp.npz
python exp2_fig_hybrid.py --path ../data/final_data_vdp.npz
python exp2_fig_hybrid.py --path ../data/final_data_dint.npz
python exp2_fig_hybrid.py --path ../data/final_data_robot.npz
python exp2_fig_hybrid.py --path ../data/final_data_car.npz

# and the results will be in `cache/*`



###############################################
## 5. Reproduce the result in Figure 4       ##
###############################################
cd plots
python exp3_var.py

# and the result will be in `cache/fig4.png`


###############################################
## 6. Generate RPM cells from NN models      ##
###############################################
python handle_rpm.py --exp_mode vdp
python handle_rpm.py --exp_mode dint
python handle_rpm.py --exp_mode robot
python handle_rpm.py --exp_mode car


###############################################
## 7. Test output for NN models              ##
###############################################
First download the exp file from https://drive.google.com/file/d/1vP_MOz57OAZLMDjfbiu_jtNL_FUY56lv/view?usp=sharing and unzip it to ./cache

Run the following command:
python get_nn_est.py --gpus 0 --exp_mode vdp --exp_name lr5k_e500k --hiddens 32 32 --num_epochs 500000 --beta 0.5 --lr 5000 --train_data_path exp_vdp --wrap_log1 --dyna_weight 1.0 --less_t --train_dyna_only --log_density --t_struct --change_after 200000 --new_show_stat --new_dyna_weight 0.25 --new_lr 2000 --pretrained_path data/models/vdp_model.ckpt

The result will be ./cache
```




