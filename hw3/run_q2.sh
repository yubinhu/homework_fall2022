source activate cs285
nohup python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_doubledqn_4 --double_q --seed 4 &> ddqn4.out &
nohup python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_doubledqn_5 --double_q --seed 5 &> ddqn5.out &
nohup python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_doubledqn_6 --double_q --seed 6 &> ddqn6.out &