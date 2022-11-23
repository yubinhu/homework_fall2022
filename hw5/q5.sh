python cs285/scripts/run_hw5_iql.py --env_name PointmassEasy-v0 \
--exp_name q5_easy_supervised_lam1_tau0.5 --use_rnd \
--num_exploration_steps=20000 \
--awac_lambda=1 \
--iql_expectile=0.5 # {0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99}