python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --exp_name q2_dqn \
--use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0 --exploit_rew_shift 1 --exploit_rew_scale 100

python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --exp_name q2_cql \
--use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.1 --exploit_rew_shift 1 --exploit_rew_scale 100