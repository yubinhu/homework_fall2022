# First subpart-------------------------------------------------------------
# python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --exp_name q2_dqn \
# --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0 --exploit_rew_shift 1 --exploit_rew_scale 100 --seed 41

# python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --exp_name q2_cql \
# --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.1 --exploit_rew_shift 1 --exploit_rew_scale 100

# python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --exp_name q2_cql_no_scale \
# --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.1


# Second subpart-------------------------------------------------------------

python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd \
--num_exploration_steps=5000 --offline_exploitation --cql_alpha=0.1 \
--unsupervised_exploration --exp_name q2_cql_numsteps_5000 --exploit_rew_shift 1 --exploit_rew_scale 100


python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd \
--num_exploration_steps=5000 --offline_exploitation --cql_alpha=0.0 \
--unsupervised_exploration --exp_name q2_dqn_numsteps_5000 --exploit_rew_shift 1 --exploit_rew_scale 100


python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd \
--num_exploration_steps=15000 --offline_exploitation --cql_alpha=0.1 \
--unsupervised_exploration --exp_name q2_cql_numsteps_15000 --exploit_rew_shift 1 --exploit_rew_scale 100


python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd \
--num_exploration_steps=15000 --offline_exploitation --cql_alpha=0.0 \
--unsupervised_exploration --exp_name q2_dqn_numsteps_15000 --exploit_rew_shift 1 --exploit_rew_scale 100


# Third subpart-------------------------------------------------------------