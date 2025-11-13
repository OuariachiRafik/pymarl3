CUDA_VISIBLE_DEVICES="0"  python src/main.py --config=hpn_qmix --env-config=sc2_v2_terran with obs_agent_id=True obs_last_action=False runner=parallel batch_size_run=8 buffer_size=5000 t_max=10050000 epsilon_anneal_time=100000 batch_size=128 td_lambda=0.6 mixer=qmix 

CUDA_VISIBLE_DEVICES="1"  python src/main.py --config=hpn_qmix --env-config=sc2_v2_protoss with obs_agent_id=True obs_last_action=False runner=parallel batch_size_run=8 buffer_size=5000 t_max=10050000 epsilon_anneal_time=100000 batch_size=128 td_lambda=0.6 mixer=qmix 

CUDA_VISIBLE_DEVICES="2"  python src/main.py --config=hpn_qmix --env-config=sc2_v2_zerg with obs_agent_id=True obs_last_action=False runner=parallel batch_size_run=8 buffer_size=5000 t_max=10050000 epsilon_anneal_time=100000 batch_size=128 td_lambda=0.6 mixer=qmix 

python3 src/main.py --config=qmix_grf --env-config=gfootball with env_args.map_name=academy_counterattack_hard env_args.num_agents=4


python3 src/main.py --config=qmix_grf --env-config=gfootball with env_args.map_name=academy_counterattack_easy env_args.num_agents=4


python3 src/main.py --config=qmix_grf --env-config=gfootball with env_args.map_name=academy_3_vs_1_with_keeper