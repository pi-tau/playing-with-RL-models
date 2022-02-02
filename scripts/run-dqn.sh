pwd
python -m src.agents.dqn.run \
--device=cpu \
--game=Pong-v4 \
--experiences=1000000 \
--max_steps=10000 \
--batch_size=256 \
--lr=1e-3 \
--epsilon=1.0 \
--Q_update_every=500 \
--Q_target_update_every=1000 \
--Q_regressions=10 \
--capacity=250000 \
--min_experiences=2000 \
--output_dir=dqn-run-pong
