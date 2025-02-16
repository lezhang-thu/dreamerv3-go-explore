set -ex
rm -rf ./logdir/atari_MontezumaRevenge
python3 dreamer.py --configs atari100k \
        --task atari_MontezumaRevenge \
        --logdir ./logdir/atari_MontezumaRevenge \
        --eval_every 1e5 \
        --eval_episode_num 10
