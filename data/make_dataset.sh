python /home/grads/tml6120/workspace/dna_grouper/data/episode_dataset.py \
  --data-dirs \
    /home/grads/tml6120/workspace/dna_grouper/data/debug/L15 \
    /home/grads/tml6120/workspace/dna_grouper/data/debug/L20 \
  --split train \
  --episode-size 6 \
  --episodes-per-epoch 24 \
  --batch-size 4 \
  --seed 3
