#!/bin/bash

# python train.py --data_dir Dataset --batch_size 4  --checkpoint_dir saveMode --epochs 8 --print_every 10 

# python train.py --data_dir Dataset --batch_size 16  --checkpoint_dir saveMode --epochs 1 --print_every 1 

for i in $(seq 1 30);
do
    echo train script iteration $i
    python train.py --data_dir Dataset --train_labels t44.txt --train_embedding train_embeddings_119.npy --test_labels test_labels_6.txt --batch_size 32 --checkpoint_dir saveMode --epochs 1 --print_every 10
    python train.py --data_dir Dataset --train_labels t33.txt --train_embedding train_embeddings_119.npy --test_labels test_labels_6.txt --batch_size 32 --checkpoint_dir saveMode --epochs 1 --print_every 10
    python train.py --data_dir Dataset --train_labels t22.txt --train_embedding train_embeddings_119.npy --test_labels test_labels_6.txt --batch_size 32 --checkpoint_dir saveMode --epochs 1 --print_every 10
    python train.py --data_dir Dataset --train_labels t11.txt --train_embedding train_embeddings_119.npy --test_labels test_labels_6.txt --batch_size 32 --checkpoint_dir saveMode --epochs 1 --print_every 10
done

