# python train.py --data_dir Dataset --batch_size 4  --checkpoint_dir saveMode --epochs 8 --print_every 10 

# python train.py --data_dir Dataset --batch_size 16  --checkpoint_dir saveMode --epochs 1 --print_every 1 

python train.py --data_dir Dataset --train_labels train_labels_119.txt --train_embedding train_embeddings_119.npy --test_labels test_labels_6.txt --batch_size 32 --checkpoint_dir saveMode --epochs 1 --print_every 10
