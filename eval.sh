rm -fr output/*

# python evaluate.py --model trainedModels/mrui.trained.pth.tar --data Dataset --num_images 5 --num_sketches 10 --batch_size 2 --output output

python evaluate.py --model saveMode/last.pth.tar --data Dataset --train_labels train_labels_119.txt --train_embedding train_embeddings_119.npy --test_labels test_labels_6.txt --num_images 5 --num_sketches 10 --batch_size 2 --output output

# python evaluate.py --model trainedModels/train.125.pth.tar --data Dataset --train_labels train_labels_119.txt --train_embedding train_embeddings_119.npy --test_labels test_labels_6.txt --num_images 5 --num_sketches 30 --batch_size 2 --output output

python show_result.py
