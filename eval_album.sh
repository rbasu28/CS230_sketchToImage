rm -fr output/*

export MODEL=trainedModels/mrui.trained.pth.tar
export MODEL=trainedModels/train.125.pth.tar

export TEST_PHOTO=/Users/minglirui/train_data/album_covers_512
export TEST_PHOTO=album_photos

python evaluate_album.py --model $MODEL --test_photos_dir $TEST_PHOTO --test_sketches_dir album_sketches --num_images 5 --num_sketches 14 --batch_size 2 --output output

# python evaluate_album.py --model trainedModels/train.125.pth.tar --test_photos_dir album_photos --test_sketches_dir album_sketches --num_images 5 --num_sketches 3 --batch_size 2 --output output


# python evaluate_album.py --model trainedModels/train.125.pth.tar --test_photos_dir album_photos --test_sketches_dir album_sketches --num_images 5 --num_sketches 3 --batch_size 2 --output output


python show_result.py

