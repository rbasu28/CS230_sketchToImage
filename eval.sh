# python evaluate.py --model saveMode/last.pth.tar --data Dataset --num_images 3 --num_sketches 4 --batch_size 2 --output output

rm -fr output/*

python evaluate.py --model trainedModels/mrui.trained.pth.tar --data Dataset --num_images 5 --num_sketches 30 --batch_size 2 --output output

python show_result.py

