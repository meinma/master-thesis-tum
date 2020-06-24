#!/usr/bin/env bash

#CUDA_VISIBLE_DEVICES="0,1"
CUDA_VISIBLE_DEVICES="1"
datasets_path="./scratch/datasets/"
out_path="./scratch/mnist_test4"
## Test 3 was the first run, 2 the 2nd and one the 3rd run
config="mnist_paper_convnet_gp"
batch_size=200
computation=1.0



if [ -d "$out_path" ]; then
	echo "Careful: directory \"$out_path\" already exists"
	exit 1
fi

space_separated_cuda="${CUDA_VISIBLE_DEVICES//,/ }"
#n_workers=$(echo $space_separated_cuda | wc -w)
n_workers=2
if [ "$n_workers" == 0 ]; then
	echo "You must specify CUDA_VISIBLE_DEVICES"
	exit 1
fi

echo "Downloading dataset"
python -c "import configs.$config as c; import cnn_gp; cnn_gp.DatasetFromConfig(\"$datasets_path\", c)"

echo "Starting kernel computation workers in parallel"
python ./exp_mnist_resnet/stop_time.py startTimer

mkdir "$out_path"
worker_rank=0
for cuda_i in $space_separated_cuda; do
	this_worker="${out_path}/$(printf "%02d_nw%02d.h5" $worker_rank $n_workers)"

	CUDA_VISIBLE_DEVICES=$cuda_i python -m exp_mnist_resnet.save_kernel --n_workers=$n_workers \
		 --worker_rank=$worker_rank --datasets_path="$datasets_path" --batch_size=$batch_size \
		 --config="$config" --out_path="$this_worker" --computation=${computation} &
	pids[${i}]=$!
	worker_rank=$((worker_rank+1))
done
# Wait for all workers
for pid in ${pids[*]}; do
	wait $pid
done

echo "combining all data sets in one"
python -m exp_mnist_resnet.merge_h5_files "${out_path}"/*


echo "Classify using the complete set"
combined_file="${out_path}/$(printf "%02d_nw%02d.h5" 0 $n_workers)"
CUDA_VISIBLE_DEVICES=1 python -m exp_mnist_resnet.classify_gp --datasets_path="$datasets_path" \
	   --config="$config" --in_path="$combined_file" --computation=$computation

python ./exp_mnist_resnet/stop_time.py endTimer


