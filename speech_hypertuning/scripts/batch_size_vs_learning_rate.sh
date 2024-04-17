
for lr in 1 0.5 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001 0.00005 0.000001;
do
	for batch_size in 1 2 4 8 16 32 64 128 256 512 1024;
	do
	ginpipe configs/default_parameters.gin \
			configs/train.gin \
			configs/load_dataset.gin \
			configs/datasets/voxceleb1.gin \
			--module_list configs/imports \
			--project_name batch_size_vs_learning_rate \
			--experiment_name "batch_size_$batch_size-lr_$lr" \
        --mods TRAIN_BATCH_SIZE=$batch_size "torch.optim.Adam.lr=$lr"
	done
done