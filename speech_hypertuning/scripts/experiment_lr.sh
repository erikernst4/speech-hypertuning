
for lr in 1e-3 1e-4 1e-5 1e-6;
do
	ginpipe configs/default_parameters.gin \
			configs/train.gin \
			configs/load_dataset.gin \
			configs/datasets/voxceleb1.gin \
			--module_list configs/imports \
			--project_name experiment_lr \
			--experiment_name "lr_$lr" \
        --mods "torch.optim.Adam.lr=$lr"
done