
for n in 1 5 10 20 30 50 75 100 150;
do
	for hidden_size in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192;
	do
	ginpipe configs/default_parameters.gin \
			configs/train.gin \
			configs/load_dataset.gin \
			configs/datasets/voxceleb1.gin \
			--module_list configs/imports \
			--project_name audios_per_sid_vs_hidden_size \
			--experiment_name "audios_per_sid_$n-hidden_size_$hidden_size" \
        --mods "tasks.models.S3PRLUpstreamMLPDownstreamForCls.hidden_dim=$hidden_size" "PROPORTIONS={'train':$n, 'validation': 30, 'test': 30}"

	done
done