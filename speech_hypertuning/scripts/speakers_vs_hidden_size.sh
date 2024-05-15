
for speakers in 2 5 10 20 30 50 75 100 150 250 500 1000;
do
	for hidden_size in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768;
	do
	ginpipe configs/default_parameters.gin \
			configs/train.gin \
			configs/load_dataset.gin \
			configs/datasets/voxceleb1.gin \
			--module_list configs/imports \
			--project_name 7-speakers_vs_hidden_size \
			--experiment_name "speakers_$speakers-hidden_size_$hidden_size" \
        --mods "tasks.models.S3PRLUpstreamMLPDownstreamForCls.hidden_dim=$hidden_size" "tasks.load_data.subsample_dataset_with_fixed_n.n_speakers=$speakers" "PROPORTIONS={'train':30, 'validation': 7, 'test': 7}"

	done
done