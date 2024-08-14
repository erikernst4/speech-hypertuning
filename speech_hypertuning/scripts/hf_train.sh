ginpipe configs/default_parameters.gin \
			configs/train_hf.gin \
			configs/load_dataset.gin \
			configs/load_wavs.gin \
			configs/datasets/voxceleb1.gin \
			--module_list configs/imports \
			--project_name 13-debugging \
			--experiment_name "16_heads-w_scaling-w_pos_encoding_+_dropout" \
			    --mods "tasks.models.SelfAttentionPooling.num_heads=16" "tasks.models.SelfAttentionPooling.dropout=0.1" "tasks.models.SelfAttentionPooling.use_positional_encoding=True" "tasks.models.HFUpstreamMLPDownstreamForCls.pooling_layer=@tasks.models.SelfAttentionPooling" "tasks.models.HFUpstreamMLPDownstreamForCls.normalize_upstream_embeddings=True" "tasks.models.HFUpstreamMLPDownstreamForCls.normalization_method='dataset_scaling'" "tasks.models.calculate_dataset_upstream_mean_and_std_from_wavs.saving_path='/home/eernst/Voxceleb1/wavlm_hf'" "tasks.models.calculate_dataset_upstream_mean_and_std_from_wavs.model_cls=@tasks.models.HFUpstreamMLPDownstreamForCls"
