ginpipe configs/default_parameters.gin \
			configs/train.gin \
			configs/load_dataset.gin \
			configs/load_wavs.gin \
			configs/datasets/voxceleb1.gin \
			--module_list configs/imports \
			--project_name 14-poolings-order \
			--experiment_name "fixed_layer_4-self_attention_time_pooling" \
			    --mods "tasks.models.S3PRLUpstreamMLPDownstreamForCls.time_pooling_before_layer_pooling=False" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.layer_pooling_layer=@tasks.models.FixedLayerPooling" "tasks.models.FixedLayerPooling.layer_idx_to_use=4" "tasks.models.SelfAttentionPooling.num_heads=16" "tasks.models.SelfAttentionPooling.dropout=0.1" "tasks.models.SelfAttentionPooling.use_positional_encoding=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.time_pooling_layer=@tasks.models.SelfAttentionPooling" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalize_upstream_embeddings=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalization_method='dataset_scaling'" "tasks.models.calculate_dataset_upstream_mean_and_std_from_wavs.saving_path='/home/eernst/Voxceleb1/wavlm'" "tasks.models.calculate_dataset_upstream_mean_and_std_from_wavs.model_cls=@tasks.models.S3PRLUpstreamMLPDownstreamForCls"

ginpipe configs/default_parameters.gin \
			configs/train.gin \
			configs/load_dataset.gin \
			configs/load_wavs.gin \
			configs/datasets/voxceleb1.gin \
			--module_list configs/imports \
			--project_name 14-poolings-order \
			--experiment_name "weighted_average-self_attention_time_pooling" \
			    --mods "tasks.models.S3PRLUpstreamMLPDownstreamForCls.time_pooling_before_layer_pooling=False" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.layer_pooling_layer=@tasks.models.WeightedAverageLayerPooling" "tasks.models.SelfAttentionPooling.num_heads=16" "tasks.models.SelfAttentionPooling.dropout=0.1" "tasks.models.SelfAttentionPooling.use_positional_encoding=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.time_pooling_layer=@tasks.models.SelfAttentionPooling" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalize_upstream_embeddings=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalization_method='dataset_scaling'" "tasks.models.calculate_dataset_upstream_mean_and_std_from_wavs.saving_path='/home/eernst/Voxceleb1/wavlm'" "tasks.models.calculate_dataset_upstream_mean_and_std_from_wavs.model_cls=@tasks.models.S3PRLUpstreamMLPDownstreamForCls"

ginpipe configs/default_parameters.gin \
			configs/train.gin \
			configs/load_dataset.gin \
			configs/load_wavs.gin \
			configs/datasets/voxceleb1.gin \
			--module_list configs/imports \
			--project_name 14-poolings-order \
			--experiment_name "self_attention_layer_pooling-self_attention_time_pooling" \
			    --mods "tasks.models.S3PRLUpstreamMLPDownstreamForCls.time_pooling_before_layer_pooling=False" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.layer_pooling_layer=@tasks.models.SelfAttentionLayerPooling" "tasks.models.SelfAttentionLayerPooling.num_heads=16" "tasks.models.SelfAttentionLayerPooling.dropout=0.1" "tasks.models.SelfAttentionLayerPooling.use_positional_encoding=False" "tasks.models.SelfAttentionPooling.num_heads=16" "tasks.models.SelfAttentionPooling.dropout=0.1" "tasks.models.SelfAttentionPooling.use_positional_encoding=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.time_pooling_layer=@tasks.models.SelfAttentionPooling" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalize_upstream_embeddings=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalization_method='dataset_scaling'" "tasks.models.calculate_dataset_upstream_mean_and_std_from_wavs.saving_path='/home/eernst/Voxceleb1/wavlm'" "tasks.models.calculate_dataset_upstream_mean_and_std_from_wavs.model_cls=@tasks.models.S3PRLUpstreamMLPDownstreamForCls"

ginpipe configs/default_parameters.gin \
			configs/train.gin \
			configs/load_dataset.gin \
			configs/load_wavs.gin \
			configs/datasets/voxceleb1.gin \
			--module_list configs/imports \
			--project_name 14-poolings-order \
			--experiment_name "weighted_average-mean_time_pooling" \
			    --mods "tasks.models.S3PRLUpstreamMLPDownstreamForCls.time_pooling_before_layer_pooling=False" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.layer_pooling_layer=@tasks.models.WeightedAverageLayerPooling" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.time_pooling_layer=@tasks.models.TemporalMeanPooling" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalize_upstream_embeddings=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalization_method='dataset_scaling'" "tasks.models.calculate_dataset_upstream_mean_and_std_from_wavs.saving_path='/home/eernst/Voxceleb1/wavlm'" "tasks.models.calculate_dataset_upstream_mean_and_std_from_wavs.model_cls=@tasks.models.S3PRLUpstreamMLPDownstreamForCls"
