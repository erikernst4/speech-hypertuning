# ginpipe configs/default_parameters.gin \
# 			configs/train.gin \
# 			configs/load_dataset.gin \
# 			configs/load_wavs.gin \
# 			configs/datasets/voxceleb1.gin \
# 			--module_list configs/imports \
# 			--project_name 15-pooling-projector \
# 			--experiment_name "self_attention_time_pooling-weighted_average" \
# 			    --mods "tasks.models.S3PRLUpstreamMLPDownstreamForCls.pooling_projector=False" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.time_pooling_before_layer_pooling=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.layer_pooling_layer=@tasks.models.WeightedAverageLayerPooling" "tasks.models.SelfAttentionPooling.num_heads=16" "tasks.models.SelfAttentionPooling.dropout=0.1" "tasks.models.SelfAttentionPooling.use_positional_encoding=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.time_pooling_layer=@tasks.models.SelfAttentionPooling" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalize_upstream_embeddings=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalization_method='dataset_scaling'" "tasks.models.calculate_dataset_upstream_mean_and_std_from_wavs.saving_path='/home/eernst/Voxceleb1/wavlm'" "tasks.models.calculate_dataset_upstream_mean_and_std_from_wavs.model_cls=@tasks.models.S3PRLUpstreamMLPDownstreamForCls"

# ginpipe configs/default_parameters.gin \
# 			configs/train.gin \
# 			configs/load_dataset.gin \
# 			configs/load_wavs.gin \
# 			configs/datasets/voxceleb1.gin \
# 			--module_list configs/imports \
# 			--project_name 15-pooling-projector \
# 			--experiment_name "0_layers-self_attention_time_pooling-weighted_average" \
# 			    --mods "tasks.models.S3PRLUpstreamMLPDownstreamForCls.downstream_hidden_layers=0" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.pooling_projector=False" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.time_pooling_before_layer_pooling=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.layer_pooling_layer=@tasks.models.WeightedAverageLayerPooling" "tasks.models.SelfAttentionPooling.num_heads=16" "tasks.models.SelfAttentionPooling.dropout=0.1" "tasks.models.SelfAttentionPooling.use_positional_encoding=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.time_pooling_layer=@tasks.models.SelfAttentionPooling" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalize_upstream_embeddings=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalization_method='dataset_scaling'" "tasks.models.calculate_dataset_upstream_mean_and_std_from_wavs.saving_path='/home/eernst/Voxceleb1/wavlm'" "tasks.models.calculate_dataset_upstream_mean_and_std_from_wavs.model_cls=@tasks.models.S3PRLUpstreamMLPDownstreamForCls"

ginpipe configs/default_parameters.gin \
			configs/train.gin \
			configs/load_dataset.gin \
			configs/load_wavs.gin \
			configs/datasets/voxceleb1.gin \
			--module_list configs/imports \
			--project_name 15-pooling-projector \
			--experiment_name "1024_hidden_dim-1_layers-self_attention_time_pooling-weighted_average" \
			    --mods "tasks.models.S3PRLUpstreamMLPDownstreamForCls.pooling_projector=False" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.time_pooling_before_layer_pooling=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.layer_pooling_layer=@tasks.models.WeightedAverageLayerPooling" "tasks.models.SelfAttentionPooling.num_heads=16" "tasks.models.SelfAttentionPooling.dropout=0.1" "tasks.models.SelfAttentionPooling.use_positional_encoding=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.time_pooling_layer=@tasks.models.SelfAttentionPooling" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalize_upstream_embeddings=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalization_method='dataset_scaling'" "tasks.models.calculate_dataset_upstream_mean_and_std_from_wavs.saving_path='/home/eernst/Voxceleb1/wavlm'" "tasks.models.calculate_dataset_upstream_mean_and_std_from_wavs.model_cls=@tasks.models.S3PRLUpstreamMLPDownstreamForCls"
