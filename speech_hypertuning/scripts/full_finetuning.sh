
# ginpipe configs/default_parameters.gin \
# 			configs/train.gin \
# 			configs/load_dataset.gin \
# 			configs/load_wavs.gin \
# 			configs/datasets/voxceleb1.gin \
# 			--module_list configs/imports \
# 			--project_name 17-full-finetuning \
# 			--experiment_name "full_finetuning-self_attention_time_pooling-weighted_average" \
# 			    --mods "models.S3PRLUpstreamMLPDownstreamForCls.frozen_upstream=False" "models.SelfAttentionPooling.dropout=0.1" "models.SelfAttentionPooling.use_positional_encoding=True" "models.S3PRLUpstreamMLPDownstreamForCls.time_pooling_layer=@models.SelfAttentionPooling" "models.test_model.test_partition='train'"

# ginpipe configs/default_parameters.gin \
# 			configs/test.gin \
# 			configs/load_dataset.gin \
# 			configs/load_wavs.gin \
# 			configs/datasets/voxceleb1.gin \
# 			--module_list configs/imports \
# 			--project_name 14-poolings-order \
# 			--experiment_name "self_attention_time_pooling-weighted_average" \
# 			    --mods "models.S3PRLUpstreamMLPDownstreamForCls.time_pooling_before_layer_pooling=True" "models.S3PRLUpstreamMLPDownstreamForCls.layer_pooling_layer=@models.WeightedAverageLayerPooling" "models.SelfAttentionPooling.num_heads=16" "models.SelfAttentionPooling.dropout=0.1" "models.SelfAttentionPooling.use_positional_encoding=True" "models.S3PRLUpstreamMLPDownstreamForCls.time_pooling_layer=@models.SelfAttentionPooling" "models.test_model.test_partition='train'"


ginpipe configs/default_parameters.gin \
			configs/train.gin \
			configs/load_dataset.gin \
			configs/load_wavs.gin \
			configs/datasets/voxceleb1.gin \
			--module_list configs/imports \
			--project_name 17-full-finetuning \
			--experiment_name "lr-schedule-full_finetuning-fixed_layer_4-self_attention_time_pooling" \
			    --mods "models.S3PRLUpstreamMLPDownstreamForCls.frozen_upstream=False" "models.S3PRLUpstreamMLPDownstreamForCls.time_pooling_before_layer_pooling=False" "models.S3PRLUpstreamMLPDownstreamForCls.layer_pooling_layer=@models.FixedLayerPooling" "models.FixedLayerPooling.layer_idx_to_use=4" "models.SelfAttentionPooling.dropout=0.1" "models.SelfAttentionPooling.use_positional_encoding=True" "models.S3PRLUpstreamMLPDownstreamForCls.time_pooling_layer=@models.SelfAttentionPooling"
