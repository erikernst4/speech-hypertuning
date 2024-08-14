ginpipe configs/default_parameters.gin \
			configs/train.gin \
			configs/load_dataset.gin \
			configs/load_wavs.gin \
			configs/datasets/voxceleb1.gin \
			--module_list configs/imports \
			--project_name 15-linear-between-poolings \
			--experiment_name "self_attention_time_pooling-linear-weighted_average" \
			    --mods "S3PRLUpstreamMLPDownstreamForCls.pooling_projector=True" "S3PRLUpstreamMLPDownstreamForCls.time_pooling_before_layer_pooling=True" "S3PRLUpstreamMLPDownstreamForCls.layer_pooling_layer=@WeightedAverageLayerPooling" "SelfAttentionPooling.num_heads=16" "SelfAttentionPooling.dropout=0.1" "SelfAttentionPooling.use_positional_encoding=True" "S3PRLUpstreamMLPDownstreamForCls.time_pooling_layer=@SelfAttentionPooling"

ginpipe configs/default_parameters.gin \
			configs/train.gin \
			configs/load_dataset.gin \
			configs/load_wavs.gin \
			configs/datasets/voxceleb1.gin \
			--module_list configs/imports \
			--project_name 15-linear-between-poolings \
			--experiment_name "self_attention_layer_pooling-linear-self_attention_time_pooling" \
			    --mods "S3PRLUpstreamMLPDownstreamForCls.pooling_projector=True" "S3PRLUpstreamMLPDownstreamForCls.time_pooling_before_layer_pooling=False" "S3PRLUpstreamMLPDownstreamForCls.layer_pooling_layer=@SelfAttentionLayerPooling" "SelfAttentionLayerPooling.num_heads=16" "SelfAttentionLayerPooling.dropout=0.1" "SelfAttentionLayerPooling.use_positional_encoding=False" "SelfAttentionPooling.num_heads=16" "SelfAttentionPooling.dropout=0.1" "SelfAttentionPooling.use_positional_encoding=True" "S3PRLUpstreamMLPDownstreamForCls.time_pooling_layer=@SelfAttentionPooling"
