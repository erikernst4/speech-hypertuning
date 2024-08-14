ginpipe configs/default_parameters.gin \
			configs/train.gin \
			configs/load_dataset.gin \
			configs/load_wavs.gin \
			configs/datasets/voxceleb1.gin \
			--module_list configs/imports \
			--project_name 13-layer-poolings--after-fix \
			--experiment_name "fixed-layer_4" \
			    --mods "models.S3PRLUpstreamMLPDownstreamForCls.layer_pooling_layer=@models.FixedLayerPooling" "models.FixedLayerPooling.layer_idx_to_use=4" "models.SelfAttentionPooling.num_heads=16" "models.SelfAttentionPooling.dropout=0.1" "models.SelfAttentionPooling.use_positional_encoding=True" "models.S3PRLUpstreamMLPDownstreamForCls.time_pooling_layer=@models.SelfAttentionPooling"

ginpipe configs/default_parameters.gin \
			configs/train.gin \
			configs/load_dataset.gin \
			configs/load_wavs.gin \
			configs/datasets/voxceleb1.gin \
			--module_list configs/imports \
			--project_name 13-layer-poolings--after-fix \
			--experiment_name "self_attention-16_heads-wo_pos_encoding" \
			    --mods "models.S3PRLUpstreamMLPDownstreamForCls.layer_pooling_layer=@models.SelfAttentionLayerPooling" "models.SelfAttentionLayerPooling.num_heads=16" "models.SelfAttentionLayerPooling.dropout=0.1" "models.SelfAttentionLayerPooling.use_positional_encoding=False" "models.SelfAttentionPooling.num_heads=16" "models.SelfAttentionPooling.dropout=0.1" "models.SelfAttentionPooling.use_positional_encoding=True" "models.S3PRLUpstreamMLPDownstreamForCls.time_pooling_layer=@models.SelfAttentionPooling"
