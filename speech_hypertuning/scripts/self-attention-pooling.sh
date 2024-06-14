ginpipe configs/default_parameters.gin \
			configs/train.gin \
			configs/load_dataset.gin \
			configs/load_wavs.gin \
			configs/datasets/voxceleb1.gin \
			--module_list configs/imports \
			--project_name 11-self-attention-pooling \
			--experiment_name "1_head-wo_scaling" \
			    --mods "tasks.models.S3PRLUpstreamMLPDownstreamForCls.pooling_layer=@tasks.models.SelfAttentionPooling"


# ginpipe configs/default_parameters.gin \
# 			configs/train.gin \
# 			configs/load_dataset.gin \
# 			configs/load_wavs.gin \
# 			configs/datasets/voxceleb1.gin \
# 			--module_list configs/imports \
# 			--project_name 11-self-attention-pooling \
# 			--experiment_name "self_attention-1_head" \
# 			    --mods "tasks.models.S3PRLUpstreamMLPDownstreamForCls.pooling_layer=@tasks.models.SelfAttentionPooling" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalize_upstream_embeddings=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalization_method='dataset_scaling'" "tasks.models.calculate_dataset_pooled_upstream_mean_and_std_from_wavs.saving_path='attn_embeddings'" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalize_upstream_embeddings=True" "tasks.models.calculate_dataset_pooled_upstream_mean_and_std_from_wavs.model_cls=@tasks.models.S3PRLUpstreamMLPDownstreamForCls"