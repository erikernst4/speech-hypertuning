# ginpipe configs/default_parameters.gin \
# 			configs/train.gin \
# 			configs/load_dataset.gin \
# 			configs/load_embeddings.gin \
# 			configs/datasets/voxceleb1.gin \
# 			--module_list configs/imports \
# 			--project_name 10-pooling-methods_2 \
# 			--experiment_name "std" \
#         --mods "tasks.models.save_upstream_embeddings.extract_embedding_method=@tasks.models.extract_upstream_embedding_w_std_pooling" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.upstream_eval_mode=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.skip_pooling=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalize_upstream_embeddings=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalization_method='dataset_scaling'" "tasks.models.calculate_dataset_pooled_mean_and_std.saving_path='/home/eernst/Voxceleb1/std_embeddings'" "tasks.models.save_upstream_embeddings.saving_path='/home/eernst/Voxceleb1/std_embeddings'" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalize_upstream_embeddings=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalization_method='dataset_scaling'"


# ginpipe configs/default_parameters.gin \
# 			configs/train.gin \
# 			configs/load_dataset.gin \
# 			configs/load_embeddings.gin \
# 			configs/datasets/voxceleb1.gin \
# 			--module_list configs/imports \
# 			--project_name 10-pooling-methods_2 \
# 			--experiment_name "max" \
#         --mods "tasks.models.save_upstream_embeddings.extract_embedding_method=@tasks.models.extract_upstream_embedding_w_temporal_max_pooling" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.upstream_eval_mode=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.skip_pooling=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalize_upstream_embeddings=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalization_method='dataset_scaling'" "tasks.models.calculate_dataset_pooled_mean_and_std.saving_path='/home/eernst/Voxceleb1/max_embeddings'" "tasks.models.save_upstream_embeddings.saving_path='/home/eernst/Voxceleb1/max_embeddings'" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalize_upstream_embeddings=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalization_method='dataset_scaling'"

# ginpipe configs/default_parameters.gin \
# 			configs/train.gin \
# 			configs/load_dataset.gin \
# 			configs/load_embeddings.gin \
# 			configs/datasets/voxceleb1.gin \
# 			--module_list configs/imports \
# 			--project_name 10-pooling-methods_2 \
# 			--experiment_name "min" \
#         --mods "tasks.models.save_upstream_embeddings.extract_embedding_method=@tasks.models.extract_upstream_embedding_w_temporal_min_pooling" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.upstream_eval_mode=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.skip_pooling=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalize_upstream_embeddings=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalization_method='dataset_scaling'" "tasks.models.calculate_dataset_pooled_mean_and_std.saving_path='/home/eernst/Voxceleb1/min_embeddings'" "tasks.models.save_upstream_embeddings.saving_path='/home/eernst/Voxceleb1/min_embeddings'" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalize_upstream_embeddings=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalization_method='dataset_scaling'"


# ginpipe configs/default_parameters.gin \
# 			configs/train.gin \
# 			configs/load_dataset.gin \
# 			configs/load_embeddings.gin \
# 			configs/datasets/voxceleb1.gin \
# 			--module_list configs/imports \
# 			--project_name 10-pooling-methods_2 \
# 			--experiment_name "min+max" \
#         --mods "tasks.models.save_upstream_embeddings.extract_embedding_method=@tasks.models.extract_upstream_embedding_w_temporal_minmax_pooling" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.pooling_layer=@tasks.models.TemporalStatisticsPooling" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.upstream_eval_mode=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.skip_pooling=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalize_upstream_embeddings=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalization_method='dataset_scaling'" "tasks.models.calculate_dataset_pooled_mean_and_std.saving_path='/home/eernst/Voxceleb1/minmax_embeddings'" "tasks.models.save_upstream_embeddings.saving_path='/home/eernst/Voxceleb1/minmax_embeddings'" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalize_upstream_embeddings=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalization_method='dataset_scaling'"

# ginpipe configs/default_parameters.gin \
# 			configs/train.gin \
# 			configs/load_dataset.gin \
# 			configs/load_embeddings.gin \
# 			configs/datasets/voxceleb1.gin \
# 			--module_list configs/imports \
# 			--project_name 10-pooling-methods_2 \
# 			--experiment_name "mean+std" \
#         --mods "tasks.models.save_upstream_embeddings.extract_embedding_method=@tasks.models.extract_upstream_embedding_w_temporal_statistics_pooling" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.pooling_layer=@tasks.models.TemporalStatisticsPooling" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.upstream_eval_mode=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.skip_pooling=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalize_upstream_embeddings=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalization_method='dataset_scaling'" "tasks.models.calculate_dataset_pooled_mean_and_std.saving_path='/home/eernst/Voxceleb1/mean+std_embeddings'" "tasks.models.save_upstream_embeddings.saving_path='/home/eernst/Voxceleb1/mean+std_embeddings'" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalize_upstream_embeddings=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalization_method='dataset_scaling'"

# ginpipe configs/default_parameters.gin \
# 			configs/train.gin \
# 			configs/load_dataset.gin \
# 			configs/load_embeddings.gin \
# 			configs/datasets/voxceleb1.gin \
# 			--module_list configs/imports \
# 			--project_name 10-pooling-methods_2 \
# 			--experiment_name "mean+std+min+max" \
#         --mods "tasks.models.save_upstream_embeddings.extract_embedding_method=@tasks.models.extract_upstream_embedding_w_temporal_statistics_plus_pooling" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.pooling_layer=@tasks.models.LayerWiseStatisticsPlusPooling" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.upstream_eval_mode=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.skip_pooling=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalize_upstream_embeddings=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalization_method='dataset_scaling'" "tasks.models.calculate_dataset_pooled_mean_and_std.saving_path='/home/eernst/Voxceleb1/mean+std+min+max_embeddings'" "tasks.models.save_upstream_embeddings.saving_path='/home/eernst/Voxceleb1/mean+std+min+max_embeddings'" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalize_upstream_embeddings=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalization_method='dataset_scaling'"


ginpipe configs/default_parameters.gin \
			configs/train.gin \
			configs/load_dataset.gin \
			configs/load_wavs.gin \
			configs/datasets/voxceleb1.gin \
			--module_list configs/imports \
			--project_name 10-pooling-methods_2 \
			--experiment_name "self_attention-1_head" \
        --mods "tasks.models.S3PRLUpstreamMLPDownstreamForCls.pooling_layer=@tasks.models.SelfAttentionPooling" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalize_upstream_embeddings=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalization_method='dataset_scaling'" "tasks.models.calculate_dataset_pooled_upstream_mean_and_std_from_wavs.saving_path='attn_embeddings'" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalize_upstream_embeddings=True" "tasks.models.calculate_dataset_pooled_upstream_mean_and_std_from_wavs.model_cls=@tasks.models.S3PRLUpstreamMLPDownstreamForCls"