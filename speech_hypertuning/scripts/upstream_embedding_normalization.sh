ginpipe configs/default_parameters.gin \
			configs/train.gin \
			configs/load_dataset.gin \
			configs/load_embeddings.gin \
			configs/datasets/voxceleb1.gin \
			--module_list configs/imports \
			--project_name 9-normalize_upstream_embedding \
			--experiment_name "without_normalization" \
        --mods "tasks.models.S3PRLUpstreamMLPDownstreamForCls.upstream_eval_mode=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalize_upstream_embeddings=False" "tasks.models.save_upstream_embeddings.saving_path='/home/eernst/Voxceleb1/avg_embeddings'" "tasks.models.save_upstream_embeddings.extract_embedding_method=@tasks.models.extract_upstream_embedding_w_temporal_average_pooling"

ginpipe configs/default_parameters.gin \
			configs/train.gin \
			configs/load_dataset.gin \
			configs/load_embeddings.gin \
			configs/datasets/voxceleb1.gin \
			--module_list configs/imports \
			--project_name 9-normalize_upstream_embedding \
			--experiment_name "with_embedding_normalization" \
        --mods "tasks.models.S3PRLUpstreamMLPDownstreamForCls.upstream_eval_mode=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalize_upstream_embeddings=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalization_method='standard_normalization'" "tasks.models.save_upstream_embeddings.saving_path='/home/eernst/Voxceleb1/avg_embeddings'" "tasks.models.save_upstream_embeddings.extract_embedding_method=@tasks.models.extract_upstream_embedding_w_temporal_average_pooling"


ginpipe configs/default_parameters.gin \
			configs/train.gin \
			configs/load_dataset.gin \
			configs/load_embeddings.gin \
			configs/datasets/voxceleb1.gin \
			--module_list configs/imports \
			--project_name 9-normalize_upstream_embedding \
			--experiment_name "with_pooled_dataset_scaling" \
        --mods "tasks.models.S3PRLUpstreamMLPDownstreamForCls.upstream_eval_mode=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.skip_pooling=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalize_upstream_embeddings=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalization_method='dataset_scaling'" "tasks.models.save_upstream_embeddings.saving_path='/home/eernst/Voxceleb1/avg_embeddings'" "tasks.models.save_upstream_embeddings.extract_embedding_method=@tasks.models.extract_upstream_embedding_w_temporal_average_pooling" "tasks.models.calculate_dataset_pooled_mean_and_std.saving_path='/home/eernst/Voxceleb1/avg_embeddings'"


ginpipe configs/default_parameters.gin \
			configs/train.gin \
			configs/load_dataset.gin \
			configs/load_embeddings.gin \
			configs/datasets/voxceleb1.gin \
			--module_list configs/imports \
			--project_name 9-normalize_upstream_embedding \
			--experiment_name "with_pooled_dataset_layerwise_scaling" \
        --mods "tasks.models.S3PRLUpstreamMLPDownstreamForCls.upstream_eval_mode=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.skip_pooling=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalize_upstream_embeddings=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalization_method='dataset_scaling'" "tasks.models.save_upstream_embeddings.saving_path='/home/eernst/Voxceleb1/avg_embeddings'" "tasks.models.save_upstream_embeddings.extract_embedding_method=@tasks.models.extract_upstream_embedding_w_temporal_average_pooling" "tasks.models.calculate_layerwise_dataset_pooled_mean_and_std.saving_path='/home/eernst/Voxceleb1/layerwise_scaling-mean_pooling'"
