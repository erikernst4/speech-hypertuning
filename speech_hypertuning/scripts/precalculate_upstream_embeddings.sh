ginpipe configs/default_parameters.gin \
			configs/extract_upstream_embeddings.gin \
			configs/datasets/voxceleb1.gin \
			--module_list configs/imports \
			--project_name experiment_lr \
			--experiment_name precalculate_embeddings
