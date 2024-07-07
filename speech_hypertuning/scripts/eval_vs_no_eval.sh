ginpipe configs/default_parameters.gin \
			configs/train.gin \
			configs/load_dataset.gin \
			configs/datasets/voxceleb1.gin \
			--module_list configs/imports \
			--project_name 8-eval_vs_no_eval \
			--experiment_name "eval" \
        --mods "tasks.models.S3PRLUpstreamMLPDownstreamForCls.frozen_upstream=False"

ginpipe configs/default_parameters.gin \
			configs/train.gin \
			configs/load_dataset.gin \
			configs/datasets/voxceleb1.gin \
			--module_list configs/imports \
			--project_name 8-eval_vs_no_eval \
			--experiment_name "eval" \
        --mods "tasks.models.S3PRLUpstreamMLPDownstreamForCls.frozen_upstream=True"