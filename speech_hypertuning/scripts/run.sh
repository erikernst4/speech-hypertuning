ginpipe configs/default_parameters.gin \
			configs/train.gin \
			configs/load_dataset.gin \
			configs/datasets/voxceleb1.gin \
			--module_list configs/imports \
			--project_name experiment_lr \
			--experiment_name test_load_dataset
