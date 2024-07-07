ginpipe configs/default_parameters.gin \
			configs/train.gin \
			configs/load_dataset.gin \
			configs/load_wavs.gin \
			configs/datasets/voxceleb1.gin \
			--module_list configs/imports \
			--project_name 12-alternative-attention-poolings \
			--experiment_name "summary_mixing_lite" \
			    --mods "tasks.models.SummaryMixingPooling.mode='SummaryMixing-lite'" "tasks.models.SummaryMixingPooling.dropout=0.1" "tasks.models.SummaryMixingPooling.num_heads=16" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.pooling_layer=@tasks.models.SummaryMixingPooling" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalize_upstream_embeddings=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalization_method='dataset_scaling'" "tasks.models.calculate_dataset_upstream_mean_and_std_from_wavs.saving_path='/home/eernst/Voxceleb1/wavlm'" "tasks.models.calculate_dataset_upstream_mean_and_std_from_wavs.model_cls=@tasks.models.S3PRLUpstreamMLPDownstreamForCls"

ginpipe configs/default_parameters.gin \
			configs/train.gin \
			configs/load_dataset.gin \
			configs/load_wavs.gin \
			configs/datasets/voxceleb1.gin \
			--module_list configs/imports \
			--project_name 12-alternative-attention-poolings \
			--experiment_name "summary_mixing" \
			    --mods "tasks.models.SummaryMixingPooling.mode='SummaryMixing'" "tasks.models.SummaryMixingPooling.dropout=0.1" "tasks.models.SummaryMixingPooling.num_heads=16" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.pooling_layer=@tasks.models.SummaryMixingPooling" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalize_upstream_embeddings=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalization_method='dataset_scaling'" "tasks.models.calculate_dataset_upstream_mean_and_std_from_wavs.saving_path='/home/eernst/Voxceleb1/wavlm'" "tasks.models.calculate_dataset_upstream_mean_and_std_from_wavs.model_cls=@tasks.models.S3PRLUpstreamMLPDownstreamForCls"

ginpipe configs/default_parameters.gin \
			configs/train.gin \
			configs/load_dataset.gin \
			configs/load_wavs.gin \
			configs/datasets/voxceleb1.gin \
			--module_list configs/imports \
			--project_name 12-alternative-attention-poolings \
			--experiment_name "transformer" \
			    --mods "tasks.models.TransformerPooling.num_heads=16" "tasks.models.TransformerPooling.dropout=0.1" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.pooling_layer=@tasks.models.TransformerPooling" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalize_upstream_embeddings=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalization_method='dataset_scaling'" "tasks.models.calculate_dataset_upstream_mean_and_std_from_wavs.saving_path='/home/eernst/Voxceleb1/wavlm'" "tasks.models.calculate_dataset_upstream_mean_and_std_from_wavs.model_cls=@tasks.models.S3PRLUpstreamMLPDownstreamForCls"