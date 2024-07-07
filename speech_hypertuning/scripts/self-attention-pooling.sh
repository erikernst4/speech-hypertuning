ginpipe configs/default_parameters.gin \
			configs/train.gin \
			configs/load_dataset.gin \
			configs/load_wavs.gin \
			configs/datasets/voxceleb1.gin \
			--module_list configs/imports \
			--project_name 11-self-attention-pooling \
			--experiment_name "1_head-wo_scaling" \
			    --mods "tasks.models.S3PRLUpstreamMLPDownstreamForCls.pooling_layer=@tasks.models.SelfAttentionPooling"


ginpipe configs/default_parameters.gin \
			configs/train.gin \
			configs/load_dataset.gin \
			configs/load_wavs.gin \
			configs/datasets/voxceleb1.gin \
			--module_list configs/imports \
			--project_name 11-self-attention-pooling \
			--experiment_name "1_head-w_scaling" \
			    --mods "tasks.models.S3PRLUpstreamMLPDownstreamForCls.pooling_layer=@tasks.models.SelfAttentionPooling" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalize_upstream_embeddings=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalization_method='dataset_scaling'" "tasks.models.calculate_dataset_upstream_mean_and_std_from_wavs.saving_path='/home/eernst/Voxceleb1/wavlm'" "tasks.models.calculate_dataset_upstream_mean_and_std_from_wavs.model_cls=@tasks.models.S3PRLUpstreamMLPDownstreamForCls"


ginpipe configs/default_parameters.gin \
			configs/train.gin \
			configs/load_dataset.gin \
			configs/load_wavs.gin \
			configs/datasets/voxceleb1.gin \
			--module_list configs/imports \
			--project_name 11-self-attention-pooling \
			--experiment_name "1_head-w_scaling-w_pos_encoding" \
			    --mods "tasks.models.PositionalEncoding.dropout=0.0" "tasks.models.SelfAttentionPooling.use_positional_encoding=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.pooling_layer=@tasks.models.SelfAttentionPooling" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalize_upstream_embeddings=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalization_method='dataset_scaling'" "tasks.models.calculate_dataset_upstream_mean_and_std_from_wavs.saving_path='/home/eernst/Voxceleb1/wavlm'" "tasks.models.calculate_dataset_upstream_mean_and_std_from_wavs.model_cls=@tasks.models.S3PRLUpstreamMLPDownstreamForCls"

ginpipe configs/default_parameters.gin \
			configs/train.gin \
			configs/load_dataset.gin \
			configs/load_wavs.gin \
			configs/datasets/voxceleb1.gin \
			--module_list configs/imports \
			--project_name 11-self-attention-pooling \
			--experiment_name "1_head-w_scaling-w_pos_encoding_+_dropout" \
			    --mods "tasks.models.SelfAttentionPooling.dropout=0.1" "tasks.models.SelfAttentionPooling.use_positional_encoding=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.pooling_layer=@tasks.models.SelfAttentionPooling" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalize_upstream_embeddings=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalization_method='dataset_scaling'" "tasks.models.calculate_dataset_upstream_mean_and_std_from_wavs.saving_path='/home/eernst/Voxceleb1/wavlm'" "tasks.models.calculate_dataset_upstream_mean_and_std_from_wavs.model_cls=@tasks.models.S3PRLUpstreamMLPDownstreamForCls"


ginpipe configs/default_parameters.gin \
			configs/train.gin \
			configs/load_dataset.gin \
			configs/load_wavs.gin \
			configs/datasets/voxceleb1.gin \
			--module_list configs/imports \
			--project_name 11-self-attention-pooling \
			--experiment_name "1_head-w_scaling-wo_pos_encoding-w_dropout" \
			    --mods "tasks.models.SelfAttentionPooling.dropout=0.1" "tasks.models.SelfAttentionPooling.use_positional_encoding=False" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.pooling_layer=@tasks.models.SelfAttentionPooling" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalize_upstream_embeddings=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalization_method='dataset_scaling'" "tasks.models.calculate_dataset_upstream_mean_and_std_from_wavs.saving_path='/home/eernst/Voxceleb1/wavlm'" "tasks.models.calculate_dataset_upstream_mean_and_std_from_wavs.model_cls=@tasks.models.S3PRLUpstreamMLPDownstreamForCls"


ginpipe configs/default_parameters.gin \
			configs/train.gin \
			configs/load_dataset.gin \
			configs/load_wavs.gin \
			configs/datasets/voxceleb1.gin \
			--module_list configs/imports \
			--project_name 11-self-attention-pooling \
			--experiment_name "8_heads-w_scaling-w_pos_encoding_+_dropout" \
			    --mods "tasks.models.SelfAttentionPooling.num_heads=8" "tasks.models.SelfAttentionPooling.dropout=0.1" "tasks.models.SelfAttentionPooling.use_positional_encoding=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.pooling_layer=@tasks.models.SelfAttentionPooling" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalize_upstream_embeddings=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalization_method='dataset_scaling'" "tasks.models.calculate_dataset_upstream_mean_and_std_from_wavs.saving_path='/home/eernst/Voxceleb1/wavlm'" "tasks.models.calculate_dataset_upstream_mean_and_std_from_wavs.model_cls=@tasks.models.S3PRLUpstreamMLPDownstreamForCls"

ginpipe configs/default_parameters.gin \
			configs/train.gin \
			configs/load_dataset.gin \
			configs/load_wavs.gin \
			configs/datasets/voxceleb1.gin \
			--module_list configs/imports \
			--project_name 11-self-attention-pooling \
			--experiment_name "16_heads-w_scaling-w_pos_encoding_+_dropout" \
			    --mods "tasks.models.SelfAttentionPooling.num_heads=16" "tasks.models.SelfAttentionPooling.dropout=0.1" "tasks.models.SelfAttentionPooling.use_positional_encoding=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.pooling_layer=@tasks.models.SelfAttentionPooling" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalize_upstream_embeddings=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalization_method='dataset_scaling'" "tasks.models.calculate_dataset_upstream_mean_and_std_from_wavs.saving_path='/home/eernst/Voxceleb1/wavlm'" "tasks.models.calculate_dataset_upstream_mean_and_std_from_wavs.model_cls=@tasks.models.S3PRLUpstreamMLPDownstreamForCls"


ginpipe configs/default_parameters.gin \
			configs/train.gin \
			configs/load_dataset.gin \
			configs/load_wavs.gin \
			configs/datasets/voxceleb1.gin \
			--module_list configs/imports \
			--project_name 11-self-attention-pooling \
			--experiment_name "32_heads-w_scaling-w_pos_encoding_+_dropout" \
			    --mods "tasks.models.SelfAttentionPooling.num_heads=32" "tasks.models.SelfAttentionPooling.dropout=0.1" "tasks.models.SelfAttentionPooling.use_positional_encoding=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.pooling_layer=@tasks.models.SelfAttentionPooling" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalize_upstream_embeddings=True" "tasks.models.S3PRLUpstreamMLPDownstreamForCls.normalization_method='dataset_scaling'" "tasks.models.calculate_dataset_upstream_mean_and_std_from_wavs.saving_path='/home/eernst/Voxceleb1/wavlm'" "tasks.models.calculate_dataset_upstream_mean_and_std_from_wavs.model_cls=@tasks.models.S3PRLUpstreamMLPDownstreamForCls"
