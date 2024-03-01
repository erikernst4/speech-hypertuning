def process_classes(state, input, output='class_map'):
    mapping = {u: i for i,u in enumerate(state['dataset_metadata'][input].unique())}
    state['dataset_metadata']['class_id'] = state['dataset_metadata'][input].apply(lambda x: mapping[x])
    state[output]=mapping
    return state