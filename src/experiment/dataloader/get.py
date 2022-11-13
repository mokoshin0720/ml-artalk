import torch
from experiment.dataloader.collate_fn import collate_normal, collate_with_object

def get_loader(dataset, conf):
    if conf['model_name'] in conf['normal_models']:
        collate_fn = collate_normal
    elif conf['model_name'] in conf['word_object_models']:
        collate_fn = collate_with_object
    else:
        assert 'Invalid model name from get_loader'
    
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=conf['batch_size'],
        shuffle=conf['shuffle'],
        num_workers=conf['num_workers'],
        collate_fn=collate_fn,
    )