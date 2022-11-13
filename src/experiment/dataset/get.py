from torchvision import transforms
from experiment.dataset.classes import WikiartDataset, WikiartDatasetWithObject

def get_dataset(conf: dict, is_train: bool):
    transform = transforms.Compose([ 
        transforms.RandomCrop(conf['crop_size']),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    wikiart_df = conf['train_df'] if is_train else conf['test_df']

    if conf['model_name'] in conf['normal_models']:
        dataset = WikiartDataset(
            root_dir=conf['image_dir'],
            wikiart_df=wikiart_df,
            vocab=conf['vocab'],
            transform=transform
        )
    elif conf['model_name'] in conf['word_object_models']:
        dataset = WikiartDatasetWithObject(
            root_dir=conf['image_dir'],
            wikiart_df=wikiart_df,
            idx2object_df=conf['idx2obj_df'],
            vocab=conf['vocab'],
            transform=transform
        )
    else:
        assert 'Invalid model name from get_loader'
        
    return dataset