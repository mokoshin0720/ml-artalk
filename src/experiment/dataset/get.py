from torchvision import transforms
from experiment.dataset.classes import WikiartDataset, WikiartDatasetWithObject
from experiment.train.config import get_conf
from experiment.utils.vocab import Vocabulary

def get_dataset(conf: dict, is_train: bool):
    transform = transforms.Compose([ 
        transforms.RandomCrop(conf['crop_size']),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    wikiart_df = conf['train_df'] if is_train else conf['test_df']
    object_dir = conf['train_object_txt_dir'] if is_train else conf['test_object_txt_dir']
    mask_dir = conf['train_mask_txt_dir'] if is_train else conf['test_mask_txt_dir']

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
            object_dir=object_dir,
            mask_dir=mask_dir,
            vocab=conf['vocab'],
            transform=transform
        )
    else:
        assert 'Invalid model name from get_loader'
        
    return dataset

if __name__ == '__main__':
    conf = get_conf()
    train_dataset = get_dataset(conf, is_train=True)
    for i in range(10):
        print(train_dataset[i])
        print('--------------------------')