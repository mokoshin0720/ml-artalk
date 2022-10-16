from pprint import pprint
import torch
from dataset import WikiartDataset

MAX_OBJECT_NUM = 5

def collate_fn(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, object_lists, captions = zip(*data)

    images = torch.stack(images, 0)

    caption_lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(caption_lengths)).long()

    for i, cap in enumerate(captions):
        end = caption_lengths[i]
        targets[i, :end] = cap[:end]

    object_nums = [len(obj) for obj in object_lists]
    object_nums = [len(obj) if len(obj) <= MAX_OBJECT_NUM else MAX_OBJECT_NUM for obj in object_lists]
    input_objects = torch.zeros(len(object_lists), MAX_OBJECT_NUM).long()

    for i, obj in enumerate(object_lists):
        if i+1 >= MAX_OBJECT_NUM: break
        end = object_nums[i]
        input_objects[i, :end] = torch.tensor(obj[:end])

    return images, input_objects, targets, caption_lengths

def get_loader(root_dir, wikiart_df, idx2object_df, vocab, transform, batch_size, shuffle, num_workers):
    wikiart = WikiartDataset(
        root_dir=root_dir,
        wikiart_df=wikiart_df,
        idx2object_df=idx2object_df,
        vocab=vocab,
        transform=transform
    )

    data_loader = torch.utils.data.DataLoader(
        dataset=wikiart,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return data_loader