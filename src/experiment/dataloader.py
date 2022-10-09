from pprint import pprint
import torch
from dataset import WikiartDataset

def collate_fn(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, object_lists, captions = zip(*data)

    images = torch.stack(images, 0)

    caption_lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(caption_lengths)).long()

    for i, cap in enumerate(captions):
        end = caption_lengths[i]
        targets[i, :end] = cap[:end]

    object_list_num = [len(object_list) for object_list in object_lists]
    max_word_num = 0
    for object_list in object_lists:
        for word_list in object_list:
            if len(word_list) > max_word_num: max_word_num = len(word_list)

    input_objects = torch.zeros(len(object_lists), max(object_list_num), max_word_num).long()

    for i, object_list in enumerate(object_lists):
        object_list_end = object_list_num[i]
        input_objects[i, :end] = object_list[:object_list_end]

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