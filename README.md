# Impression generation of painting

## model
1. CNN-LSTM
- normal
```
====================================================================================================
Layer (type:depth-idx)                             Output Shape              Param #
====================================================================================================
CNNLSTM                                            [93, 513]                 --
├─Encoder: 1-1                                     --                        --
│    └─Sequential: 2-1                             [4, 2048, 1, 1]           --
│    │    └─Conv2d: 3-1                            [4, 64, 112, 112]         9,408
│    │    └─BatchNorm2d: 3-2                       [4, 64, 112, 112]         128
│    │    └─ReLU: 3-3                              [4, 64, 112, 112]         --
│    │    └─MaxPool2d: 3-4                         [4, 64, 56, 56]           --
│    │    └─Sequential: 3-5                        [4, 256, 56, 56]          215,808
│    │    └─Sequential: 3-6                        [4, 512, 28, 28]          2,339,840
│    │    └─Sequential: 3-7                        [4, 1024, 14, 14]         40,613,888
│    │    └─Sequential: 3-8                        [4, 2048, 7, 7]           14,964,736
│    │    └─AdaptiveAvgPool2d: 3-9                 [4, 2048, 1, 1]           --
│    └─Linear: 2-2                                 [4, 256]                  524,544
│    └─BatchNorm1d: 2-3                            [4, 256]                  512
├─Decoder: 1-2                                     --                        --
│    └─Embedding: 2-4                              [4, 32, 256]              131,328
│    └─LSTM: 2-5                                   [93, 512]                 1,576,960
│    └─Linear: 2-6                                 [93, 513]                 263,169
====================================================================================================
```
- input with object
```
====================================================================================================
Layer (type:depth-idx)                             Output Shape              Param #
====================================================================================================
Net                                                [107, 513]                --
├─Encoder: 1-1                                     --                        --
│    └─Sequential: 2-1                             [4, 2048, 1, 1]           --
│    │    └─Conv2d: 3-1                            [4, 64, 112, 112]         9,408
│    │    └─BatchNorm2d: 3-2                       [4, 64, 112, 112]         128
│    │    └─ReLU: 3-3                              [4, 64, 112, 112]         --
│    │    └─MaxPool2d: 3-4                         [4, 64, 56, 56]           --
│    │    └─Sequential: 3-5                        [4, 256, 56, 56]          215,808
│    │    └─Sequential: 3-6                        [4, 512, 28, 28]          2,339,840
│    │    └─Sequential: 3-7                        [4, 1024, 14, 14]         40,613,888
│    │    └─Sequential: 3-8                        [4, 2048, 7, 7]           14,964,736
│    │    └─AdaptiveAvgPool2d: 3-9                 [4, 2048, 1, 1]           --
│    └─Linear: 2-2                                 [4, 256]                  524,544
│    └─BatchNorm1d: 2-3                            [4, 256]                  512
│    └─Embedding: 2-4                              [4, 5, 256]               131,328
├─Decoder: 1-2                                     --                        --
│    └─Embedding: 2-5                              [4, 53, 1536]             787,968
│    └─LSTM: 2-6                                   [107, 512]                4,198,400
│    └─Linear: 2-7                                 [107, 513]                263,169
====================================================================================================
Total params: 64,049,729
Trainable params: 64,049,729
Non-trainable params: 0
Total mult-adds (G): 276.09
====================================================================================================
Input size (MB): 2.41
Forward/backward pass size (MB): 1447.00
Params size (MB): 256.20
Estimated Total Size (MB): 1705.61
====================================================================================================
```

2. Show-Attend-Tell
- normal
```
====================================================================================================
Layer (type:depth-idx)                             Output Shape              Param #
====================================================================================================
ShowAttendTell                                     [4, 32, 513]              --
├─Encoder: 1-1                                     --                        --
│    └─Sequential: 2-1                             [4, 2048, 7, 7]           --
│    │    └─Conv2d: 3-1                            [4, 64, 112, 112]         (9,408)
│    │    └─BatchNorm2d: 3-2                       [4, 64, 112, 112]         (128)
│    │    └─ReLU: 3-3                              [4, 64, 112, 112]         --
│    │    └─MaxPool2d: 3-4                         [4, 64, 56, 56]           --
│    │    └─Sequential: 3-5                        [4, 256, 56, 56]          (215,808)
│    │    └─Sequential: 3-6                        [4, 512, 28, 28]          2,339,840
│    │    └─Sequential: 3-7                        [4, 1024, 14, 14]         40,613,888
│    │    └─Sequential: 3-8                        [4, 2048, 7, 7]           14,964,736
│    └─AdaptiveAvgPool2d: 2-2                      [4, 2048, 14, 14]         --
├─DecoderWithAttention: 1-2                        --                        --
│    └─Embedding: 2-3                              [4, 33, 512]              262,656
│    └─Linear: 2-4                                 [4, 512]                  1,049,088
│    └─Linear: 2-5                                 [4, 512]                  1,049,088
│    └─Attention: 2-6                              [4, 2048]                 --
│    │    └─Linear: 3-9                            [4, 196, 512]             1,049,088
│    │    └─Linear: 3-10                           [4, 512]                  262,656
│    │    └─ReLU: 3-11                             [4, 196, 512]             --
│    │    └─Linear: 3-12                           [4, 196, 1]               513
│    │    └─Softmax: 3-13                          [4, 196]                  --
│    └─Linear: 2-7                                 [4, 2048]                 1,050,624
│    └─Sigmoid: 2-8                                [4, 2048]                 --
│    └─LSTMCell: 2-9                               [4, 512]                  6,295,552
│    └─Dropout: 2-10                               [4, 512]                  --
│    └─Linear: 2-11                                [4, 513]                  263,169
====================================================================================================
```
- input with object