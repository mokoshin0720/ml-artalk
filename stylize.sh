content_root="data/imagenet/ImageNet-LVIS/"
output_root="data/imagenet/ImageNet-LVIS-stylized/"
for dir_name in $(ls ./data/imagenet/ImageNet-LVIS); do
    echo $dir_name
    make stylize filename=src/stylize-datasets/stylize.py contentdir=$content_root$dir_name outputdir=$output_root$dir_name
done

# content_root="data/objects365/val/images/"
# output_root="data/objects365/val/images-stylized/"
# for dir_name in $(ls ./data/objects365/val/images); do
#     echo $dir_name
#     make run filename=src/stylize-datasets/stylize.py contentdir=$content_root$dir_name outputdir=$output_root$dir_name
# done

# content_root="data/oid/images/"
# output_root="data/oid/images-stylized/"
# for dir_name in $(ls ./data/oid/images); do
#     echo $dir_name
#     make run filename=src/stylize-datasets/stylize.py contentdir=$content_root$dir_name outputdir=$output_root$dir_name
# done
