cd ./ODML-Swin-Transformer/
python main.py --eval --cfg configs/swin/swin_tiny_patch4_window7_224.yaml --resume ./pretrained/swin_tiny_patch4_window7_224.pth --data-path ./data/imagenet/zipped_archives/ --zip --cache-mode no
