# This script runs training, testing for RESISC45

cd ./ODML-Swin-Transformer/
python main.py --cfg configs/swin/swin_tiny_patch4_window7_224_resisc45.yaml --resume ./pretrained/swin_tiny_patch4_window7_224.pth --data-path ./data/RESISC45/zipped_archives/ --zip --cache-mode no