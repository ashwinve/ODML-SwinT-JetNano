# This script runs training, testing for RESISC45

cd ./ODML-Swin-Transformer/
python main.py --cfg configs/swin/swin_tiny_patch4_window7_224_resisc45.yaml --resume ./pretrained/swin_tiny_patch4_window7_224_resisc45.pth --data-path ./data/RESISC45/ --zip --cache-mode no --output /afs/ece.cmu.edu/usr/ashwinve/Private/output/