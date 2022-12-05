# This script runs training, testing for RESISC45

cd ./ODML-Swin-Transformer/
python main.py --cfg configs/swin/swin_tiny_patch4_window7_224_resisc45.yaml --resume /afs/ece.cmu.edu/usr/ashwinve/Public/output/swin_tiny_patch4_window7_224_resisc45/default/L2B4_L3B0_Epochs10_Run2.pth --data-path ./data/RESISC45/ --zip --cache-mode no --output /afs/ece.cmu.edu/usr/ashwinve/Public/output/
# python main.py --cfg configs/swin/swin_tiny_patch4_window7_224_resisc45.yaml --resume /afs/ece.cmu.edu/usr/ashwinve/Public/golden_resisc45.pth --data-path ./data/RESISC45/ --zip --cache-mode no --output /afs/ece.cmu.edu/usr/ashwinve/Public/output/


