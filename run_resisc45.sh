# This script runs training, testing for RESISC45

cd ./ODML-Swin-Transformer/
python main.py --cfg configs/swin/swin_tiny_patch4_window7_224_resisc45.yaml --resume /afs/ece.cmu.edu/usr/ashwinve/Public/golden_resisc45.pth --data-path ./data/RESISC45/ --zip --cache-mode no --output /afs/ece.cmu.edu/usr/ashwinve/Private/output/
# python main.py --cfg configs/swin/swin_tiny_patch4_window7_224_resisc45.yaml --resume /afs/ece.cmu.edu/usr/ashwinve/Private/output/saved_resisc45/ckpt_epoch_29_IN_Author.pth --data-path ./data/RESISC45/ --zip --cache-mode no --output /afs/ece.cmu.edu/usr/ashwinve/Private/output/
# python main.py --cfg configs/swin/swin_tiny_patch4_window7_224_resisc45.yaml --resume /afs/andrew.cmu.edu/usr20/ashwinve/Documents/GitHub/ODML-SwinT-JetNano/ODML-Swin-Transformer/pretrained/swin_tiny_patch4_window7_224_resisc45.pth --data-path ./data/RESISC45/ --zip --cache-mode no --output /afs/ece.cmu.edu/usr/ashwinve/Private/output/
