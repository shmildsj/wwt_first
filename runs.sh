# This file is for mage_planetoid.py to tune hyperparameters
# ogbl-ppa lr=0.005, dropout=0.5
# ogbl-ppa lr=0.01 dropout=0.0
# ogbl-ppa lr=0.01 dropout=0.0
# Key hyper-parameters for encoder is num_layers 2 {2,3} hidden_channels 128 {64, 128, 256} keep_prob 0.8 {0.1, 0.2, ..., 0.9};
# Key hyper-parameters for decoder is decode_layers 2 {2,3,4} decode_channels 128 {128, 256, 512, 1024} lr 0.01 {0.01, 0.01, 0.01, 0.1}

python -u s2gae_nc_acc.py --lr 0.01 --device 3 --dataset Pubmed --num_layers 2 --dropout 0.5 --mask_type dm --mask_ratio 0.5 --decode_layers 2 --decode_channels 1024 > result_svm/s2gae_GCN_nc_dm_Pubmed_layer2_hid128_lr-0.01_drop-0.5_ratio0.5_de-layer2_hid1024_final.txt
python -u s2gae_nc_acc.py --lr 0.01 --device 3 --dataset Pubmed --num_layers 2 --dropout 0.5 --mask_type dm --mask_ratio 0.5 --decode_layers 2 --decode_channels 128 > result_svm/s2gae_GCN_nc_dm_Pubmed_layer2_hid128_lr-0.01_drop-0.5_ratio0.5_de-layer2_hid128_final.txt
python -u s2gae_nc_acc.py --lr 0.01 --device 3 --dataset Pubmed --num_layers 2 --dropout 0.5 --mask_type dm --mask_ratio 0.5 --decode_layers 2 --decode_channels 256 > result_svm/s2gae_GCN_nc_dm_Pubmed_layer2_hid128_lr-0.01_drop-0.5_ratio0.5_de-layer2_hid256_final.txt
python -u s2gae_nc_acc.py --lr 0.01 --device 3 --dataset Pubmed --num_layers 2 --dropout 0.5 --mask_type dm --mask_ratio 0.5 --decode_layers 2 --decode_channels 512 > result_svm/s2gae_GCN_nc_dm_Pubmed_layer2_hid128_lr-0.01_drop-0.5_ratio0.5_de-layer2_hid512_final.txt
python -u s2gae_nc_acc.py --lr 0.01 --device 3 --dataset Pubmed --num_layers 2 --dropout 0.5 --mask_type dm --mask_ratio 0.6 --decode_layers 2 --decode_channels 1024 > result_svm/s2gae_GCN_nc_dm_Pubmed_layer2_hid128_lr-0.01_drop-0.5_ratio0.6_de-layer2_hid1024_final.txt
python -u s2gae_nc_acc.py --lr 0.01 --device 3 --dataset Pubmed --num_layers 2 --dropout 0.5 --mask_type dm --mask_ratio 0.6 --decode_layers 2 --decode_channels 128 > result_svm/s2gae_GCN_nc_dm_Pubmed_layer2_hid128_lr-0.01_drop-0.5_ratio0.6_de-layer2_hid128_final.txt
python -u s2gae_nc_acc.py --lr 0.01 --device 3 --dataset Pubmed --num_layers 2 --dropout 0.5 --mask_type dm --mask_ratio 0.6 --decode_layers 2 --decode_channels 256 > result_svm/s2gae_GCN_nc_dm_Pubmed_layer2_hid128_lr-0.01_drop-0.5_ratio0.6_de-layer2_hid256_final.txt
python -u s2gae_nc_acc.py --lr 0.01 --device 3 --dataset Pubmed --num_layers 2 --dropout 0.5 --mask_type dm --mask_ratio 0.6 --decode_layers 2 --decode_channels 512 > result_svm/s2gae_GCN_nc_dm_Pubmed_layer2_hid128_lr-0.01_drop-0.5_ratio0.6_de-layer2_hid512_final.txt
python -u s2gae_nc_acc.py --lr 0.01 --device 3 --dataset Pubmed --num_layers 2 --dropout 0.5 --mask_type dm --mask_ratio 0.7 --decode_layers 2 --decode_channels 1024 > result_svm/s2gae_GCN_nc_dm_Pubmed_layer2_hid128_lr-0.01_drop-0.5_ratio0.7_de-layer2_hid1024_final.txt
python -u s2gae_nc_acc.py --lr 0.01 --device 3 --dataset Pubmed --num_layers 2 --dropout 0.5 --mask_type dm --mask_ratio 0.7 --decode_layers 2 --decode_channels 128 > result_svm/s2gae_GCN_nc_dm_Pubmed_layer2_hid128_lr-0.01_drop-0.5_ratio0.7_de-layer2_hid128_final.txt
python -u s2gae_nc_acc.py --lr 0.01 --device 3 --dataset Pubmed --num_layers 2 --dropout 0.5 --mask_type dm --mask_ratio 0.7 --decode_layers 2 --decode_channels 256 > result_svm/s2gae_GCN_nc_dm_Pubmed_layer2_hid128_lr-0.01_drop-0.5_ratio0.7_de-layer2_hid256_final.txt
python -u s2gae_nc_acc.py --lr 0.01 --device 3 --dataset Pubmed --num_layers 2 --dropout 0.5 --mask_type dm --mask_ratio 0.7 --decode_layers 2 --decode_channels 512 > result_svm/s2gae_GCN_nc_dm_Pubmed_layer2_hid128_lr-0.01_drop-0.5_ratio0.7_de-layer2_hid512_final.txt
python -u s2gae_nc_acc.py --lr 0.01 --device 3 --dataset Pubmed --num_layers 2 --dropout 0.5 --mask_type dm --mask_ratio 0.8 --decode_layers 2 --decode_channels 1024 > result_svm/s2gae_GCN_nc_dm_Pubmed_layer2_hid128_lr-0.01_drop-0.5_ratio0.8_de-layer2_hid1024_final.txt
python -u s2gae_nc_acc.py --lr 0.01 --device 3 --dataset Pubmed --num_layers 2 --dropout 0.5 --mask_type dm --mask_ratio 0.8 --decode_layers 2 --decode_channels 128 > result_svm/s2gae_GCN_nc_dm_Pubmed_layer2_hid128_lr-0.01_drop-0.5_ratio0.8_de-layer2_hid128_final.txt
python -u s2gae_nc_acc.py --lr 0.01 --device 3 --dataset Pubmed --num_layers 2 --dropout 0.5 --mask_type dm --mask_ratio 0.8 --decode_layers 2 --decode_channels 256 > result_svm/s2gae_GCN_nc_dm_Pubmed_layer2_hid128_lr-0.01_drop-0.5_ratio0.8_de-layer2_hid256_final.txt
python -u s2gae_nc_acc.py --lr 0.01 --device 3 --dataset Pubmed --num_layers 2 --dropout 0.5 --mask_type dm --mask_ratio 0.8 --decode_layers 2 --decode_channels 512 > result_svm/s2gae_GCN_nc_dm_Pubmed_layer2_hid128_lr-0.01_drop-0.5_ratio0.8_de-layer2_hid512_final.txt