# download dataset
python download_dataset.py --dataset_name CUFED
python download_dataset.py --dataset_name CUFED5
rm -rf data/CUFED/*.zip data/CUFED5.zip data/__MACOS__

# download pre-trained model
python download_pretrained_model.py

# texture swapping
python offline_texture_swapping.py --dataroot data/CUFED

# training
python train.py --use_weights
