# training deep network for detecting whale

Data preparation:

-gen_trainingpaches.py: script to automatically tile big images into small patches for training.
usage:
python gen_trainingpatches.py --root DIR_TO_ORIGINAL_IMAGES --step STEP_SIZE --size SIZE_OF_PATCHES --output OUTPUT_DIR

The folder “A” that contains the training dataset should be organized as follow:

A\train\whales contains all training images for whale

A\train\water contains all training images for water 


Training: 

python transfer_learning.py --name MODEL_NAME --data_dir TRAINING_DATA_DIR

