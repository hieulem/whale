# whale

Data preparation:

-gen_trainingpaches.py: script to automaticly tile big images into small patches for training.
usagE:
python gen_trainingpatches.py --root DIR_TO_ORIGINAL_IMAGES --step STEP_SIZE --size SIZE_OF_PATCHES --output OUTPUT_DIR


Training: 

python transfer_learning.py --name MODEL_NAME --data_dir TRAINING_DATA_DIR
