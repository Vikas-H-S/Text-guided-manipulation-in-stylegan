Refer the following steps to run the code

1. You would required a Python>=3.8 and PyTorch compatible CUDA>=11.8

2. To install PyTorch execute the below command based on your CUDA version:
	For CUDA 11.8 - pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	For CUDA 12.1 - pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
	For CUDA 12.4 - pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

3. To process textual prompts, manipulate StyleGAN images and visualize results install all the dependencies by executing this command - pip install ftfy regex tqdm pyyaml matplotlib pillow==6.2.2

4. To install OpenAI's CLIP library execute this command - pip install git+https://github.com/openai/CLIP.git

5. Then download the precomputed pairs and pretrained models from this google drive link and place all the folders in your directory - https://drive.google.com/uc?export=download&id=1wkCJxUf7YD-ov70DCZaN8GRUoKsKZJOs

6. To train the Multi2One dictionary for a specific dataset execute the below command - python train_dictionary.py --dataset <dataset_name>
	Replace <dataset_name> with the dataset you want to use (e.g., ffhq, church). This creates a new dictionary at dictionary/<dataset_name>/multi2one_new.pt.


7. To generate manipulated images using the trained dictionary, you need to edit the config.yaml file to set manipulation parameters (e.g., text prompts, alpha values, etc.), then execute the below command - python generate_samples.py --dataset <dataset_name>
	Replace <dataset_name> with the desired dataset (e.g., ffhq, church). And the generated images will be saved in the logs/ directory.