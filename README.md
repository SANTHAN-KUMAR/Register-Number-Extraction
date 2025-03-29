# Register-Number-Extraction
### Accuracy achieved : 58.88% ( on 1000 iterations )
### improved accuracy : 64.11% ( with 2000 iterations )
### updated accuracy : 78.88% ( with 5000 iterations and applied regex for test predictions )
#### [ NOTE : Running the training everytime gives variable accuracy with (+-)1 variation in the score]

### Note : Increasing the iterations is no more reflecting in positive accuracy score, the model seems to be overfitting.

## CUSTOM MODEL'S ACCURACY : 95.14% (without data augmentation)
### it achieved 93% with data augmentation and increased epochs (5)

## Best models to be considered in :

CRNN with just dropout.ipynb

deep CRNN TEMP (with data augmentation).ipynb


### deep CRNN TEMP (with data augmentation).ipynb
when this was trained on 70 epochs, the accuracy was around 91 and the predictions were mediocre.
after increasing the epochs to 75, the accuracy was around 92.9 and the predictions were pretty good.
 #### ( TRY TO SAVE THIS MODEL COMPLETELY )
previosuly there were some issues in saving and loading this model.

increasing the epochs to 80 resulted in 94% accuracy, but the predictions are bad.


# Setting up the project on your PC :

### What You’ll Need
- **Computer**: Windows, Linux, or macOS.
- **Python**: Version 3.8 to 3.13 (we recommend 3.10, but newer ones like 3.12 work too).
- **Images**: PNG files with 12-digit numbers (e.g., `21222xxxxxx.png`).
- ### NOTE : All the Register Numbers are under the folder ```data/Register Numbers```, if you face any file not found errors pls verify your path!
- **GPU (Optional)**: An NVIDIA graphics card with CUDA 12.1 for faster performance (e.g., RTX 4060).


### 1. Get the Code
First, download this project to your computer:
1. Open your terminal (Command Prompt on Windows, Terminal on Mac/Linux).
2. Type this command and press Enter:
   
   ```bash
   git clone https://github.com/SANTHAN-KUMAR/Register-Number-Extraction

### 2. Move into the project folder:

```bash
cd Register-Number-Extraction
```

### 3. Set Up a Space for the Code

We’ll create a “virtual environment” to keep things organized:

1. Run this command:

```bash
python -m venv env
```

2. Activate it :

```bash
env\Scripts\activate
```

### 4. Install the Tools
This project needs some Python libraries. Pick the option below based on your setup:

Option A: Regular Computer (CPU)
If you don’t have an NVIDIA GPU or just want to keep it simple:

1. Run :

```bash
pip install -r requirements.txt
```

2. Check if it worked :

```bash
python -c "import torch; print('PyTorch version:', torch.__version__); print('GPU available?', torch.cuda.is_available())"
```

It should output something like this :

```
PyTorch version: 2.5.1
GPU available? False
```

Option B: NVIDIA GPU (Faster Setup)
If you have an NVIDIA GPU with CUDA 12.1 (like the RTX Series):

1. Check your CUDA version:

```bash
nvcc --version
```

Look for “release 12.1” (e.g., Cuda compilation tools, release 12.1, V12.1.105). If it’s different (e.g., 11.8), see the “Other GPUs” note below.

2. Install the GPU version:

```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
pip install numpy>=1.26.0 pillow>=10.1.0
```

3. Check if it worked:

```bash
python -c "import torch; print('PyTorch version:', torch.__version__); print('GPU available?', torch.cuda.is_available())"
```

You should see :

```
PyTorch version: 2.5.1+cu121
GPU available? True
```

### Note for Other GPUs: If your CUDA version isn’t 12.1 (e.g., 11.8), use this instead:

```bash
pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install numpy>=1.26.0 pillow>=10.1.0
```
Find your version at PyTorch’s site.

### 4. Run the Project :

1.Start Jupyter Notebook:

```bash
jupyter notebook
```

#### A browser window will open.

#### Open the .ipynb file (deep CRNN TEMP (with data augmentation)-updated.ipynb).

#### Click “Run All” in the menu, or run each cell one-by-one:

#### Loads your images.

#### Trains the model (takes time, especially on CPU).

#### Tests it and predicts numbers from images.

#### The code picks CPU or GPU automatically, so no changes needed!


### What’s in the Project:

- ### Cells 1-2: Load and prepare your images.
- ### Cell 3: Defines the CRNN (the brain of the project).
- ### Cell 4: Trains it and saves the best version to saved_models/.
- ### Cell 5: Tests accuracy on some images.
- ### Cell 6: Predicts a number from one image (e.g., test_reg.jpeg).
