# Register Number Extraction

## To test the project, follow this link : 

https://colab.research.google.com/drive/1TRqyKe7JmP0dOh4_ijVkAqoEycNRgFWP?usp=sharing

## 1. Tesseract OCR Experiments

This section details the results obtained using Tesseract OCR for register number extraction.

**Initial Accuracy:** 58.88% (on 1000 iterations)

**Accuracy Improvements:**

* **Increased Iterations (2000):** Accuracy improved to 64.11%.
* **Increased Iterations (5000) and Regex Application:** Accuracy further improved to 78.88% (regex was applied to the test predictions).

**Important Notes on Tesseract Experiments:**

* **Accuracy Variability:** Running the training multiple times resulted in variable accuracy, with a fluctuation of approximately (+-) 1%.
* **Overfitting:** Increasing the number of iterations beyond 5000 did not lead to further positive improvements in accuracy, suggesting potential overfitting of the Tesseract model to the training data.

## 2. Custom Model Experiments

This section outlines the results achieved with custom-trained models for register number extraction.

**Initial Custom Model Accuracy (without Data Augmentation):** 95.14%

**Impact of Data Augmentation and Increased Epochs:**

* Applying data augmentation and increasing the number of training epochs to 5 resulted in an accuracy of 93%.

**Detailed Experiment with "deep CRNN TEMP (with data augmentation).ipynb":**

* **70 Epochs:** Achieved an accuracy of approximately 91%, with mediocre prediction quality.
* **75 Epochs:** Increased the accuracy to around 92.9%, and the prediction quality was observed to be significantly better.
* **80 Epochs:** Resulted in a higher accuracy of 94%, but the prediction quality deteriorated, indicating potential overfitting.

**Note on Model Saving and Loading:**

* Previously, there were issues encountered while saving and loading the trained models. Efforts should be made to ensure the complete and reliable saving of the best-performing model.

## 3. Best Models to Consider

Based on the experiments, the following model notebooks appear to be the most promising and should be considered for further development or deployment:

* CRNN with just dropout.ipynb
* deep CRNN TEMP (with data augmentation).ipynb


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

### Option A: Regular Computer (CPU)

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



### Option B: NVIDIA GPU (Faster Setup)
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
