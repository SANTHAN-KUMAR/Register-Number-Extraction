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


## Setting up the project on your PC :

### What You’ll Need
- **Computer**: Windows, Linux, or macOS.
- **Python**: Version 3.8 to 3.13 (we recommend 3.10, but newer ones like 3.12 work too).
- **Images**: PNG files with 12-digit numbers (e.g., `21222xxxxxx.png`).
- ### NOTE : All the Register Numbers are under the folder data/Register Numbers, if you face any file not found errors pls verify your path!
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
