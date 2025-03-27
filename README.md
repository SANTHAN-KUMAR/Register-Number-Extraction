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
