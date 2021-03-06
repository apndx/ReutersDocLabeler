# ReutersDocLabeler
This project is about topic classification on the Reuters corpus. Some of the project code is in Jupyter notebooks, and some parts are in .py files to easily be able to run scripts in Puhti environment.

## Notebooks

Initial data loading and preprocessing is only as a notebook, also explorative data-analysis and result analysis is also done in notebooks.

### Load and preprocess data

In some steps files input.csv and test.csv are needed. These can be produced by running load-data.ipynb. In the notebooks folder there should be REUTERS_CORPUS_2 folder to create model development data, and reuters-test-data folder to create final test data. These folders have the reuter documents as zipped xml-files. In the notebook, there is constant 'TESTING = True/False' to indicate which data is loaded and processed.

Running load-data.ipynb will result in input.csv or test.csv being created to the 'reuters-csv' folder.


##  Script commands

To run scripts, first go to folder ReutersDocLabeler.

### Making a data loader

You should have inputs.csv in the folder reuters-csv. You should have three arguments: The path and file for the script, a name for the loaderset, and 'train' or 'test' to choose if the loader is for train or test. For example like this:

```
python notebooks/csv-to-dataloader.py full-data-loader train
``` 

Running [csv-to-dataloader.py](https://github.com/apndx/ReutersDocLabeler/blob/main/notebooks/csv-to-dataloader.py) with train will result in three dataloaders (train/dev/test) that are saved in the data-loaders folder. If the script is run with test, only test loader is created. Trainloader shuffles data, but dev and test loaders keep the row order.

### Training

You should have four arguments: the first defining the folder/file of the train script, the second the dataloader name that is used, the third should be a name for the model, and the fourt the amount of epocs.
For example like this:

```
python notebooks/train.py train-full-data-loader model_2 4
``` 

Running [train.py](https://github.com/apndx/ReutersDocLabeler/blob/main/notebooks/train.py) will result in as many model files as there are epocs, and the models are saved in the models folder. The model files are approximately 418 MB each. Training losses and scores are also saved in the scores folder.

### Validating

You should have three arguments: the first defining the folder/file of the validation script, the second the dataloader name that is used, and the third the model that is validated. For example like this:

```
python3 notebooks/validate.py dev-full-data-loader model_2_epoch_2.pt
```

Running [validate.py](https://github.com/apndx/ReutersDocLabeler/blob/main/notebooks/validate.py) will result in losses, scores and totals files that are saved in the scores folder.

### Testing

There should be test.csv in the reuter-csv folder, test dataloader in the data-loaders folder and model in the models folder.

You should have three arguments: the first defining the folder/file of the test script, the second the dataloader name that is used, and the third the model that is tested. For example like this:

```
python3 notebooks/test.py test-data-loader model_2_epoch_2.pt
```

Running [test.py](https://github.com/apndx/ReutersDocLabeler/blob/main/notebooks/test.py) will result in a test result CSV file that is saved in the scores folder.
