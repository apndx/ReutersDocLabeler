# ReutersDocLabeler
This project is about topic classification on the Reuters corpus.


##  Script commands

To run scripts, first go to folder ReutersDocLabeler.

### Making a data loader

You should have inputs.csv in the folder reuters-csv. You should have two arguments: The path and file for the script, and a name for the loaderset. For example like this:

```
python notebooks/csv-to-dataloader.py full-data-loader
``` 

Running csv-to-dataloader.py will result in three dataloaders (train/dev/test) that are saved in the data-loaders folder.
### Training

You should have four arguments: the first defining the folder/file of the train script, the second the dataloader name that is used, the third should be a name for the model, and the fourt the amount of epocs.
For example like this:

```
python notebooks/train.py train-full-data-loader model_2 4
``` 

Running train.py will result in as many model files as there are epocs, and the models are saved in the models folder. The model files are approximately 418 MB each. Training losses and scores are also saved in the scores folder.

### Testing

You should have three arguments: the first defining the folder/file of the test script, the second the dataloader name that is used, and the third the model that is tested. For example like this:

```
python3 notebooks/test.py dev-full-data-loade model_2_epoch_2.pt
```

Running test.py will result in losses and scores files that are saved in the scores folder.
