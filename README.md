# ReutersDocLabeler
This project is about topic classification on the Reuters corpus.


###  Script commands

To run scripts, first go to folder ReutersDocLabeler.
#### Making a data loader

You should have inputs.csv in the folder reuters-csv. The data loader script wants a name to the loaderset as an argument. For example like this:

```
python notebooks/csv-to-dataloader.py full-data-310321
``` 


#### Training

You should have three arguments: the first defining the folder/file of the train script, the second the dataloader name that is used, and the third should be a name for the model file that is created. 
For example like this:

```
python notebooks/train.py mini-train_data_loader_bs48 model_2_300321
``` 
