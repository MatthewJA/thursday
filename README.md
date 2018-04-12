# thursday

Classifying radio galaxies as Fanaroff-Riley Type I (FR-I) and Fanaroff-Riley Type II (FR-II).


## Input files

You will need
- The data used in https://arxiv.org/pdf/1705.03413.pdf. This is stored in asu.tsv.


## Running the code

1. Run `fri-frii-download.py` to get all FRI/FRII samples in fri-cat, frii-cat, and FIRST. Data will be stored in as data.h5 in (current_directory/data/data.h5)
2. Use the get_data function in `format_data.py` to generate the training and testing data (train_x, train, y, test_x, test_y) from data.h5 and asu.tsv.
3. Use the data_gen function in `format_data.py` to create the data generator.
4. Instantiate HognetModel class in model_hognet.py. Do the following with HognetModel class methods: Use build method to construct the model, the training method to train the model, and then the load method to return the fully trained model (This has not yet been pushed due to some highly irritating tensorflow issues).
5. Instantiate desired SklearnModel class in models.py. Use train method to build and train the sklearn classifier, and then the load method to return the fully trained model. If comparing a sklearn model to hognet, use sklearn after (keras uses a generator and stops training when loss plateaus )
5. Use trained models to make predictions.


![alt text](https://github.com/josh-marsh/thursday/blob/reload/Basic/files/flow.jpg)

## Example

A fully working example can be found in test.ipynb. 

## Remarks
While our implementation is functioning properly, the scripts will not produce any meaningful classifier. Due to poorly cited data origins in certain papers, we are currently missing all bar one of the FR-II samples found in asu.tsv. To make the script work, we increased the radius dramatically when matching catalogs, adding some FRIs into the FR-II  set. 