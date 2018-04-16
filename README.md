# thursday

Classifying radio galaxies as Fanaroff-Riley Type I (FR-I) and Fanaroff-Riley Type II (FR-II).



## Input files

You will need
- The data used in https://arxiv.org/pdf/1705.03413.pdf. This is stored in `asu.tsv`.


## Running the code

1. Run `fri-frii-download.ipynb` to download all FRI/FRII samples in fri-cat, frii-cat, and FIRST. Data will be stored as `data.h5` in (current_directory/data/data.h5)
2. Use the get_data function from `format_data.py` to generate the training and testing data indices from `data.h5` and `asu.tsv`.
3. Open `data.h5`  as `data` and and use indices to select training and test images from `data['images']` and `data['labels']`
4. Use the `data_gen` function in `format_data.py` to construct the data generator.
5. Instantiate the`HOGNnet` and/or `SklearnModel` from `models.py`. Both take `datagen` and `seed` as arguments. `SklearnModel` takes additional arguments like `Model` (the sklearn classifier being used), `nb_augment` (factor data is increased via augmentations), as well as any number of parameters specific to `Model`. `HOGNnet` takes `batch_size`, `steps_per_epoch`, `max_epoch`, and `patience` as additional arguments.
6. Use the `fit` method to train the model. Both have `train_x` and `train_y` as inputs. (Note: For the `HOGnet` `fit` method, the the labels must first be converted to a confusion matrix using the `keras.utils.to_categorical` function)
7. Optional: If training times are long or you want to store your trained model, use the save method to save the model to disk (`SklearnModel` as a `.pk` file and `HOGnet` as a `.h5` file) 
8. Optional: Load model from disk with the `load` method.
9. Use `predict` method to predict class labels and `predict_proba` class probabilities for unseen samples (`test_x`). 


![alt text](https://github.com/josh-marsh/thursday/blob/reload/Basic/files/flow.jpg)

## Example

A working example can be found at [`example_usage.ipynb`](thursday/example_usage.ipynb). 

