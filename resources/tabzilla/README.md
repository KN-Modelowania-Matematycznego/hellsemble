# TabZilla Integration

This code is sourced from [TabZilla GitHub repository](https://github.com/naszilla/tabzilla).

It includes a few separated functions required to download and preprocess data from OpenML.

## Usage Instructions

### Download and Preprocess All Datasets

To download and preprocess all datasets, run the following command from the TabZilla folder:

```bash
python tabzilla_data_preprocessing.py --process_all
```

This will download all datasets and write a preprocessed version of each to a local directory:

```
tabzilla/datasets/<dataset_name>
```

### Download and Preprocess a Single Dataset

To download and preprocess a single dataset, run the following command from the root of the repo folder:

```bash
python tabzilla_data_preprocessing.py --dataset_name <dataset_name>
```

#### Example

The following command will download and preprocess the dataset `openml__california__361089`:

```bash
python tabzilla_data_preprocessing.py --dataset_name openml__california__361089
```

### Transforming to usable format

Once the tabzilla data is downloaded we need to transform it to suite our testing framework format, this can be done by the script

```
python transform_tabzilla.py
```
*Note*

 - Make sure to check for binary classification data. 
 - Make sure to check if the paths are correct in the file.