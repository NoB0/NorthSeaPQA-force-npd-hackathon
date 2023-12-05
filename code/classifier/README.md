# Passage classification

The goal of this task is to classify passages from documents related to the oil and gas domain into a set of predefined categories.

## Data

The default data is in the file `data/classification/annotates_good_text.xlsm`. It contains a set of +1,000 passages annotated by one expert.
You can use other data as long as it is an Excel file with at least the following columns:

  - `_id`: the id of the passage
  - `cat`: the category of the passage
  - `content_scrubbed_light`: the text of the passage
  - `label`: the label of the passage

The default labels are:

  - B: biostratigraphy
  - DD: daily drilling report
  - L: lithostratigraphy
  - G: geological description
  - D: drilling description
  - PP: petrophysical description
  - GC: geochemical description
  - R: rubbish

## Training

To train a classifier with defaults parameters, run the following command:

```bash
python -m code.classifier.train_passage_classifier 
```

For more information on the parameters, run:

```bash
python -m code.classifier.train_passage_classifier --help
```

At the end of the training, the model is saved locally and can be used for inference later.
The predictions on the test set are also saved locally in the file `data/classification/test_predictions.csv`.

## Inference

For inference, you can use the notebook `code/classifier/use_passage_classifier.ipynb`.
It loads the model and can be used to predict the category of passages in an interactive way.
