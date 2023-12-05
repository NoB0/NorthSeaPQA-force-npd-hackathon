# Generation of Question-Answer (QA) pairs

The goal of this task is to create a benchmark dataset for the development of QA systems in the oil and gas domain.

## North Sea Petroleum Question Answer (NorthSeaPQA)

NorthSeaPQA is a reading comprehension dataset tailored to the oil and gas domain.
It consists of questions posed on a set of passages from documents related to the oil and gas domain, with a focus on the North Sea region, where the answer to each question is extracted from the corresponding passage. Note that some questions may be unanswerable.

### Data curation

The dataset was created by a team of 5 people during the FORCE NPD LLM Hackathon.
The QA pairs are generated from a set of 1,000+ passages using OpenAI GPT4 model with the following prompt:

```python
system_prompt = """The user will provide you with a paragraph of text and a metadata string.
The data will follow this format:

WELLBORE NAME: 
PARAGRAPH:

The paragraphs and metadata can be in different languages. Always answer in english. They all come from the same domain of oil and gas exploration and production. 

You will create questions that can be answered from the text, alongside the two answers for each question.
The first answer should be as short and explicit as possible. Put extra weight on making it as short as possible with no filler words at all. Use the exact words and phrasing from the text.
The second answer should be answered using natural language.
The wellbore name should always be explicitly written in the questions and explicitly written in the second answer.
You may also use exact sentences or shorten the sentences to phrases to generate examples.
You may paraphrase the content in sentences or change them slightly but you must preserve both language and style of individual texts.
If there are no examples to generate then create an empty list. 
Provide a confidence score between 0 and 1 for each of the question answer pairs where 1 is full confidence in the answer, and 0 is no confidence in the answer.

You are to produce a JSON formatted output, with JSON output only in the following form:

[{"Q": "<Question 1a>", "A1": "<Answer 1b>", "A2": "<Answer 1c>", "C" : "<Confidence 1d>"}, 
 {"Q": "<Sentence 2a>", "A1": "<Sentence 2b>", "A2": "<Answer 2c>", "C" : "<Confidence 2d>"},
 ...,
 {"Q": "<Sentence Na>", "A1": "<Sentence Nb>", "A2": "<Answer Nc>", "C" : "<Confidence Nd>"}]

You are to determine yourself how many examples N you can come up with, but make as many as possible from the text provided, preferably at least 10. It is very important to make sure its always JSON formatted.
"""
```

Out of the ~11,200 generated QA pairs, ~550 were manually checked by a human annotator from which 443 were found to be of good quality.

### Data snapshot

| Split | # QA pairs |
| ----- | ---------- |
| Train | 10,630     |
| Test  | 443        |

Note that the test set contains 443 QA pairs that were manually checked by a human annotator.

### Data format

The dataset is provided in JSON format.

**Description of the data fields for one document**

| Field name | Field value | Description |
| ---------- | ----------- | ----------- |
| document   | String      | Document ID |
| metadata   | Object      | Document metadata |
| metadata.well_name | String | Well name |
| paragraphs | List        | List of passages in the document |
| paragraphs.context | String | Passage text |
| paragraphs.qas | List | List of questions and answers for the passage |
| paragraphs.qas.question | String | Question |
| paragraphs.qas.id | String | Question ID |
| paragraphs.qas.answer | Object | Answer to the question |
| paragraphs.qas.answer.text | String | Answer text |
| paragraphs.qas.answer.span | String | Answer span in the passage |
| paragraphs.qas.is_impossible | Boolean | Whether the question is answerable or not |

Below is a typical data point

```json
{
    "document": "393cdb34763616307b3e8142d5f20eeb95de0e73",
    "paragraphs": [
        {
            "context": "nutrient-providing currents. The area is considered particularly important for coral communities and fish populations (Det Kongelige Miljverndepartement, 2008-2009; MAREANO, 2011). Submarine structures made by leaking gases (seeps or pockmarks) with methane-derived authigenic carbonates (MDAC) are noteworthy as they are protected by the EC Habitats Directive. However these habitats are not described under any legislation relevant to the Norwegian Sea and were not considered high priority during the current investigation for this reason.",
            "qas": [
                {
                    "question": "What is considered important for coral communities and fish populations around the 6506/11-9 S wellbore?",
                    "id": "393cdb34763616307b3e8142d5f20eeb95de0e73_75.0_689",
                    "answer": {
                        "text": "The area around the 6506/11-9 S wellbore is considered important for coral communities and fish populations.",
                        "span": "The area"
                    },
                    "is_impossible": false
                },
                ...
            ]
        },
        ...
    ],
    "metadata": {
        "well_name": "6506/11-9"
    }
}
```

### Dataset version and Maintenance

**Version**: 1.0  
**Release date**: 2020-12-05

NorthSeaPQA is a static dataset from a specific point in time and maintenance will be limited.
If you find any issues with the dataset, please open an issue in this repository.

### Dataset cost estimation

The cost estimation is done for the newest GPT4 Turbo model available on Nov. 30, 2023.

  - Input 900 tokens *1200 rows* 0.01 \$ / per 1000 tokens = 10.8 \$
  - Output 700 tokens *1200 rows* 0.03 \$ / per 1000 tokens = 25.2 \$

For 11,200 QA pairs, the total cost is estimated to be 36 \$.

### Potential use cases

The dataset can be used to train and evaluate QA systems in the oil and gas domain. Thanks to the two types of answers (i.e., span and text) provided for each question, it can be used to train and evaluate QA systems that can provide both span and natural language answers.

## Evaluation

As part of this task, we also provide a script (`code/qa_generation/evaluate_qa.py`) to evaluate the performance of a QA system on the NorthSeaPQA dataset.
The script computes three metrics: exact match (EM), F1 score, and BLEU score. The first two metrics are computed on the span answers, while the last one is computed on the text answers using `sacrebleu`.

### Usage

To run the evaluation script, you need to provide the paths to the test file, prediction file, and optionally output file, as follows:

```bash
python -m code.qa_generation.evaluate_qa <path to the test file> <path to the prediction file> --out_file <path to the output file>
```

Note that the prediction file should be in the same format as the test file.

For more details on the arguments, you can run `python -m code.qa_generation.evaluate_qa -h`.
