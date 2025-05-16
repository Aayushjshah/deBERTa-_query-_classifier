# Email and Calendar Search Query Classifier

## Overview

This project focuses on classifying search queries as either related to "email" or "calendar". It includes the dataset generation process, data preprocessing, model training, and inference. The project utilizes the DeBERTa model for classification.

**Author:** Aayush Shah
**Email:** 2001aayushshah@gmail.com
**Phone:** +91 8879090901

## File Descriptions

- **`email_events_data/`**: This folder contains the dataset of events used for generating calendar-related queries.
- **`mlruns/`**: This folder stores the logs and artifacts of experiments tracked using MLFlow.
- **`Aayush_Shah_QueryClassfier_DeBERTA_FullNotebook_15_May_2025.ipynb`**: This Jupyter Notebook provides a complete end-to-end workflow, covering:
    - Dataset generation
    - Exploratory data analysis
    - Data preprocessing steps
    - Model training using DeBERTa
    - Inference and results evaluation
- **`calendarSeeds.txt`**: This file contains a list of sub-topics related to the calendar, used for generating calendar-specific queries.
- **`emailSeeds.txt`**: This file contains a list of sub-topics related to email, used for generating email-specific queries.
- **`data_preprocessed/`**: This folder contains the preprocessed dataset that was used for training and testing the classification model.
- **`testClassifier.ipynb`**: This directly runnable Kaggle Notebook is designed to generate predictions for new input queries using the trained model.
- **`important_resources.txt`**: This file contains links to various important resources, including:
    - Kaggle hosted models
    - Kaggle hosted dataset
    - Runnable Kaggle notebooks
    - GitHub project link
- **`model/`**: This folder contains the trained model's weights and configuration files.
- **`calendar_5000_queries_generated_from events_dataset`** - 5000+ Queries generated via events_dataset
## Important Resources

- **Kaggle Testing Notebook Link:** [https://www.kaggle.com/code/shah2001aayush/testclassifier/](https://www.kaggle.com/code/shah2001aayush/testclassifier/)
- **GitHub Link:** [https://github.com/Aayushjshah/deBERTa-_query-_classifier](https://github.com/Aayushjshah/deBERTa-_query-_classifier)
- **Model hosted over Kaggle:** [https://www.kaggle.com/models/shah2001aayush/deberta_classification](https://www.kaggle.com/models/shah2001aayush/deberta_classification)
- **Dataset hosted over Kaggle:** [https://www.kaggle.com/datasets/shah2001aayush/dataprocessed](https://www.kaggle.com/datasets/shah2001aayush/dataprocessed)

**Important:** **Change HuggingFace Token below** to access the Mistral LLM model within the notebooks. Ensure you have a valid HuggingFace API token and replace the placeholder in the relevant notebook sections.