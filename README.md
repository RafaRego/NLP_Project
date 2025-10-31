# NLP_Project
**Dataset links:**

128k Airline Reviews — main large review set: https://www.kaggle.com/datasets/joelljungstrom/128k-airline-reviews

Skytrax Reviews (GitHub) — detailed reviews with "recommended" and countries: https://github.com/quankiquanki/skytrax-reviews-dataset

Twitter US Airline Sentiment — labeled tweets for training: https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment

**File Overview:**

train_model.py — fine-tunes the DistilBERT model on the Twitter airline sentiment dataset.

preprocessing.py / preprocessing_large_dataset.py / preprocessing_rafa.py — text cleaning and preprocessing pipelines for the Twitter and review datasets.

use_model.py / use_model_on_reviews_large.py / use_model_on_reviews_min.py — apply the trained model to the different review datasets and save predicted sentiments.

assess_performance_validation_set.py — evaluates model performance on the Skytrax validation dataset (generates confusion_matrix.png and confidence_distributions.png).

assess_performance_large_dataset.py — evaluates performance on the large test set (generates confusion_matrix_large.png and confidence_distributions_large.png).

questin_2_processing.py — performs the analysis for Research Question 2 (market capitalization vs. sentiment correlation).

main.py 

environment.yml / requirements.txt — specify Python environment and dependencies for reproducibility.

Images and datasets (.png, .csv) — included for reference and reproducibility (confusion matrices, confidence plots, intermediate data).

**How to Run**

Install dependencies:
pip install -r requirements.txt

### Authors
**Rafael Lima Araujo Rego**
  Erasmus — TUD (Student ID: 6538118)
  Delft, Netherlands
  Email: [R.LimaAraujoRego@student.tudelft.nl](mailto:R.LimaAraujoRego@student.tudelft.nl)
**Sebastiaan Helbing**
  Erasmus — TUD (Student ID: 6593801)
  Delft, Netherlands
  Email: [M.S.Helbing@student.tudelft.nl](mailto:M.S.Helbing@student.tudelft.nl)
**Casper Sinck**
  Erasmus — TUD (Student ID: 5638828)
  Delft, Netherlands
  Email: [C.W.Sinck-1@student.tudelft.nl](mailto:C.W.Sinck-1@student.tudelft.nl)
**Julian Pickert Mayén**
  Erasmus — TUD (Student ID: 6594700)
  Delft, Netherlands
  Email: [jpickertmayen@tudelft.nl](mailto:jpickertmayen@tudelft.nl)
**Marijn Zetsma**
  Erasmus — TUD (Student ID: 5818591)
  Delft, Netherlands
  Email: [M.Zetsma-1@student.tudelft.nl](mailto:M.Zetsma-1@student.tudelft.nl)
