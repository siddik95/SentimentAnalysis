### Sentiment Analysis of Hotel Reviews using OpenAI API
This project performs sentiment analysis on hotel reviews to classify them as "good" or "bad" based on the text of the review. 
The analysis leverages the OpenAI API to generate sentiment scores and then evaluates the results.
### Project Overview
The dataset consists of over 500,000 hotel reviews from Europe. Each review includes a textual feedback component (positive and negative) and a numerical "Reviewer_Score". For this project, we simplify the problem by categorizing reviews into two classes:

* `Bad reviews: Reviewer_Score < 5`

* `Good reviews: Reviewer_Score >= 5`

The primary goal is to predict this sentiment classification using only the text of the reviews.

### 1. Data Loading and Preprocessing
First, we import the necessary libraries and load the dataset. This dataset comprises 515,000 customer reviews and ratings of 1,493 luxury hotels across Europe. While this dataset provides additional information, for this project, I will focus solely on the review text. If you are interested, [click here to download or know more about this data](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)

```
import pandas as pd
import numpy as np
import os
```
This code block imports the essential Python libraries for this project:

`pandas`: For data manipulation and analysis, particularly for handling the data in a DataFrame.

`numpy`: For numerical operations.

`os`: For interacting with the operating system, which can be useful for file path management.

Next, we load the dataset from a CSV file and display its information.

```
#read the already downloaded data
df = pd.read_csv(r"F:\onedrive\OneDrive - University of Central Florida\Data Mining I\Data\Hotel_Reviews.csv")
df.info()
```
This code reads the `Hotel_Reviews.csv` file into a pandas DataFrame called `df`. The `df.info()` command is used to get a concise summary of the DataFrame, including the number of entries, the number of columns, the data types of each column, and the number of non-null values.
<img width="1197" height="442" alt="image" src="https://github.com/user-attachments/assets/1dcdba13-a149-4de5-a2c1-f96ddcc04832" />

We then preprocess the data by combining the `Negative_Review` and `Positive_Review` columns into a single text column. We also create a binary label column based on the `Reviewer_Score`.

```
#Since every since review could have both positive and negative parts. I will append both of together as review
df["text"] = df["Negative_Review"] + df["Positive_Review"]
df['label'] = df['Reviewer_Score'].apply(lambda x: 1 if x>=5 else 0)
df = df[['text', 'label']]
df.head()
```
* We concatenate the text from `Negative_Review` and `Positive_Review` to create a single comprehensive review text.

* A new `label` column is created where a score of 5 or greater is marked as `1` (good) and anything less is marked as `0` (bad).

* We then create a new DataFrame containing only the `text` and `label` columns, and `df.head()` displays the first five rows.

### 2. Sentiment Classification with OpenAI API
We use the OpenAI API to classify the sentiment of each review.
```
from openai import OpenAI
client = OpenAI(api_key="your_api_key_here")
def classify_sentiment(text):
    response = list(client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Analyze the sentiment of the text and rate the sentiment from -1 to 1, 1 is super positive, -1 is super negative, 0 is neutral."},
            {"role": "user", "content": text}
        ]
    ).choices)[0].message.content
    return response
```
* This code defines a function `classify_sentiment` that takes a text string as input.

* It sends a request to the OpenAI `gpt-3.5-turbo` (one could change it to a different model based on the need) model, asking it to analyze the sentiment of the text and return a rating between -1 (very negative) and 1 (very positive).

* The function then returns the model's response.

The following code applies this function to each review in our DataFrame and saves the results to a new CSV file. I used this on the whole dataset, which takes a significant amount of time to complete, if you want 
You could also use a pandas sample to select the number of data points you want randomly.

```
# Apply the classification function to each entry in the DataFrame
df['sentiment'] = ''
for j in range(df.shape[0]):
    if df.loc[j,'sentiment'] == '':
        try:
            df.loc[j,'sentiment'] = classify_sentiment(df.loc[j,'text'])
            df.to_csv('review_with_sentiment_updated.csv')
        except Exception as e:
            continue
```
* This loop iterates through each row of the DataFrame.

* It calls the `classify_sentiment` function for each review text.

* The returned sentiment is stored in a new `sentiment` column.

* The DataFrame is saved to `review_with_sentiment_updated.csv` after each successful API call to prevent data loss in case of an interruption. The `try-except` block handles potential errors that may occur during the API call.

### 3. Data Cleaning and Feature Extraction
After running the sentiment classification, we load the updated data.
```
#import the saved data
import pandas as pd
data = pd.read_csv(r'review_with_sentiment_updated.csv')
data.dropna(inplace = True)
data.info()
```
This block reads the `review_with_sentiment_updated.csv` file, which now includes the sentiment analysis results from the OpenAI API. `data.dropna(inplace=True)` removes any rows with missing values.

The sentiment returned by the API is a descriptive text. We need to extract the numerical score from it.

```
#extract only values from the entire text
import numpy as np
data['extracted_sentiment'] = data['sentiment'].str.findall(r'[-+]?\\d*\\.\\d+|\\d+')
# Function to filter numbers within the range of -1 to 1
def filter_numbers(matches):
    filtered = [float(num) for num in matches if -1 <= float(num) <= 1]
    return np.median(filtered)

data['extracted_sentiment'] = data['extracted_sentiment'].apply(filter_numbers)
```
We use a regular expression `r'[-+]?\\d*\\.\\d+|\\d+'` to find all numbers (including decimals and negative numbers) in the `sentiment` text.

The `filter_numbers` function then takes these extracted numbers, converts them to floats, and keeps only those that fall within the -1 to 1 range. It returns the median of these valid numbers to get a single representative sentiment score.

This score is stored in a new column called `extracted_sentiment`.

### 4. Model Evaluation
We can now evaluate how well the extracted sentiment scores correlate with the original labels.

##### Sentiment Score Distribution
A histogram shows the distribution of the extracted sentiment scores. The following code will save the plot as `sentiment_distribution.png`.

```
import matplotlib.pyplot as plt
data['extracted_sentiment'].hist()
plt.title('Distribution of Extracted Sentiment Scores')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.savefig('sentiment_distribution.png')
plt.show()
```
##### Sentiment Score Distribution
A histogram shows the distribution of the extracted sentiment scores. The following code will save the plot as `sentiment_distribution.png`. 
```
import matplotlib.pyplot as plt
data['extracted_sentiment'].hist()
plt.title('Distribution of Extracted Sentiment Scores')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.savefig('sentiment_distribution.png')
plt.show()
```
Based on the histogram, we can see that the tail of the distribution is towards
negative values.
<img width="589" height="426" alt="image" src="https://github.com/user-attachments/assets/80ed2af4-df41-47c1-9661-15f880534dfc" />

##### Boxplot Analysis
A boxplot helps us see the relationship between the original labels (good/bad reviews) and the sentiment scores. This code saves the plot as `sentiment_boxplot.png`.
```
import matplotlib.pyplot as plt
data.boxplot(
    column = 'extracted_sentiment',
    by = 'label'
)
plt.title('Sentiment Score vs. Review Label')
plt.suptitle('') # Suppress the default title
plt.xlabel('Review Label (0: Bad, 1: Good)')
plt.ylabel('Extracted Sentiment Score')
plt.savefig('sentiment_boxplot.png')
plt.show()
```
<img width="597" height="472" alt="image" src="https://github.com/user-attachments/assets/4ccd9601-19b8-45ab-8cc8-ce4bfd1f693f" />

##### Crosstabulation
A crosstabulation gives us a table showing the counts of positive, negative, and neutral sentiment scores for each label.
```
pd.crosstab(
    index = data['label'],
    columns = np.sign(data['extracted_sentiment'])
)
```
We can see that positive reviews are concentrated towards positive values, while negative reviews are concentrated towards negative values, with a few exceptions.
<img width="1154" height="124" alt="image" src="https://github.com/user-attachments/assets/e072d1aa-6b6d-46f1-b1bb-61730f2903eb" />

This table helps us see how many of the "good" reviews were classified with positive sentiment and how many of the "bad" reviews were classified with negative sentiment.

##### ROC Curve
The Receiver Operating Characteristic (ROC) curve is a graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied. This code saves the plot as `roc_curve.png`.
```
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
fpr, tpr, thresholds = roc_curve(
    y_true = data.loc[~data['extracted_sentiment'].isna(),'label'],
    y_score = data.loc[~data['extracted_sentiment'].isna(),'extracted_sentiment']
)
roc_auc = auc(fpr, tpr)
# Plot ROC curve
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('roc_curve.png')
plt.show()
```
This code calculates the True Positive Rate (TPR) and False Positive Rate (FPR) for various threshold settings.

The Area Under the Curve (AUC) is calculated, which provides an aggregate measure of performance across all possible classification thresholds. An AUC of 1.0 represents a perfect model, while an AUC of 0.5 represents a model with no discriminative ability. The ROC curve score of 0.79 indicates that the model does a decent job of classifying sentiment correctly. 
<img width="594" height="465" alt="image" src="https://github.com/user-attachments/assets/b0665647-d81e-46af-9c7c-d813409d14a5" />

The resulting plot shows the ROC curve.

#### Conclusion
This project successfully demonstrates a pipeline for performing sentiment analysis on hotel reviews. By leveraging the OpenAI API, we were able to generate nuanced sentiment scores which, after some processing, showed a strong correlation with the original reviewer ratings. The evaluation metrics, particularly the ROC curve with a high AUC, indicate that this approach is effective in distinguishing between positive and negative reviews.
















