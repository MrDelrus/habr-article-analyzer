# Baselines

At this checkpoint, it is important to say that we have revised our project as a 'hub' recommendation system, not a 'hub' classification system. Here are the main points behind this decision:

- There are >2000 hubs, and new ones are added over time. If we want to build a classifier with high recall, we would have to constantly adjust it to new hubs (imagining our service works in a real production environment).

- We also reconsidered the product more as a recommendation system where you input your article text and it returns the best-fitting hubs, which is essentially a ranking task.

Therefore, we now measure our models on a ranking dataset using DCG and nDCG metrics.

## Word2Vec + KNN

We used pre-fitted word2vec (dim=300) and tried KNN for different hyperparams. We clearly see that such baseline is almost dummy and predicts the same probabilities, though we see good metrics for the validation step due to class inbalance. However, it's not an issue for the project since we use a different metric for model comparison, which should tackle the mentioned problem.

## BoW + NN

We implemented a Bag-of-Words model based on the top 5000 tokens selected by TF-IDF over the markdown text corpus and used it as a text embedder. For label representations, we used the mean embedding of texts from the training dataset (also based on the same 5000 tokens). We then applied a 3-layer Neural Network on top of these two embeddings.

We used a balanced dataset with a 1:3 positive-to-negative sample ratio. We mined negative hubs proportionally to their popularity to prevent the model from becoming biased.

## Metrics:
Both models showed suboptimal but non-random performance on the test set. Here are the metrics:

| Model | DCG | nDCG |
| :--- | :--- | :--- |
| W2V + KNN | 0.043 | 0.028 |
| BoW + NN | 0.101 | 0.075 |

## What to do next?
We can tune our models in many ways:
- Change arcitecture:
    * use transformers
    * use split-architecture
- Change loss functions:
    * use ranking pair-wise loss functions
- Optimize dataset:
    * mine more positives
    * make it even more balanced
- Do some feature engineering:
    * Use article author, comments and other data from the dataset as features for our models
