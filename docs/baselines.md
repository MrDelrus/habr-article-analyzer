# Baselines

## Word2Vec + KNN

We used pre-fitted word2vec (dim=300) and tried KNN for different hyperparams. We clearly see that such baseline is almost dummy and predicts the same probabilities, though we see good metrics for the validation step due to class inbalance. However, it's not an issue for the project since we use a different metric for model comparison, which should tackle the mentioned problem.
