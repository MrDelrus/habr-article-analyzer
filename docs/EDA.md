# Explorationaly Data Analasis

## Analysis of Targets and Their Clustering

We explored various columns in the dataset as potential targets for the classification task. We also examined how these columns are automatically populated when writing articles on Habr.

Among the four types of article tags available in the dataset, the “hubs” — Habr’s thematic sections — are the most suitable for our task.

Although hubs are a good fit, they still require preprocessing, as there are over 2,000 of them.

We considered two approaches:
- Selecting the top-N hubs that cover the majority of articles.
- Clustering all hubs into N clusters.

As a result, we formulated three target variants, each with its own advantages and drawbacks:

- **Top 150 hubs**, covering slightly less than 90% of articles

    * [+] This approach is likely to achieve the best classification quality for the most popular labels.
    * [-] The model will have no knowledge of the remaining 1,900+ hubs.

- **75 clusters**, providing 100% coverage

    * [+] Full coverage of all hubs.
    * [-] The largest labels are mixed within clusters, so performance on top hubs is likely lower than in the first approach.
- **Mixed solution**: 55 hubs assigned to “personal clusters” and the remaining hubs grouped into 75 clusters

    * [+] Combines the advantages of both approaches: full coverage and likely good performance on top hubs.
    * [-] The actual results are yet to be seen during model training.

In the next steps, we will evaluate which approach provides the best end-to-end experience for our service users.

## Analysis of Features

Detailed analysis you can see in [jupyter notebook](../notebooks/exploratory_data_analysis/data_visualization.ipynb). Here only the key conclusions are mentioned:

1. The data contains outliers that need to be filtered.

2. The classes (`hubs`) are highly imbalanced. A small group of classes occurs very frequently, while most appear in only a few dozen articles.

3. The distribution of statistics depends on the class.

4. Different years contribute differently to the dataset — statistical distributions vary across years.

5. The texts are technical; most words are nouns, while adjectives are relatively rare.
