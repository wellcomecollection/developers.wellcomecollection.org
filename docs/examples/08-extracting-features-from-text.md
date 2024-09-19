# 8. Extracting features from text

[View on GitHub](https://github.com/wellcomecollection/developers.wellcomecollection.org/blob/dependabot/npm_and_yarn/webpack-5.94.0/notebooks/08-extracting-features-from-text.ipynb) | [Run in Google Colab](https://colab.research.google.com/github/wellcomecollection/developers.wellcomecollection.org/blob/dependabot/npm_and_yarn/webpack-5.94.0/notebooks/08-extracting-features-from-text.ipynb)

In the last notebook, we saw that using a pre-trained network allowed us to extract features from images, and train a classifier for new categories on top of those features. We can do the same thing with text, using a pre-trained network to extract features from text. In this notebook, we'll use those features the visualise the similarities and differences between works in the collection, and try to find clusters of related material.

First, we need to install a few packages. We'll use `sentence-transformers` to manage our pre-trained language models, and `umap-learn` to compress our high-dimensional features, and `plotly` to visualise the results.


```python
! pip install -U --quiet sentence-transformers umap-learn plotly
```


```python
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from umap import UMAP
import gzip
import io
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import requests
```

## 8.1 Building a dataset 

We'll use the works snapshot in this exercise (as explained in notebook 4), but this data could just as easily be fetched from the API. The following code is the same as in notebook 4.


```python
snapshot_url = "https://data.wellcomecollection.org/catalogue/v2/works.json.gz"

data_dir = Path("./data").resolve()
data_dir.mkdir(exist_ok=True)

file_name = Path(snapshot_url).parts[-1]
zipped_path = data_dir / file_name
unzipped_path = zipped_path.with_suffix("")


if not unzipped_path.exists():
    if not zipped_path.exists():
        r = requests.get(snapshot_url, stream=True)
        download_progress_bar = tqdm(
            unit="B",
            total=int(r.headers["Content-Length"]),
            desc=f"downloading {file_name}",
        )
        with open(zipped_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    download_progress_bar.update(len(chunk))

    with gzip.open(zipped_path, "rb") as f_in:
        unzip_progress_bar = tqdm(
            unit="B",
            total=f_in.seek(0, io.SEEK_END),
            desc=f"unzipping {file_name}",
        )
        with open(unzipped_path, "wb") as f_out:
            for line in f_in:
                f_out.write(line)
                unzip_progress_bar.update(len(line))

if zipped_path.exists():
    zipped_path.unlink()
```

Now we can start building a dataset of work titles. Let's select 50,000 random works from the collection, and then extract their title text into a list of strings.


```python
n_works = sum(1 for line in unzipped_path.open())
```


```python
random_indexes = np.random.choice(n_works, 50_000, replace=False)
with open(unzipped_path, "r") as f:
    works = []
    for i, line in enumerate(f):
        if i in random_indexes:
            works.append(json.loads(line))
```


```python
titles = [work["title"] for work in works]
```


```python
titles[:5]
```

## 8.2 Text embedding models

Now that we have a dataset to work with, we can download the weights for a pretrained feature-extraction model. We're going to use the small but powerful `all-MiniLM-L6-v2` model (see the [docs on huggingface](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), or a [table comparing its performance to other models in the sentence transformers docs](https://www.sbert.net/docs/pretrained_models.html#model-overview)).

We'll save the weights locally to `./data/models`.


```python
model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name, cache_folder="./data/models")
```

We can use our model to extract features from our text. The `model.encode()` method takes a list of strings, and returns a list of 384-dimensional vectors. These features behave similarly to the image features we extracted in the last notebook. 

For example, the sentence

```the cat sat on the mat```

should be very similar (ie have a small distance from) the sentence

```a feline sits above the rug```

despite having few words in common.

Both should have a much larger distance from the sentence

```i hate this film```


```python
texts = [
    "the cat sat on the mat",
    "a feline sits above the rug",
    "i hate this film",
]

embeddings = model.encode(texts)
```


```python
embeddings.shape
```

We can calculate the similarity of embeddings using the cosine distances between them.


```python
from scipy.spatial.distance import cdist
```


```python
cdist(embeddings, embeddings, metric="cosine")
```

The diagonal here represents the distance from each sentence to itself, while the off-diagonal values represent the distance between each pair of sentences. We can see that the first two sentences are very similar (distance ~= 0.4), while the third is very different (distance ~= 1).

We can run the same encoding process for every title in our dataset:


```python
title_vectors = np.array([model.encode(title) for title in tqdm(titles)])
```

Again, we should expect that very similar titles will produce similar embeddings, while very different titles will produce very different embeddings.

## 8.3 Visualising the embeddings

The embeddings we've produced are 384-dimensional - too many to visualise directly. While the 384 dimensions give the model lots of room to express the differences between sentences, it's very hard to visualise more than 3 dimensions at a time. To get around this, we can use a [dimensionality reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction) technique to compress our 384-dimensional vectors into 3 dimensions. We'll use the [UMAP](https://umap-learn.readthedocs.io/en/latest/) algorithm to compress our initial vectors down to 2 dimensions so that they can be scattered on a 2D plot.


```python
dimension_reducer = UMAP(n_components=2, random_state=42)
title_embeddings_2d = dimension_reducer.fit_transform(title_vectors)
```


```python
plt.figure(figsize=(20, 20))
plt.scatter(
    title_embeddings_2d[:, 0], title_embeddings_2d[:, 1], alpha=0.2, c="k", s=5
)
plt.show()
```

By trying to preserve the distances between points, UMAP can give us a good idea of the relationships between our titles. We can see clusters of points across the plot, indicating that there are a few groups of similar titles, distinct from the rest of our dataset

## 8.4 Interactive visualisations

We can visualise this data in a more interactive way using [plotly](https://plotly.com/python/). Plotly is a powerful plotting library that allows us to create interactive plots that can be embedded in web pages. We can use plotly to create a scatter plot of our data, and then add a hover effect that shows the title of each work when we hover over it.

N.B. You won't be able to see this if you're reading the markdown version of this notebook, or viewing it on GitHub. You'll need to run the notebook yourself to see the interactive plot.

We'll start by loading our 2d embeddings into a dataframe, along with the original titles.


```python
df = pd.DataFrame(title_embeddings_2d, columns=["x", "y"])
df["title"] = titles
```

We'll then use plotly to create a scatter plot of our data, with the the `hover_data` parameter set to add the title of each work when the user hovers over it.


```python
fig = px.scatter(
    df, x="x", y="y", hover_data=["title"], width=1000, height=1000
)
fig.update_traces(marker=dict(size=5, opacity=0.2, color="grey"))
fig.update_layout(
    paper_bgcolor="white",
    plot_bgcolor="white",
    xaxis=dict(gridcolor="rgb(220, 220, 220)", showgrid=True),
    yaxis=dict(gridcolor="rgb(220, 220, 220)", showgrid=True),
)
```

As expected, similar titles have been placed in similar regions of the space! We can see that the model has learned to distinguish between titles that are similar in meaning, but different in wording, and titles that are completely different.

## 8.5 Clustering

We can use the features we've extracted to cluster our works into groups of similar titles. We'll use the k-means algorithm to cluster our works into 50 groups. We'll then add the cluster labels to our dataframe, and use plotly to colour the points in our plot by cluster.

N.B. Many other clustering algorithms are available, and might yield better results! If you're running this notebook yourself, try switching the clusterer to use the `OPTICS` algorithm instead, taking advantage of the fact that it doesn't require us to specify the number of clusters in advance.


```python
from sklearn.cluster import OPTICS, KMeans
```


```python
clusterer = KMeans(n_clusters=50)

# clusterer = OPTICS(min_samples=10, xi=0.01, min_cluster_size=0.001)
```

Note here that we're finding our clusters in our original, 384-dimensional space, instead of our reduced 2d space. This allows us to retain all of the complexity of our original embeddings, and find clusters that are more meaningful than those we'd find in our reduced space.


```python
clusters = clusterer.fit_predict(title_vectors)
```

Let's add those cluster labels to our dataframe.


```python
import pandas as pd

df = pd.DataFrame(
    {
        "title": titles,
        "cluster": clusters,
        "x": title_embeddings_2d[:, 0],
        "y": title_embeddings_2d[:, 1],
    }
)

df.head()
```

And look at the number of titles which have been added to each bucket


```python
df["cluster"].value_counts()
```

## 8.6 Visualising the clusters

Remember, we've found our clusters in our original, 384-dimensional space, but we're visualising them in our reduced 2d space. This means that we might not see all of the complexity of our original embeddings in our reduced-space visualisation, so the clusters might look less coherent when we plot them!

N.B. You won't be able to see this if you're reading the markdown version of this notebook, or viewing it on GitHub. You'll need to run the notebook yourself to see the interactive plot.


```python
fig = px.scatter(
    df,
    x="x",
    y="y",
    color="cluster",
    hover_data=["title"],
    width=1000,
    height=1000,
)

fig.update_layout(
    paper_bgcolor="white",
    plot_bgcolor="white",
    xaxis=dict(gridcolor="rgb(220, 220, 220)", showgrid=True),
    yaxis=dict(gridcolor="rgb(220, 220, 220)", showgrid=True),
)
```

## 8.7 3D Visualisation

We can also use plotly to build 3D interactive scatter plots, which can be rotated and zoomed to explore the data. We'll use roughly the same code as before, but use a UMAP model with `n_components` set to `3` to reduce our embeddings to 3 dimensions instead of 2. 




```python

dimension_reducer = UMAP(n_components=3, random_state=42, n_jobs=-1)
title_embeddings_3d = dimension_reducer.fit_transform(title_vectors)
```


```python
df = pd.DataFrame(
    {
        "title": titles,
        "cluster": clusters,
        "x": title_embeddings_3d[:, 0],
        "y": title_embeddings_3d[:, 1],
        "z": title_embeddings_3d[:, 2],
    }
)
```

N.B. You won't be able to see this if you're reading the markdown version of this notebook, or viewing it on GitHub. You'll need to run the notebook yourself to see the interactive plot.


```python
fig = px.scatter_3d(
    df,
    x="x",
    y="y",
    z="z",
    color="cluster",
    hover_data=["title"],
    width=1000,
    height=1000,
    size_max=5,
)
```


```python
fig.show()
```

## Exercises

1. Adapt the data-fetching code to use the API, instead of the works snapshot.
2. Use a different pre-trained model to extract features from the text. How does the visualisation change?
3. Try using a different clustering algorithm to cluster the works. How do the meanings/boundaries of the clusters change?
4. Try to build a simple semantic search function, by allowing the user to enter a search term, embedding their search term using the feature-extracting model, and returning the titles that are closest to that term. How well does it work? How could you improve it?


```python

```
