# 9. Building a recommender system for subjects

[View on GitHub](https://github.com/wellcomecollection/developers.wellcomecollection.org/blob/dependabot/npm_and_yarn/webpack-5.94.0/notebooks/09-building-a-recommender-system-for-subjects.ipynb) | [Run in Google Colab](https://colab.research.google.com/github/wellcomecollection/developers.wellcomecollection.org/blob/dependabot/npm_and_yarn/webpack-5.94.0/notebooks/09-building-a-recommender-system-for-subjects.ipynb)

Finally, we'll consider building a recommender system using data from Wellcome Collection. Thes machine learning models work slightly differently to the ones we've seen so far. Rather than being trained to predict a single value, they're trained to predict a whole matrix of interactions between two kinds of entity.

Classically, these entities are 'users' and 'items', where an item might be a film, a book, a song, etc. However, because we don't have data about the interactions between users of wellcomecollection.org and its works, we're instead going to train a recommender system to predict the interactions between works and the subjects they're tagged with.

The hope is that we'll then be able to make recommendations for subjects which could appear on another work, based on the other subjects which it has been tagged with. We might also be able to use some of the features learned along to way to explore the similarity between works, or between subjects.

## 9.1 Model architecture

This class of problem is known as a [collaborative filtering](https://en.wikipedia.org/wiki/Collaborative_filtering) problem, and there are a number of different approaches to solving it. We're going to use a relatively simple technique called [matrix factorisation](https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems)).

First, we'll create a matrix to represent interactions between works and subjects. The interaction matrix should be binary, where the value of each element `(i, j)` is 1 if the work `i` is tagged with the subject `j`, and 0 otherwise. This interaction matrix will be what our model is trained to predict. Because we're representing the interaction between all possible works and all possible subjects, the interaction matrix will be very large, and very sparse.

Target interaction matrix shape: `(n_works, n_subjects)`

Then, we'll create two sets of randomly-initialised embeddings, one for works and one for subjects, with a shared dimensionality. 

Work embedding matrix shape: `(n_works, embedding_dim)`

Subject embedding matrix shape: `(n_subjects, embedding_dim)`

We'll multiply these two matrices together to get a matrix of predicted interactions between works and subjects.

Predicted interaction matrix shape: `(n_works, n_subjects)`

We'll then train the model by incrementally tweaking the embeddings for all of our works and subjects, making slightly better predictions about the likely interactions between works and subjects each time. We'll use the [binary cross entropy](https://en.wikipedia.org/wiki/Cross_entropy) error loss function to measure our progress along the way.


```python
import json
import pandas as pd
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
```

## 9.2 Building the target interaction matrix

First, we need to load all of the works and keep track of which subjects exist on each one. We'll do this by loading the local `works.json` snapshot (see the notebook on snapshots to download this if you haven't already!). We'll then iterate over the works and subjects, and create a dictionary mapping from work ID to a list of subject IDs.


```python
works_path = Path("./data/works.json")
n_works = sum(1 for line in open(works_path))
```


```python
n_works
```


```python
subject_dict = {}

with open(works_path, "r") as f:
    for i, line in tqdm(enumerate(f), total=n_works):
        work = json.loads(line)
        if len(work["subjects"]) == 0:
            continue
        try:
            subject_dict[work["id"]] = []
            for subject in work["subjects"]:
                for concept in subject["concepts"]:
                    subject_dict[work["id"]].append(concept["id"])
        except KeyError:
            continue
```


```python
len(subject_dict)
```

Now we can start building the target interaction matrix. We'll use the `scipy.sparse` module to create a [sparse matrix](https://en.wikipedia.org/wiki/Sparse_matrix) with the correct shape, and then iterate over the dictionary we just created, setting the value of each element in the matrix to 1.


```python
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer
```


```python
mlb = MultiLabelBinarizer()
mlb.fit(subject_dict.values())

len(mlb.classes_)
mlb.classes_

subject_matrix = csr_matrix(
    (
        np.ones(sum(len(v) for v in subject_dict.values())),
        (
            np.repeat(list(subject_dict.keys()), [len(v) for v in subject_dict.values()]),
            np.concatenate(list(subject_dict.values())),
        ),
    )
)
```

## 9.3 Building our embedding layers

Now that we've got a sparse matrix of interactions, we can begin modelling the features which will go into the calculation of our predictions. We'll start by creating two embedding layers, one for works and one for subjects. We'll use the `torch.nn.Embedding` class to do this. We'll set the size of the embedding to 10, which means that each work and subject will be represented by a vector of 10 floating point numbers. We'll create one embedding for each unique work, and one for each unique subject.


```python
import torch
from torch.nn import Embedding
```


```python
unique_work_ids = list(subject_dict.keys())
unique_subjects = list(
    set([subject for subjects in subject_dict.values() for subject in subjects])
)

len(unique_work_ids), len(unique_subjects)
```


```python
work_embeddings = Embedding(
    num_embeddings=len(unique_work_ids), embedding_dim=10
)

subject_embeddings = Embedding(
    num_embeddings=len(unique_subjects), embedding_dim=10
)

work_embeddings.weight.shape, subject_embeddings.weight.shape
```

These embedding layers can be multiplied together to produce a matrix of predicted interactions between works and subjects. We'll use `torch.matmul` to do the matrix multiplication.


```python
a = torch.matmul(
    work_embeddings.weight[:100], subject_embeddings.weight[:100].T
)
a
```


```python
a.shape
```

That's the core of our model! The predictions might not be meaningful at the moment (we're just multiplying two random matrices together), but we can train the model to make better predictions by tweaking the values of the embeddings.

## 9.4 Grouping our layers into a model

Before we begin training our model, we should wrap our embedding layers up into a single model class. We'll use the `torch.nn.Module` class to do this, giving us access to all sorts of pytorch magic with very little effort. 

We'll need to add a `.forward()` method to the class, which will be used to calculate predictions during each training step, and at inference time.


```python
from torch.nn import Module

class Recommender(Module):
    def __init__(self, n_works, n_subjects, embedding_dim):
        super().__init__()
        self.work_embeddings = Embedding(
            num_embeddings=n_works, embedding_dim=embedding_dim
        )
        self.subject_embeddings = Embedding(
            num_embeddings=n_subjects, embedding_dim=embedding_dim
        )

    def forward(self, work_ids, subject_ids):
        work_embeddings = self.work_embeddings(work_ids)
        subject_embeddings = self.subject_embeddings(subject_ids)
        predictions = torch.matmul(work_embeddings, subject_embeddings.T)
        return torch.sigmoid(predictions)
```


```python
model = Recommender(
    n_works=len(unique_work_ids), 
    n_subjects=len(unique_subjects), 
    embedding_dim=10
)
```

# 9.5 Training the model

Now that we've got a model, we need to come up with a way of training it. Typically, machine learning models are trained using a technique called [stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent). This involves taking a small batch of training examples, calculating the error for each one, and then using the average error to calculate the gradient of the error function with respect to each of the model's parameters. The parameters are then updated by a small amount in the direction of the gradient, and the process is repeated.

Our interaction matrix is huge, and sparsely populated, so training on the whole thing at once would be very slow. Instead, we'll randomly sample a small batch of interactions from the matrix, and train on those, incrementally updating the weights. We'll do this repeatedly, until we've seen every interaction in the matrix at least once.

This process will be wrapped up into a custom `BatchGenerator` class.


```python
class BatchGenerator:
    def __init__(self, subject_dict, batch_size):
        self.subject_dict = subject_dict
        self.batch_size = batch_size
        self.n_batches = len(subject_dict) // batch_size
        self.work_ids = list(self.subject_dict.keys())
        self.work_id_to_index = {
            work_id: i for i, work_id in enumerate(self.work_ids)
        }
        self.index_to_work_id = {
            i: work_id for i, work_id in enumerate(self.work_ids)
        }

        self.unique_subjects = list(
            set(
                [
                    subject
                    for subjects in subject_dict.values()
                    for subject in subjects
                ]
            )
        )
        self.subject_to_index = {
            subject: i for i, subject in enumerate(self.unique_subjects)
        }
        self.index_to_subject = {
            i: subject for i, subject in enumerate(self.unique_subjects)
        }

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        """
        Yields a tuple of work_ids and subject_ids, and the target
        adjacency matrix for each batch.
        """
        # split the work ids into randomly shuffled batches
        work_ids_batches = np.array_split(
            np.random.permutation(self.work_ids), self.n_batches
        )
        for work_ids_batch in work_ids_batches:
            # get the work ids for each work in the batch
            work_ids = [
                self.work_id_to_index[work_id] for work_id in work_ids_batch
            ]

            # get the subset of subjects which appear on the works in the batch.
            # this is the set of subjects we want to predict against for each work
            # in the batch.
            subject_ids = [
                self.subject_to_index[subject]
                for work_id in work_ids_batch
                for subject in self.subject_dict[work_id]
            ]

            # create the target adjacency matrix using the work ids and subject
            # ids
            target_adjacency_matrix = torch.zeros(
                len(work_ids), len(subject_ids)
            )

            for i, work_id in enumerate(work_ids_batch):
                for subject in self.subject_dict[work_id]:
                    j = subject_ids.index(self.subject_to_index[subject])
                    target_adjacency_matrix[i, j] = 1

            yield (
                torch.tensor(work_ids),
                torch.tensor(subject_ids),
                target_adjacency_matrix,
            )
```

We'll use the `Adam` optimiser to update the weights of our model at each step. Adam is a variant of stochastic gradient descent which uses a slightly more sophisticated update rule, and is generally more effective than raw SGD.


```python
from torch.optim import Adam

optimizer = Adam(params=model.parameters(), lr=0.001)
```

Our optimiser will be led by a `BinaryCrossEntropyLoss` loss function, which will calculate the error between our predictions and the target interactions.


```python
from torch.nn import BCELoss

binary_cross_entropy = BCELoss()
```

We can now set our model training, using a batch size of 512, and training for 10 epochs. As usual, we'll keep track of the loss at each step, and plot it at the end.


```python
n_epochs = 10
batch_size = 512

losses = []
progress_bar = tqdm(range(n_epochs * (len(subject_dict) // batch_size)), unit="batches")
for epoch in progress_bar:
    progress_bar.set_description(f"Epoch {epoch}")
    batch_generator = BatchGenerator(subject_dict, batch_size=batch_size)
    for work_ids, subject_ids, target_adjacency_matrix in batch_generator:
        predictions = model(work_ids, subject_ids)
        loss = binary_cross_entropy(predictions, target_adjacency_matrix)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        progress_bar.set_postfix({
            "BCE": np.mean(losses[-100:]),
        })
        progress_bar.update()
```


```python
from matplotlib import pyplot as plt

plt.plot(losses)
```


```python
# plot the log of the loss to see the trend more clearly
plt.plot(np.log(losses))
```

## 9.6 Making predictions

We can now use our trained model to make predictions about the interactions between works and subjects. If we select a random work (and find its trained embedding) we can multiply it by the subject embeddings to get a vector of predicted interactions between that work and each subject. We can then sort the subjects by their predicted interaction, and select the 10 highest scoring subjects. These are the subjects which our model thinks are most likely to be relevant to the work.


```python
model.eval()

work_embeddings = model.work_embeddings.weight.detach().numpy()
subject_embeddings = model.subject_embeddings.weight.detach().numpy()
```


```python
random_work_id = np.random.choice(unique_work_ids)
random_work_index = unique_work_ids.index(random_work_id)
```

Let's use the API to find out what this work is


```python
base_url = "https://api.wellcomecollection.org/catalogue/v2/"
work_url = f"{base_url}works/{random_work_id}"
```


```python
import requests

requests.get(work_url).json()["title"]
```

Let's find its embedding


```python
random_work_embedding = work_embeddings[random_work_index]
random_work_embedding
```

And make some predictions about which subjects it's likely to be tagged with


```python
predictions = np.matmul(
    random_work_embedding, subject_embeddings.T
)

top_predicted_subject_indexes = predictions.argsort()[::-1][:10]
```


```python
predicted_concept_ids = [
    unique_subjects[index] for index in top_predicted_subject_indexes
]

predicted_concept_ids
```

Let's have a look at the top 10 predicted subjects' labels


```python
for concept_id in predicted_concept_ids:
    concept_url = f"{base_url}concepts/{concept_id}"
    label = requests.get(concept_url).json()["label"]
    print(label)
```

## 9.7 Visualising the embeddings

We can also visualise the similarity of the embeddings we've learned, in the same way as we did for the text embeddings in the previous notebook. Again, we'll use the [UMAP](https://umap-learn.readthedocs.io/en/latest/) algorithm to reduce the dimensionality of the embeddings to 2, and then plot them on a scatter plot.


```python
from umap import UMAP

reducer = UMAP(n_components=2)
subject_embeddings_2d = reducer.fit_transform(subject_embeddings)

subject_embeddings_2d.shape
```


```python
from matplotlib import pyplot as plt

plt.figure(figsize=(20, 20))
plt.scatter(
    subject_embeddings_2d[:, 0], 
    subject_embeddings_2d[:, 1], 
    alpha=0.2, 
    c="k",
    s=5
)
plt.show()
```

## Exercises

1. Try training the model for longer, or with a different batch size. How does this affect the loss, and the quality of the corresponding predictions?
2. Try changing the size of the embedding. How does this affect the loss, and the quality of the corresponding predictions?
3. Can you think of a way of incorporating more prior information into the embeddings? Can you make these constraints trainable in the model's backward pass?


