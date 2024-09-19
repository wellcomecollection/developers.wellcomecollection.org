# 2. Extracting more data for local analysis

[View on GitHub](https://github.com/wellcomecollection/developers.wellcomecollection.org/blob/rk/remove-buildkite-pipeline/notebooks/02-extracting-more-data-for-local-analysis.ipynb) | [Run in Google Colab](https://colab.research.google.com/github/wellcomecollection/developers.wellcomecollection.org/blob/rk/remove-buildkite-pipeline/notebooks/02-extracting-more-data-for-local-analysis.ipynb)

In the last notebook, we saw that the `/works` API can do some clever querying and filtering. However, we often have questions which can't be answered by the API by itself. In those cases, it's useful to collect a load of data from the API and then analyse it locally.

In this notebook, we'll try to query the API for bigger chunks of data so that we can answer a more interesting question.

We'll aim to find out:

> If we filter the works API for a set of subjects, can we find the other subjects that most commonly co-occur with them?

We'll start by fetching all of the works which are tagged with a single subject.

Here's our base URL again:


```python
base_url = "https://api.wellcomecollection.org/catalogue/v2/"
```

Lets' make a request to the API, asking for all the works which are tagged with the subject "Influenza".


```python
import requests

response = requests.get(
    base_url + "works", params={"subjects.label": "Influenza"}
).json()
```


```python
response["totalResults"]
```

## 2.1 Page sizes


```python
response["totalPages"]
```


```python
len(response["results"])
```

At the moment, we're getting our results spread across 9 pages, because `pageSize` is set to 10 by default. 

We can increase the `pageSize` to get all of our 81 works in one go (up to a maximum of 100):


```python
import requests

response = requests.get(
    base_url + "works", params={"subjects.label": "Influenza", "pageSize": 100}
).json()
```


```python
response["totalResults"]

```


```python
response["totalPages"]
```

## 2.2 Requesting multiple pages of results

Some subjects only appear on a few works, but others appear on thousands. If we want to be able to analyse those larger subjects, we'll need to fetch more than 100 works at a time. To do this, we'll page through the results, making multiple requests and building a local list of results as we go.

If the API finds more than one page of results for a query, it will provide a `nextPage` field in the response, with a link to the next page of results. We can use this to fetch the next page of results, and the next, and the next, until the `nextPage` field is no longer present, at which point we know we've got all the results.

We're going to use these results to answer our question from the introduction, so we'll also ask the API to include the subjects which are associated with each work, and collect them too.


```python
from tqdm.auto import tqdm

```


```python
results = []

# fetch the first page of results
response = requests.get(
    base_url + "works",
    params={
        "subjects.label": "England",
        "include": "subjects",
        "pageSize": "100",
    },
).json()

# start a progress bar to keep track of how many results we've fetched
progress_bar = tqdm(total=response["totalResults"])

# add our results to the list and update our progress bar
results.extend(response["results"])
progress_bar.update(len(response["results"]))

# as long as there's a "nextPage" key in the response, keep fetching results
# adding them to the list, and updating the progress bar
while "nextPage" in response:
    response = requests.get(response["nextPage"]).json()
    results.extend(response["results"])
    progress_bar.update(len(response["results"]))

progress_bar.close()

works_about_england = results
```

let's check that we've got the correct number of results:


```python
len(works_about_england) == response["totalResults"]
```

Great! Now let's try collecting works for a second subject:


```python
results = []

response = requests.get(
    base_url + "works",
    params={
        "subjects.label": "Germany",
        "include": "subjects",
        "pageSize": "100",
    },
).json()

progress_bar = tqdm(total=response["totalResults"])

results.extend(response["results"])
progress_bar.update(len(response["results"]))

while "nextPage" in response:
    response = requests.get(response["nextPage"]).json()
    results.extend(response["results"])
    progress_bar.update(len(response["results"]))

progress_bar.close()

works_about_germany = results
```

## 2.3 Analyzing our two sets of results 

Let's find the works which are tagged with both subjects by filtering the results of the first list by IDs from the second list.


```python
ids_from_works_about_england = set([work["id"] for work in works_about_england])
```


```python
works_about_england_and_germany = [
    work
    for work in works_about_germany
    if work["id"] in ids_from_works_about_england
]
```


```python
len(works_about_england_and_germany)
```


```python
works_about_england_and_germany
```

That's 32 works which are tagged with both `England` and `Germany`. Let's see if we can find the other subjects which are most commonly found on these works. 

Let's use a `Counter` to figure that out:

N.B. We're collecting the _concepts_ on each work because they are the atomic constituent parts of subjects. Our catalogue includes subjects like "Surgery - 18th Century" which are made up of the concepts "Surgery" and "18th Century". It's more desirable to compare the concepts, because the subjects can be so specific and are less likely to overlap.


```python
from collections import Counter

concepts = Counter()

for record in works_about_england_and_germany:
    # we need to navigate the nested structure of the subject and its concepts to
    # get the complete list of _concepts_ on each work
    for subject in record["subjects"]:
        for concept in subject["concepts"]:
            concepts.update([concept["label"]])
```



The `Counter` object keeps track of the counts of each unique item we pass to it. Now that we've added the complete list, we can ask it for the most common items:


```python
concepts.most_common(20)
```

Great! We've solved our original problem:

> If we filter the works API for a set of subjects, can we find the other concepts that most commonly co-occur with them?

## 2.4 Creating a generic function for finding subject intersections

Now that we've solved this problem, let's try to make it more generic so that we can use it for other pairs of subjects.

We can re-use a lot of the code we've already written, and wrap it in a couple of reusable function definitions.


```python
def get_subject_results(subject):
    response = requests.get(
        base_url + "works",
        params={
            "subjects.label": subject,
            "include": "subjects",
            "pageSize": "100",
        },
    ).json()

    progress_bar = tqdm(total=response["totalResults"])
    results = response["results"]
    progress_bar.update(len(response["results"]))

    while "nextPage" in response:
        response = requests.get(response["nextPage"]).json()
        results.extend(response["results"])
        progress_bar.update(len(response["results"]))

    progress_bar.close()
    
    return results


def find_intersecting_subject_concepts(subject_1, subject_2, n=20):
    subject_1_results = get_subject_results(subject_1)
    subject_2_results = get_subject_results(subject_2)
    subject_2_ids = set(result["id"] for result in subject_2_results)

    intersecting_results = [
        result for result in subject_1_results if result["id"] in subject_2_ids
    ]

    concepts = Counter()
    for record in intersecting_results:
        for subject in record["subjects"]:
            for concept in subject["concepts"]:
                concepts.update([concept["label"]])

    return concepts.most_common(n)
```

Calling the `find_intersecting_subject_concepts()` function with any two subjects will return a counter of the most common concepts found on the works which are tagged with both subjects.


```python
find_intersecting_subject_concepts("Europe", "United States")
```


```python
find_intersecting_subject_concepts("Vomiting", "Witchcraft")
```

## Exercises

1. Try running the function with different subjects. Use the API to find two subjects which appear on a few hundred or a few thousand works, and see if you can find the most common concepts which appear on both of them.
2. Adapt the code to compare an arbitrary number of subjects, rather than just two.

