# 1. Exploring Wellcome Collection's APIs

[View on GitHub](https://github.com/wellcomecollection/developers.wellcomecollection.org/blob/upgrade-everything/notebooks/01-exploring-wellcome-collections-apis.ipynb) | [Run in Google Colab](https://colab.research.google.com/github/wellcomecollection/developers.wellcomecollection.org/blob/upgrade-everything/notebooks/01-exploring-wellcome-collections-apis.ipynb)

Wellcome collection has a few public APIs which can be used to fetch things like works, images, and concepts. They all live behind the following base URL


```python
base_url = "https://api.wellcomecollection.org/catalogue/v2/"
```

The APIs are primarily built to serve the [Wellcome Collection website](https://wellcomecollection.org/), but they are also available for anyone to use! They're a great way to get access to the data that Wellcome Collection has about its collections programmatically.

## 1.1 Making requests

We can make requests to that base API URL using the `requests` library. Let's have a look at the `/works` endpoint first.


```python
import requests

response = requests.get(base_url + "works")
response.status_code
```

The response has a `200` status code, which indicates that the works API has responded successfully.

Let's have a look at the fields it gives us.


```python
list(response.json())
```

Let's look at everything _except_ the `results` field for now


```python
for key, value in response.json().items():
    if key != "results":
        print(key, value)
```

1,160,572 works! That's a lot of works. We're only seeing 10 in this response though, because the `pageSize` is set to 10 by default.

Let's have a look at the fields in the first result.


```python
results = response.json()["results"]
first_result = results[0]
list(first_result)
```

And here's the full first result, with all of its values.


```python
first_result
```

## 1.2 Requesting individual works

We can make requests for individual works by adding an ID to the end of our works API URL. Here's the first work again, but this time we're requesting it by ID.


```python
first_work_id = results[0]["id"]
work_url = base_url + "works/" + first_work_id
work_url
```


```python
response = requests.get(work_url).json()
response
```

As expected, the data is the same as the first result in the previous response.

## 1.3 Sorting and searching

By default, works are sorted by the alphabetical order of their IDs (so we're seeing `a222wwjt` first, followed by other works starting with `a22...`).


```python
response = requests.get(base_url + "works").json()

for work in response["results"]:
    print(work["id"])
```

We can add a `query` query parameter to our request to see results sorted by relevance. Let's search for works that contain the word "horse".


```python
response = requests.get(base_url + "/works", params={"query": "horse"}).json()
for i, result in enumerate(response["results"]):
    print(f"{i+1}. {result['title']}")
    print(f"   https://wellcomecollection.org/works/{result['id']}")
    print()
```

Here, the results are sorted by how relevant they are to the search term.

We can also sort the results by other fields. Let's try sorting by when the works were produced, using the `production.dates` field. We'll also add an `include` parameter to our request, so that we can see the `production.dates` field in the results.


```python
response = requests.get(
    base_url + "/works",
    params={
        "query": "horse",
        "sort": "production.dates",
        "include": "production",
    },
).json()
```

The `production` field is an array of `ProductionEvent` objects, each of which has:
- a `label`
- a list of `agents`
- a list of `dates`
- a list of `places`

We can see those for the first result in the list like this:


```python
response["results"][0]["production"]
```

Internally, each `productionEvent` date has a start and an end (because often we don't know _exactly_ when a work was produced). The works for our request are sorted by the earliest _start_ date in their `production.dates` field.


```python
for i, result in enumerate(response["results"]):
    print(f"{i+1}. {result['title']}")
    print(f"   {result['production'][0]['dates'][0]['label']}")
    print(f"   https://wellcomecollection.org/works/{result['id']}")
    print()
```

Those results are in ascending order, but we can also change the `sortOrder` to give us newer works first.


```python
response = requests.get(
    base_url + "/works",
    params={
        "query": "horse",
        "sort": "production.dates",
        "sortOrder": "desc",
        "include": "production",
    },
).json()

for i, result in enumerate(response["results"]):
    print(f"{i+1}. {result['title']}")
    print(f"   {result['production'][0]['dates'][0]['label']}")
    print(f"   https://wellcomecollection.org/works/{result['id']}")
    print()
```

## 1.4 Filtering results

We can ask the API to return works between a set of dates, using the `production.dates.from` and `production.dates.to` parameters. Let's ask for works produced between 1900 and 1910.


```python
response = requests.get(
    base_url + "/works",
    params={
        "production.dates.from": "1900-01-01",
        "production.dates.to": "1910-01-01",
    },
).json()

response["totalResults"]
```

Previously, we used the `production.dates` to _sort_ our results. 

Here, the results are sorted in the default order (ie sorted by `id`), but they're _filtered_ to only show works which were produced in the range we're interested in.

We can also filter by lots of other fields, like subjects! Let's ask for works about cats, by using the `subjects.label` field.


```python
response = requests.get(
    base_url + "/works",
    params={
        "subjects.label": "Cats",
    },
).json()

response["totalResults"]
```



## 1.5 Including extra fields in the response

We can ask the API to give us extra information in the response by adding an `include` query parameter to our request, as we did above to get our `production` events for each work.

There are lots of other fields we can request for each work:

- `identifiers`
- `items`
- `holdings`
- `subjects`
- `genres`
- `contributors`
- `production`
- `languages`
- `notes`
- `images`
- `succeededBy`
- `precededBy`
- `partOf`
- `parts`

The full documentation for each of them is available in the [API documentation](https://developers.wellcomecollection.org/api/catalogue#tag/Works/operation/getWorks).

Let's have a look at `subjects`, as an example.


```python
response = requests.get(
    base_url + "/works",
    params={
        "include": "subjects",
    },
).json()
```

The `subjects` field is an array of `Subject` objects, each of which has:
- a `label`
- an `id`
- a list of `concepts`, where each concept has
    - a `label`
    - an `id`
    - a `type`, eg `Concept`, `Period`, `Person`, `Place`

We can see those for the first result in the response like this:


```python
response["results"][0]["subjects"]
```

In the next notebook, we'll start requesting bigger batches of data and doing some more local data science and analysis to answer some more interesting questions. In the meantime, here are some exercises to test your understanding of what we've covered so far.


## Exercises

1. Fetch the data for the work with the id `ca5c6h4x`
2. Make a request for a work which includes all of its `genres` (these are the types/techniques of the work, eg `painting`, `etching`, `poster`)
3. Find the oldest and newest work about `pigs` in the collection
4. Filter the works about `pigs` to only include those that were produced in the 20th century


