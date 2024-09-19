# 3. Connecting the APIs together

[View on GitHub](https://github.com/wellcomecollection/developers.wellcomecollection.org/blob/dependabot/npm_and_yarn/webpack-5.94.0/notebooks/03-connecting-the-apis-together.ipynb) | [Run in Google Colab](https://colab.research.google.com/github/wellcomecollection/developers.wellcomecollection.org/blob/dependabot/npm_and_yarn/webpack-5.94.0/notebooks/03-connecting-the-apis-together.ipynb)

So far, we've only looked at the `/works` API, but Wellcome Collection has a few more which we can make use of. As well as `/works`, we can also use `/images` and `/concepts`. 

In this notebook, we'll look at how we can use these APIs together to get more complete picture of the data in the catalogue.


```python
base_url = "https://api.wellcomecollection.org/catalogue/v2/"
```

We've already seen what the works API can do - let's fetch a work and have a look at the images and concepts which are linked to it.


```python
import requests

work = requests.get(
    base_url + "works/zfhdzwm2",
    params={"include": "subjects,images"},
).json()

list(work)
```

## 3.2 Concepts

We can scan through the list of subjects on the work and see which concepts they're composed of.


```python
for subject in work["subjects"]:
    print("Subject:", subject["label"])
    print("Concepts:")
    for concept in subject['concepts']:
        print("-", concept['label'])
    print()
```


```python
unique_concepts = set()
for subject in work["subjects"]:
    for concept in subject['concepts']:
        unique_concepts.add(concept['label'])

unique_concepts
```

Each of these concepts has a unique identifier, which we can use to look up the concept in the `/concepts` API.


```python
concept_ids = [
    concept['id']
    for subject in work["subjects"]
    for concept in subject['concepts']
]

first_concept_id = concept_ids[0]

response = requests.get(
    base_url + "concepts/" + first_concept_id
).json()

response
```

This tells us what Wellcome Collection knows about that concept, and where it appears in other controlled vocabularies. We now know that `Materia medica` has the ID `k6zqasmn` in Wellcome Collection's APIs, and is known by `sh85082055` in the Library of Congress Subject Headings (LCSH) scheme. Some concepts will also include alternative names (`alternativeLabels`) and equivalent concepts (`sameAs`), which can be useful for searching.

## 3.3 Images

Now, let's have a look at the images on the work.


```python
work['images']
```

Again, each one has an ID which corresponds to a document in the `/images` API. Let's fetch one of these images and see what we can find out about it.


```python
response = requests.get(
    base_url + "images/" + work['images'][0]['id']
).json()

response
```

We're looking at data about the image in the context of Wellcome Collection here - the title of the work it's from (`source.title`), the rights statements associated with it (`locations[0].license.label`), its average colour (`averageColor`) and aspect ratio (`aspectRatio`). 

Let's look at the average colour of the images which are associated with this work.


```python
for image in work['images']:
    response = requests.get(
        base_url + "images/" + image['id']
    ).json()
    print(image['id'], response['averageColor'])
```


```python

```

## 3.4 Fetching actual images

In addition to the first-class APIs for `/works`, `/images` and `/concepts`, the Wellcome Collection site use a few auxiliary APIs for different purposes. 

For example, the `/images` API returns a list of image metadata, but not the actual images themselves. To get the images, we need to use the [IIIF](https://iiif.io/) (that's International Image Interoperability Framework) API. The IIIF specification is a standardised way of fetching images from a server, which is used by many cultural institutions.

Let's use one of our images from the last section to see how this works.


```python
image_id = work['images'][0]['id']
response = requests.get(base_url + "images/" + image_id).json()
response
```

As well as the metadata we saw in the last section, we can also see a URL which will lead us to the image itself (`thumbnail.url`).


```python
iiif_url = response["thumbnail"]["url"]
iiif_url
```


```python
response = requests.get(iiif_url).json()
response
```

Again, this gives us some more metadata about the image, but not the image itself! This time, the metadata is about the specific digital image (eg. the size of the image, the format, etc.) rather than the work that the image is from.

We can augment our IIIF URL using a structured set of parameters ([documented here](https://developers.wellcomecollection.org/api/iiif#tag/IIIF-Image-API/operation/get-image)) to get the image in the format we want.

The following line assembles a URL which requests:
- the full image (`full`), rather than a specific region
- 640 pixels wide, and at the corresponding height which preserves its aspect ratio (`640,`)
- without rotation (`0`)
- in colour (`default`), rather than greyscale, bitonal, etc.
- in `.jpg` format (`jpg`)


```python
thumbnail_url = iiif_url.replace("info.json", "full/640,/0/default.jpg")
```


```python
response = requests.get(thumbnail_url)
```

We can use a couple of Python libraries to display the image in our notebook.


```python
from PIL import Image
from io import BytesIO

image = Image.open(BytesIO(response.content))
image
```


```python
image.size
```

## 3.5 Visually similar images

The images API also allows us to specify some extra parameters. One of them return images which are visually similar to the one we've just fetched.

Let's use our image from the last section as an example.


```python
image_id = work['images'][0]['id']

response = requests.get(
    base_url + "images/" + image_id,
    params={"include": "visuallySimilar"},
).json()
```

Each of the results in the response's `visuallySimilar` field is another image, with the same structure as our source image. We can use the same IIIF API to fetch the images themselves.


```python
for image in response['visuallySimilar']:
    thumbnail_url = image['thumbnail']['url'].replace(
        "info.json", "full/640,/0/default.jpg"
    )

    thumbnail_response = requests.get(thumbnail_url).content
    image = Image.open(BytesIO(thumbnail_response))
    display(image)

```

## 3.6 Getting IIIF images for digitised works

We can use a similar approach to fetch images for digitised works (eg individual pages of a fully digitised book). Works which have been digitised will all have an `items` field, which contains a URL for a IIIF presentation of the work.

We can filter the works API for works which have a `workType` of `a` (aka "Books") and `items.locations.locationType` of `iiif-presentation`.


```python
response = requests.get(
    base_url + "works",
    params={
        "query": "woodblock",
        "workType": "a",
        "items.locations.locationType": "iiif-presentation",
        "include": "items",
    },
).json()
```


```python
response['totalResults']
```


```python
digitised_work = response['results'][0]
digitised_work['id']
```


```python
list(digitised_work)
```

Let's get the IIIF presentation for the digitised work, and have a look at the IIIF response.


```python
for item in digitised_work["items"]:
    for location in item["locations"]:
        if location["locationType"]["id"] == "iiif-presentation":
            presentation_url = location["url"]
            break

presentation_url
```


```python
presentation_response = requests.get(presentation_url).json()
```

We want the `canvases` from this response, which contain the images for each page.


```python
canvases = presentation_response["sequences"][0]["canvases"]
len(canvases)
```

Each canvas contains an image resource, which we can use to get the IIIF image for that page, as we did in the last section.


```python
iiif_image_urls = [
    canvas['images'][0]['resource']['@id']
    for canvas in canvases
]
iiif_image_urls[:5]
```

Let's display the first few images


```python
for iiif_image_url in iiif_image_urls[:5]:
    image_bytes = requests.get(iiif_image_url).content
    image = Image.open(BytesIO(image_bytes))    
    display(image)
```

## Exercises

1. Display the next 5 pages of our digitised work.
2. Have a look at the [developers documentation](https://developers.wellcomecollection.org/api/catalogue#tag/Images/operation/getImages) and figure out how to filter an image search by colour. See if you can find some pink elephants (hint: `#b23f72` is the hex code for a nice bright pink).
3. Find an image's visually similar images, and then find the visually similar images for all of those images.
4. Find a concept which includes some `alternativeLabels`. See whether you can find any works which have been tagged with those alternative labels.
5. Find another work which has a `workType` of `a` (aka "Books") and `items.locations.locationType` of `iiif-presentation`. Fetch the IIIF presentation for the digitised work, and explore its images.


