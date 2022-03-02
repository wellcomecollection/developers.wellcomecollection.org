---
title: Catalogue
sidebar_position: 2
---

You can create apps that search across our musuem and library collections using the Catalogue API. It provides data from our different catalogues, matched together and merged into a single view of our collections.

Right now, you can search for works, e.g. books, pictures, manuscripts and archives. If a work has been digitised, the Catalogue API links to IIIF resources that provide access to the images and, where available, full text of the work.

We are working to expand the Catalogue API to include more detailed information about our archive collections, as well as people, organisations, places, subjects and genres.

## How do I use it?

You can just get started! There’s no sign up or authentication required, but you might want to browse the documentation, which shows how you make requests. The best place to begin is to search for a work, then explore how things are connected from there.\n\nThere are some licensing restrictions, as different parts of the data may have different licenses. If it's data that has been created by us, it’s [CC0](https://creativecommons.org/publicdomain/zero/1.0/); if it’s not created by us, then it isn’t. We are working to make data licensing clear on a per work basis; in the meantime, if this is a concern, [please get in touch](mailto:digital@wellcomecollction.org).

## What can I do with it?

We use the API to build [our collections search](https://wellcomecollection.org/works) and you have access to all of the same capabilities that we do. Here are a couple of examples of what other people have done with the data:

* [Sourcera](https://chrome.google.com/webstore/detail/sourcera/jlgcbklkbenknacclbadbhpahmnpkagb) is an add-on for Google Docs that provides access to our image collections. With the add-on enabled you can highlight a word, see a list of relevant images and insert them straight into your document.
* [Europeana](https://www.europeana.eu) used the data for bringing our images into Europeana. Previously they could only show thumbnails of our images, but now they can now use IIIF to display images and associated metadata.

There are numerous ways on how you can search data, such as contributor filters, publication date ranges, and ID search.

As features are released we will document them - and all documented features should be considered stable within a version, and we will adhere to [`semver`](https://semver.org/) versioning for changing these features.

If you spot a feature that undocumented but publically available this will most likely be in testing helping us define how it should work. It should be considered unstable, likely to change, or be removed completely.

If you have an idea for a feature, or something is not working as expected, [do get hold of us](mailto:digital@wellcomecollection.org).

Please note that if you require a full harvest of the catalogue, you should use the [daily snapshot](datasets.md) instead of the API.

## About our collections

Find out more about [what's in our collections](https://wellcomecollection.org/pages/YE99nRAAACMAb7YE).


## Data model




