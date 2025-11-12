# 5. Working with snapshots of the API

[View on GitHub](https://github.com/wellcomecollection/developers.wellcomecollection.org/blob/add-linkedWork-params/notebooks/05-working-with-snapshots-of-the-api.ipynb) | [Run in Google Colab](https://colab.research.google.com/github/wellcomecollection/developers.wellcomecollection.org/blob/add-linkedWork-params/notebooks/05-working-with-snapshots-of-the-api.ipynb)

As we saw at the end of the last notebook, the API limits its responses to 10,000 total results - after that point, users are directed to work with the snapshots. For example, making a request to [https://api.wellcomecollection.org/catalogue/v2/works?pageSize=100&page=101](https://api.wellcomecollection.org/catalogue/v2/works?pageSize=100&page=101) gives us:

```json
{
    "errorType": "http",
    "httpStatus": 400,
    "label": "Bad Request",
    "description": "Only the first 10000 works are available in the API. If you want more works, you can download a snapshot of the complete catalogue: https://developers.wellcomecollection.org/docs/datasets",
    "type": "Error"
}
```

Let's download a snapshot and see what it contains. In later notebooks, we'll make use of the file we're downloading here.

**NB: The uncompressed snapshot is >10GB, so this will take a while to download! Make sure your machine has enough space before running this notebook.**


```python
import requests
import json
from pathlib import Path
from tqdm.auto import tqdm
import gzip
import io
```

The urls for the snapshots can be found at [https://developers.wellcomecollection.org/docs/datasets](https://developers.wellcomecollection.org/docs/datasets). We're going to work with the `works` snapshot, but all of the logic which follows should be extendable to the images snapshot on that page too!

Let's start by establishing the url for the compressed snapshot file, and the path where the data is going to be saved.


```python
snapshot_url = "https://data.wellcomecollection.org/catalogue/v2/works.json.gz"
```

Note that the URL ends with `.gz`, indicating that we're going to be downloading a _zipped_ version of the file, which will need to be unzipped later.

We're going to create a new directory next to these notebooks called `data`, where the zipped file will be saved.


```python
data_dir = Path("./data").resolve()
data_dir.mkdir(exist_ok=True)

file_name = Path(snapshot_url).parts[-1]
zipped_path = data_dir / file_name
```

Let's download the file using the `requests` library, and save it to the path we've just created.


```python
# check whether the file already exists before doing any work
if not zipped_path.exists():

    # make a request to the snapshot URL and stream the response
    r = requests.get(snapshot_url, stream=True)
    
    # use the length of the response to create a progress bar for the download
    download_progress_bar = tqdm(
        unit="bytes",
        total=int(r.headers["Content-Length"]),
        desc=f"Downloading {file_name}",
    )

    # write the streamed response to our file in chunks of 1024 bytes
    with open(zipped_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                download_progress_bar.update(len(chunk))

        download_progress_bar.close()

```

Now that we have the zipped file, we can create a path for the _unzipped_ data to be saved, and use the `gzip` library to unzip the data.


```python
unzipped_path = zipped_path.with_suffix("")

# open the zipped file, and the unzipped file
with gzip.open(zipped_path, "rb") as f_in, open(unzipped_path, "wb") as f_out:
    # measure the length of the zipped file using `.seek()`
    file_length = f_in.seek(0, io.SEEK_END)
    unzip_progress_bar = tqdm(
        unit="bytes",
        total=file_length,
        desc=f"unzipping {file_name}",
    )

    # we used `.seek()` to move the cursor to the end of the file, so we need to
    # move it back to the start before we can read the whole thing file
    f_in.seek(0)

    # read the zipped file in chunks of 1MB
    for chunk in iter(lambda: f_in.read(1024 * 1024), b""):
        f_out.write(chunk)
        unzip_progress_bar.update(len(chunk))

    unzip_progress_bar.close()
```

Great! We've now got a copy of the works snapshot, and we're ready to start exploring it!

The snapshot is saved in [jsonl](http://jsonlines.org/) format; a variant of json where each line is a separate json object. This is a common format for large json files, as it allows the user to read the file line-by-line, rather than having to load the entire file into memory.

Let's have a look at the first line of the file.


```python
with gzip.open(zipped_path, 'rt') as f:
    first_line = f.readline()

print(first_line)
```

That first line is a standalone json document - we can parse it using the `json` library and play around with the keys and values, just as we did with the API responses.


```python
work = json.loads(first_line)
```


```python
work.keys()
```


```python
work["title"]
```

You'll notice that the works in the snapshot include _all_ of the fields which were available in the default API response, _and_ all of the optional fields too. The snapshots include the complete set of information we have about our works, and are a great way to get a complete picture of the collection.


```python
unzipped_path.unlink(missing_ok=True)
```

## Wrapping up

Let's delete the snapshot we've downloaded, and wrap up all of the logic we've established so far into a single cell which we can copy and reuse in later notebooks.


```python
import requests
from pathlib import Path
from tqdm.auto import tqdm
import gzip
import io

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
        download_progress_bar.close()
        
    with gzip.open(zipped_path, "rb") as f_in, open(unzipped_path, "wb") as f_out:
        unzip_progress_bar = tqdm(
            unit="B",
            total=f_in.seek(0, io.SEEK_END),
            desc=f"unzipping {file_name}",
        )
        f_in.seek(0)
        for chunk in iter(lambda: f_in.read(1024 * 1024), b""):
            f_out.write(chunk)
            unzip_progress_bar.update(len(chunk))
    
        unzip_progress_bar.close()
    zipped_path.unlink()
```

We can check the location of the unzipped file.


```python
print(unzipped_path)
```

## Exercises

1. Count how many lines exist in the file, without loading the entire file into memory.
2. Load the first 10 lines of the file into memory, and print them out.
3. Adapt the code to download the images snapshot instead, and repeat the exercises above.
