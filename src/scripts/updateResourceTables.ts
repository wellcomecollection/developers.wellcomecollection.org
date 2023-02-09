import ExecutionEnvironment from "@docusaurus/ExecutionEnvironment";

// Initially from https://gist.github.com/zentala/1e6f72438796d74531803cc3833c039c
// but with modifications
const formatBytes = (bytes: number, sigFigs: number) => {
  if (bytes == 0) {
    return "0 Bytes";
  }
  const sizes = ["Bytes", "KB", "MB", "GB", "TB"];
  const sizeIdx = Math.floor(Math.log(bytes) / Math.log(1000));
  const number = Number((bytes / Math.pow(1000, sizeIdx)).toPrecision(sigFigs));
  return `${number} ${sizes[sizeIdx]}`;
};

type ResourceData = {
  size?: string;
  lastModified?: Date;
};

const getResourceData = async (
  resourceUrl: string
): Promise<ResourceData | undefined> => {
  const resp = await fetch(resourceUrl, { method: "HEAD" });
  if (resp.ok) {
    const contentLength = resp.headers.get("Content-Length");
    const size =
      contentLength !== null
        ? formatBytes(parseInt(contentLength), 3)
        : undefined;
    const lastModifiedHeader = resp.headers.get("Last-Modified");
    const lastModified =
      lastModifiedHeader !== null ? new Date(lastModifiedHeader) : undefined;
    return {
      size,
      lastModified
    };
  }
};

const updateResourceTables = () => {
  const dataLinks = document.querySelectorAll(
    "table a[href^='https://data.wellcomecollection.org']"
  );
  dataLinks.forEach(async link => {
    const linkCell = link.parentNode;
    if (!(linkCell instanceof HTMLTableCellElement)) {
      return;
    }

    const sizeCell = linkCell.previousElementSibling;
    if (!(sizeCell instanceof HTMLTableCellElement)) {
      return;
    }

    const resourceUrl = link.getAttribute("href");
    const resourceData = await getResourceData(resourceUrl);
    if (resourceData?.size) {
      sizeCell.innerText = resourceData.size;
    }
    if (resourceData?.lastModified) {
      const isoString = resourceData.lastModified.toISOString();
      const localeDateString = resourceData.lastModified.toLocaleDateString();
      linkCell.innerHTML += ` <span class="last-updated">(last updated <time datetime=${isoString}>${localeDateString}</time>)</span>`;
    }
  });
};

if (ExecutionEnvironment.canUseDOM) {
  window.addEventListener("load", updateResourceTables);
}
