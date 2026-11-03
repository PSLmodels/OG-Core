/* Sphinx pages are separate static documents, outside the MyST router. */
(() => {
  const script = document.currentScript;
  const basePath = new URL(script.src).pathname.replace(
    /\/ogcore-navigation\.js$/, ""
  );

  document.addEventListener("click", (event) => {
    if (event.button !== 0 || event.metaKey || event.ctrlKey ||
        event.shiftKey || event.altKey) return;
    const link = event.target.closest?.("a[href]");
    if (!link || link.hasAttribute("download") ||
        (link.target && link.target !== "_self")) return;
    const url = new URL(link.href, document.baseURI);
    if (url.origin !== window.location.origin) return;
    if (!url.pathname.startsWith(`${basePath}/content/api/`) &&
        url.pathname !== `${basePath}/index.html`) return;

    event.preventDefault();
    event.stopPropagation();
    window.location.assign(url.href);
  }, true);
})();
