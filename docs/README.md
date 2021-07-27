# Instructions for files in OG-Core/docs
The files in this directory `OG-Core/docs/` include images and all the files necessary for rendering the Jupyter Book documentation. One image [`OG-Core_logo_gitfig.png`](docs/OG-Core_logo_gitfig.png) is only used for the GitHub social preview image. GitHub suggests that this image should be 1280x640px for best display. The image we created [`OG-Core_logo_long.png`](docs/OG-Core_logo_long.png) is natively 500x346px. We do the following to resize the image.

1. Open the image in Adobe Photoshop: **File** > **Open**
2. Open the **Image Size** dialogue:
3. Adjust the canvas size: **Image** > **Canvas Size**. Because the 500x346px image is taller than the optimal 1280x640px GitHub size, we first adjust the canvas size. We have to add some width. So here adjust the width to 692px [`346 * 2` or `346 * (1280 / 640)`] and keep the height at 346px.
4. Adjust the image size: **Image** > **Image Size**. Now adjust the image size to the GitHub optimal 1280x640px. The dimesions will be correct and nothing will be stretched.
5. Save the image as [`OG-Core_logo_gitfig.png`](docs/OG-Core_logo_gitfig.png).
6. Upload the image [`OG-Core_logo_gitfig.png`](docs/OG-Core_logo_gitfig.png) as the GitHub social preview image by clicking on the [**Settings**](https://github.com/PSLmodels/OG-Core/settings) button in the upper-right of the main page of the repository and uploading the formatted image [`OG-Core_logo_gitfig.png`](docs/OG-Core_logo_gitfig.png) in the **Social preview** section.
