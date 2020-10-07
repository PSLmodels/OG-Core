# Instructions for files in OG-USA/docs
The files in this directory `OG-USA/docs/` include images and all the files necessary for rendering the Jupyter Book documentation. One image [`OG-USA_logo_gitfig.png`](docs/OG-USA_logo_gitfig.png) is only used for the GitHub social preview image. GitHub suggests that this image should be 1280x640px for best display. The image we created [`OG-USA_logo_long.png`](docs/OG-USA_logo_long.png) is natively 2083x1334px. We do the following to resize the image.

1. Open the image in Adobe Photoshop: **File** > **Open**
2. Open the **Image Size** dialogue:
3. Adjust the canvas size: **Image** > **Canvas Size**. Because the 2083x1334px image is taller than the optimal 1280x640px GitHub size, we first adjust the canvas size. We have to add some width. So here adjust the width to 2668px [`(1334 / 640) * 1280`] and keep the height at 1334px.
4. Adjust the image size: **Image** > **Image Size**. Now adjust the image size to the GitHub optimal 1280x640px. The dimesions will be correct and nothing will be stretched.
5. Save the image as [`OG-USA_logo_gitfig.png`](docs/OG-USA_logo_gitfig.png).
6. Upload the image [`OG-USA_logo_gitfig.png`](docs/OG-USA_logo_gitfig.png) as the GitHub social preview image by clicking on the [**Settings**](https://github.com/PSLmodels/OG-USA/settings) button in the upper-right of the main page of the repository and uploading the formatted image [`OG-USA_logo_gitfig.png`](docs/OG-USA_logo_gitfig.png) in the **Social preview** section.
