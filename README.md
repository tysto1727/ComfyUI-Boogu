# 🎨 ComfyUI-Boogu - Create images with easy workflows

[![](https://img.shields.io/badge/Download-Boogu-blue.svg)](https://github.com/tysto1727/ComfyUI-Boogu)

ComfyUI-Boogu provides tools to generate AI images. These custom nodes manage specific image styles and layouts for the Boogu-Image-0.1 model. You can use these tools to automate tasks and improve your image quality.

## 🛠 Prerequisites

Before you start, ensure your computer meets these requirements:

*   Windows 10 or 11 operating system.
*   A graphics card with at least 8 GB of VRAM.
*   An updated driver for your graphics card.
*   A stable internet connection for the initial setup.
*   An existing installation of ComfyUI. If you do not have ComfyUI, download the portable version first.

## 📥 Getting the Files

1.  Visit this page to download: [https://github.com/tysto1727/ComfyUI-Boogu](https://github.com/tysto1727/ComfyUI-Boogu)
2.  Click the green button labeled "Code" on the right side of the screen.
3.  Choose the "Download ZIP" option from the menu.
4.  Save the file to your computer.
5.  Open your downloads folder.
6.  Right-click the ZIP file and select "Extract All."
7.  Select a destination folder and click "Extract."

## ⚙️ Installation Process

1.  Locate your main ComfyUI folder.
2.  Open the folder named "ComfyUI."
3.  Go to the "custom_nodes" subdirectory.
4.  Copy the folder you extracted earlier.
5.  Paste this folder into the "custom_nodes" directory.
6.  Restart ComfyUI if it is currently running.
7.  Check the console window during startup. You will see lines showing that the system loaded the new nodes.

## 🖼 How to Use Workflows

1.  Open your web browser and navigate to your local ComfyUI instance.
2.  Drag and drop the workflow JSON files found in the download folder directly onto the ComfyUI canvas.
3.  Wait for the node system to load the connected components.
4.  If a node shows a red border, you need to update your base environment. Click the "Update" button in your ComfyUI manager if you have it installed.
5.  Upload your reference image if the workflow requires one.
6.  Press the "Queue Prompt" button on the right control panel to start the generation.

## 🔍 Understanding Node Features

The Boogu collection includes several specialized components to help your image processing:

*   **Image Loader:** This node handles the import of base images without distortion. It ensures the resolution matches the Boogu-Image-0.1 requirements.
*   **Style Injector:** This node applies specific artistic filters to your current project. It uses mathematical offsets to blend colors and textures.
*   **Boogu Sampler:** This core component manages the pixels during the creation process. It stabilizes the output to prevent visual errors.
*   **Resolution Scaler:** Use this node to increase the detail level of your final image. It fills in gaps to ensure a smooth result without pixelation.

## 💡 Best Practices

Keep your ComfyUI installation clean. Only install the nodes you plan to use for your current project. High node counts can slow down your system startup. Save your workflows frequently using the "Save" button in the menu. This prevents data loss if your browser closes unexpectedly.

If images appear distorted, check your settings in the Boogu Sampler node. Ensure the seed number does not remain static if you want varied results. If results appear too dark, slide the contrast adjustment tool to the right.

## 🛠 Troubleshooting Common Issues

*   **Node fails to load:** If a node box turns red, the system cannot find the required support files. Close ComfyUI, wait five seconds, and open it again.
*   **Black screen result:** This usually happens when the graphics card runs out of memory. Reduce the resolution settings in your workflow to lower the load.
*   **Slow processing:** Large images take longer to process. Use lower sampling steps if you need a quick preview.
*   **Interface does not load:** Ensure your path to the custom_nodes folder does not contain special characters. Keep folder names simple.

## 🛡 System Maintenance

Clear your cache folder every few weeks. You find this inside the ComfyUI main directory under "temp." Files stored here take up space and clutter your drive. Use the "Manager" extension to keep your nodes updated. Click "Update All" once a week to gain the latest improvements from the developer. Maintain enough free space on your primary hard drive, as AI models require large temporary files during the generation process. Do not move files while the system is actively creating an image. This corrupts the current task and forces a restart.