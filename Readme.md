<p float="left">
  <img src="faceExtract.png" width="400" />
  <img src="original.png" width="400" /> 
</p>

# PyFaceExtract

### Description:

This project aims to detect faces in images using the `face_recognition` library and then crop the detected faces. 

Once the faces are detected, the program expands the bounding box around each face to ensure the entire head is included in the cropped result. 

It supports processing multiple images concurrently for enhanced performance.

---

## Steps to Utilize:

### 1. Setup:

Ensure Python is installed on your system. If it's not, download and install the latest Python version from the official website.

__Creating a virtual environment:__

Windows:
```bash
$ python -m venv face_env
```

macOS and Linux:
```bash
$ python3 -m venv face_env
```

__Activating the virtual environment:__

Windows:

```bash
$ .\face_env\Scripts\activate
```


macOS and Linux:
```bash
$ source face_env/bin/activate
```

---

### 2. Installation:

With the virtual environment activated, install the required packages using the following command:

```bash
$ pip install -r requirements.txt
```

Note: You may need additional dependencies for face_recognition to work smoothly on your machine. Refer to its official documentation for details.

---

### 3. Configuration:

The project uses the `src.config` module for its configuration settings. 

Ensure that `settings.PATH_ORIGIN` is set to the directory containing the images to process and `settings.PATH_DESTINY` is set to the directory where you want to save the cropped faces.

If you want to run a test, set `settings.IS_TEST` to `True` and use the `test/origin` and `test/destiny` directories respectively for input and output.

---

### 4. Execution:

Simply run the script:

```bash
$ python main.py
```

The program will detect faces in the images, crop the faces ensuring the entire head is included, and save them to the destination directory. 

Log messages will keep you informed about the progress.

---

### 5. Deactivating the Virtual Environment:

Once you're done, you can deactivate the virtual environment using:
```bash
$ deactivate
```

---

## Additional Notes:

- Ensure you have the necessary permissions to read/write to the directories you're working with.

- Depending on the number of images and their resolutions, processing can be resource-intensive. Adjust the max_workers parameter in the process_directory function if needed. By default, it's set to use all available CPU cores.

*__That's all you need to know to get started with the face detection and cropping project!__*