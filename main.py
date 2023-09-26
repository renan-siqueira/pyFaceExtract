import os
import logging
import time
import gc

from concurrent.futures import ProcessPoolExecutor

import cv2
import dlib
import face_recognition

from src.config import settings


print('\n\t\tDlib Version:', dlib.__version__)
print('\t\t GPU AVAILABLE:', dlib.DLIB_USE_CUDA, '\n')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def image_paths_generator(src_dir):
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                yield os.path.join(root, file)


def create_directories(paths_list):
    for path in paths_list:
        os.makedirs(path, exist_ok=True)


def detect_faces(image_path, resize_scale):
    logging.info(f"Detecting faces in the image: {image_path}")
    
    image = cv2.imread(image_path)
    
    if resize_scale:
        image = cv2.resize(image, (int(image.shape[1] * resize_scale), int(image.shape[0] * resize_scale)))
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image, model="cnn")
    del rgb_image
    gc.collect()
    
    return image, face_locations


def save_faces(image, face_locations, save_path):
    logging.info(f"Saving {len(face_locations)} detected faces in: {save_path}")
    
    base_path, extension = os.path.splitext(save_path)
    
    for index, (top, right, bottom, left) in enumerate(face_locations):
        # Calculating face width and height
        face_width = right - left
        face_height = bottom - top
        
        # Determining the larger dimension
        max_dim = max(face_width, face_height)
        
        # Calculating the center of the face
        center_y, center_x = (top + bottom) // 2, (left + right) // 2
        
        # Expanding the cutout by 50% to ensure the whole head is included
        expanded_dim = int(max_dim * 2)
        
        # Adjusting the cutout to center the face and include the entire head
        top = max(0, center_y - expanded_dim // 2)
        bottom = min(image.shape[0], center_y + expanded_dim // 2)
        left = max(0, center_x - expanded_dim // 2)
        right = min(image.shape[1], center_x + expanded_dim // 2)
        
        # Cropping the image
        face_image = image[top:bottom, left:right]
        
        # If there's more than one face, add a suffix to the filename
        if len(face_locations) > 1:
            new_save_path = f"{base_path}_{index + 1}{extension}"
        else:
            new_save_path = save_path

        cv2.imwrite(new_save_path, face_image)


def process_image(src_path, dest_dir, src_dir, resize_scale):
    start_time = time.time()

    dest_path = os.path.join(dest_dir, os.path.relpath(src_path, src_dir))
    dest_folder = os.path.dirname(dest_path)
    os.makedirs(dest_folder, exist_ok=True)

    image, face_locations = detect_faces(src_path, resize_scale)

    if face_locations and resize_scale:
        face_locations = [(int(top / resize_scale), int(right / resize_scale), int(bottom / resize_scale), int(left / resize_scale)) for (top, right, bottom, left) in face_locations]
        del image
        gc.collect()
        image = cv2.imread(src_path)

    if face_locations:
        save_faces(image, face_locations, dest_path)
        elapsed_time = time.time() - start_time
        logging.info(f"\tImage {src_path} processed in {elapsed_time:.2f} seconds.")

    del image, face_locations
    gc.collect()


def process_directory(src_dir, dest_dir, max_workers, resize_scale):
    os.makedirs(dest_dir, exist_ok=True)

    logging.info(f"Processing directory: {src_dir}")
    logging.info(f"Running with {max_workers} simultaneous processes.")

    paths_generator = image_paths_generator(src_dir)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for image_path in paths_generator:
            executor.submit(process_image, image_path, dest_dir, src_dir, resize_scale)


if __name__ == '__main__':

    logging.info(f"\nNumber of available CPUs: {os.cpu_count()}\n")
    resize_scale = settings.RESIZE_SCALE
    max_workers = settings.MAX_WORKERS

    if settings.IS_TEST:
        directories = ['test/origin', 'test/destiny']
        create_directories(directories)
        process_directory(directories[0], directories[1], max_workers, resize_scale=resize_scale)
    else:
        process_directory(settings.PATH_ORIGIN, settings.PATH_DESTINY, max_workers, resize_scale=resize_scale)
        
    logging.info("\nProcessing Finished!")
