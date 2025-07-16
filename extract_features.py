import shutil
import tqdm
import numpy as np
import cv2
import os
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
import config


def video_to_frames(video):
    path = os.path.join(config.test_path, 'temporary_images')
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    video_path = os.path.join(config.test_path, 'video', video)
    count = 0
    image_list = []
    # Path to video file
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break
        cv2.imwrite(os.path.join(config.test_path, 'temporary_images', 'frame%d.jpg' % count), frame)
        image_list.append(os.path.join(config.test_path, 'temporary_images', 'frame%d.jpg' % count))
        count += 1

    cap.release()
    cv2.destroyAllWindows()
    return image_list


def model_cnn_load():
    model = VGG16(weights="imagenet", include_top=True)  # include_top=True gives 4096-dim features
    out = model.get_layer("fc2").output                  # fc2 layer gives (4096,)
    model_final = Model(inputs=model.input, outputs=out)
    return model_final



def load_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    return img


def extract_features(video, model):
    """
    Extract features frame-by-frame (to reduce memory usage)
    :param video: The video whose frames are to be extracted to convert into a numpy array
    :param model: the pretrained VGG16 model
    :return: numpy array of size (num_frames, 4096)
    """
    video_id = video.split(".")[0]
    print(video_id)
    print(f'Processing video {video}')

    image_list = video_to_frames(video)

    # Sample 80 evenly spaced frames
    if len(image_list) < 80:
        print("⚠️ Warning: Not enough frames. Skipping video.")
        return None
    samples = np.round(np.linspace(0, len(image_list) - 1, 80))
    image_list = [image_list[int(sample)] for sample in samples]

    img_feats = []
    for image_path in image_list:
        img = load_image(image_path)
        img = np.expand_dims(img, axis=0)  # Shape: (1, 224, 224, 3)
        feature = model.predict(img)       # Shape: (1, 4096)
        img_feats.append(feature[0])       # Flatten

    # cleanup
    shutil.rmtree(os.path.join(config.test_path, 'temporary_images'))

    return np.array(img_feats)


def extract_feats_pretrained_cnn():
    """
    saves the numpy features from all the videos
    """
    model = model_cnn_load()
    print('Model loaded')

    if not os.path.isdir(os.path.join(config.test_path, 'feat')):
        os.mkdir(os.path.join(config.test_path, 'feat'))

    video_list = os.listdir(os.path.join(config.test_path, 'video'))
    
    #ًWhen running the script on Colab an item called '.ipynb_checkpoints' 
    #is added to the beginning of the list causing errors later on, so the next line removes it.
    if '.ipynb_checkpoints' in video_list:
        video_list.remove('.ipynb_checkpoints')

    
    for video in video_list:

        outfile = os.path.join(config.test_path, 'feat', video + '.npy')
        img_feats = extract_features(video, model)
        if img_feats is not None:
            np.save(outfile, img_feats)



if __name__ == "__main__":
    extract_feats_pretrained_cnn()
