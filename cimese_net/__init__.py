import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Libraries for load_vgg16
import keras
from keras import applications
# Libraries for extract_features
import numpy as np
# Libraries for extract_clip_encodings
import cv2
import dlib
# Libraries for load_candidate_encodings
import pickle as dill
# Libraries for build_LSH_Forest
from sklearn.neighbors import LSHForest
# Libraries for clip alignment
from collections import Counter
# Libraries for load_top_model
import os
from keras.models import load_model

def load_vgg16():
    model_vgg16_conv = applications.VGG16(weights='imagenet', include_top=True)
    model_vgg16_conv.layers.pop()
    model_vgg16_conv.layers.pop()
    model_vgg16_conv.outputs = [model_vgg16_conv.layers[-1].output]
    model_vgg16_conv.layers[-1].outbound_nodes = []
    return model_vgg16_conv
    
def extract_features(image, vgg16_model):
    image_features = np.zeros((1, 4096))
    mean_pixel = [103.939, 116.779, 123.68]
    im = image.astype(np.float32, copy=False)
    for c in range(3):
        im[:, :, c] = im[:, :, c] - mean_pixel[c]        
    im = np.expand_dims(im, axis=0)
    image_features[0,:] = vgg16_model.predict(im)[0]
    return image_features
    
def extract_clip_encodings(clip, vgg16_model):
    video = cv2.VideoCapture(clip)
    fps = video.get(5)
    fps = round(fps)
    video_length = video.get(7)

    count = 0
    rec_frames = []
    while(video.isOpened()):
        ret, frame = video.read()
        if video.get(1) % (fps) == 0:
            small_frame = cv2.resize(frame, (224, 224))
            small_frame = extract_features(small_frame, vgg16_model)[0]
            rec_frames.append(small_frame)
            count += (fps)
        if count > video_length - (2*fps):
            video.release()
    return rec_frames
    
def load_candidate_encodings(candidate_film):
    candidate_film = candidate_film.replace(' ', '')
    file_name = str(candidate_film + '_AllFrames.dill')
    orig_frames = dill.load(open(file_name, 'rb'))
    return orig_frames
    
def build_LSH_Forest(orig_frames):
    lshf = LSHForest(n_estimators=20, n_candidates=1000, random_state=42)
    lshf.fit(orig_frames)
    return lshf
    
def clip_alignment(lshf, rec_frames):
    top_10 = []
    for i in range(10):
        frame = np.array(rec_frames[i]).reshape(1,-1)
        distances, indices = lshf.kneighbors(frame, n_neighbors=5)
        for j in range(len(indices[0])):
            temp_range = range(int(indices[0][j])-i, int(indices[0][j]+(10-i)),1)
            top_10.append(temp_range)
    first_frame = [row[0] for row in top_10]
    frame_freq = Counter(first_frame)
    return frame_freq.most_common()[0][0]
    
def load_top_model(DATA_DIR):
    model_file = os.path.join(DATA_DIR, "models", "vgg16-cat-final.h5")
    model = load_model(model_file)
    return model
    
def subset_candidate_film(init_frame, orig_frames, rec_frames):
    clip_length = len(rec_frames)
    subset = orig_frames[init_frame:(init_frame + clip_length + 1)]
    return subset
    
def run_model(input1, input2, model):
    prediction = model.predict([input1, input2])
    return prediction
    
def prob_determination(prob):
    thresh = [1 if x > 0.5 else 0 for x in prob]
    return sum(thresh)/float(len(thresh))
    
def infringement_probability(clip, candidate_film, DATA_DIR):
    # Clip Alignment Processes
    vgg16_model = load_vgg16()
    print('Extracting frames from video clip ...')
    rec_frames = extract_clip_encodings(clip, vgg16_model)
    print("Extracted {n_frames} frames from {clip_file}".format(n_frames = len(rec_frames), clip_file=clip))
    orig_frames = load_candidate_encodings(candidate_film)
    print('Building LSH Forest ...')
    lshf = build_LSH_Forest(orig_frames)
    print('Aligning clip with candidate film ...')
    init_frame = clip_alignment(lshf, rec_frames)
    
    # Classification Model Processes
    print('Subsetting candidate film ...')
    subset = subset_candidate_film(init_frame, orig_frames, rec_frames)
    top_model = load_top_model(DATA_DIR)
    print('Conducting Classification via CNN ...')
    prob = []
    for i in range(len(subset)-1):
        encoding1 = subset[i].reshape(1,-1)
        encoding2 = rec_frames[i].reshape(1,-1)
        prediction = run_model(encoding1, encoding2, top_model)
        prob.append(prediction[0][1])
    return prob_determination(prob)