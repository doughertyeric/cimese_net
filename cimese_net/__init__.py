import pkg_resources
import gzip

DATA_PATH = pkg_resources.resource_filename('cimese_net', 'data/')

# Libraries for load_vgg16
import keras
from keras import applications
# Libraries for extract_features
import numpy as np
# Libraries for extract_clip_encodings
import cv2
# Libraries for load_candidate_encodings
import pickle as dill
# Libraries for build_LSH_Forest
from sklearn.neighbors import LSHForest
# Libraries for clip alignment
from collections import Counter
# Libraries for load_top_model
import os
from keras.models import model_from_json
from keras.models import load_model

def load_vgg16():
    '''
    Inputs: None
    Outputs: pre-trained VGG16 network model
    Purpose: Import the pre-trained VGG16 network from the keras.applications module, then
    remove the two upper layers that would normally transform the features extraction
    vector into a vector of length 1000 that serves the classify the images into the 1000
    categories used for the ImageNet competition. When the predict function is called, the
    new top layer will output a vector of length 4096.
    '''
    model_vgg16_conv = applications.VGG16(weights='imagenet', include_top=True)
    model_vgg16_conv.layers.pop()
    model_vgg16_conv.layers.pop()
    model_vgg16_conv.outputs = [model_vgg16_conv.layers[-1].output]
    model_vgg16_conv.layers[-1].outbound_nodes = []
    return model_vgg16_conv

def load_top_model():
    '''
    Inputs: None; automatically loads model architechture and weights of the trained
    classification layers 
    Outputs: Loaded model file
    Purpose: The siamese structure relies upon the pre-trained top layers that are
    imported here. These effectively serve to classify a set of images as a match or 
    not a match based on the feature vectors of each.
    '''
    MODEL_FILE = pkg_resources.resource_filename('cimese_net', 'data/cimese_net_best_model.h5')
    model = load_model(MODEL_FILE)
    print('Model {file_name} successfully loaded'.format(file_name = 'cimese_net_best_model.h5'))
    return model
    
def extract_features(image, vgg16_model):
    '''
    Inputs: Image file (single frame of video) and the pre-trained VGG16 network model
    Outputs: Feature encoding vector of length 4096
    Purpose: Formats images of size (224,224,3) and runs the resulting vector through the
    predict function of the pre-trained VGG16 neural network.
    '''
    image_features = np.zeros((1, 4096))
    mean_pixel = [103.939, 116.779, 123.68]
    im = image.astype(np.float32, copy=False)
    for c in range(3):
        im[:, :, c] = im[:, :, c] - mean_pixel[c]        
    im = np.expand_dims(im, axis=0)
    image_features[0,:] = vgg16_model.predict(im)[0]
    return image_features
    
def extract_clip_encodings(clip, vgg16_model):
    '''
    Inputs: Video file name and the pre-trained VGG16 network model
    Outputs: List of encoded vectors for frames extracted at a rate of one per second.
    Purpose: Loads the potentially infringing video file, extracts one frame per second,
    resizes the image and runs it through the extract features function. The results are
    appended in a single list (rec_frames). The same function can be called on the
    original (high-quality) movie file to get an initial set of encodings that can be
    loaded using the load_candidate_encodings function.
    '''
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
            small_frame = (small_frame - small_frame.mean()) / small_frame.std()
            rec_frames.append(small_frame)
            count += (fps)
        if count > video_length - (2*fps):
            video.release()
    return rec_frames
    
def load_candidate_encodings(candidate_film, DATA_DIR=os.getcwd()):
    '''
    Inputs: List of encodings of the candidate film and the directory in which this file
    (which should be named MovieName_AllFrames.dill) can be found. Defaults to the current
    working directory.
    Outputs: List of encoded vectors for the full movie.
    Purpose: Loads in the set of previously calculated encodings for the entire movie
    whose copyright is being infringed upon and returns them in the form of a list with
    each entry corresponding to a single frame per second (the output of the
    extract_clip_encodings function).
    '''
    candidate_film = candidate_film.replace(' ', '')
    file_name = str(candidate_film + '_AllFrames.dill')
    try:
        orig_frames = dill.load(open(file_name, 'rb'))
    except Exception:
        orig_frames = dill.load(open(os.path.join(DATA_DIR, file_name), 'rb'))
    return orig_frames
    
def build_LSH_Forest(orig_frames):
    '''
    Inputs: The list of feature encodings for the full movie.
    Outputs: A locality-specific hashing (LSH) forest object (as implemented in the
    scikit-learn.neighbors module)
    Purpose: Efficiently creates a neighbor-based system so that single frames can be
    placed near similar frames in terms of their mutual encodings emerging from the 
    VGG16 network model.
    '''
    lshf = LSHForest(n_estimators=20, n_candidates=1000, random_state=42)
    lshf.fit(orig_frames)
    return lshf
    
def clip_alignment(lshf, rec_frames):
    '''
    Inputs: LSH Forest object and the list of encodings extracted from the potentially
    infringing video clip.
    Outputs: Single value indicated the most likely frame index of the full movie
    correspoding to the first frame of the potentially infringing video clip.
    Purpose: The neural net is most effective when the frames being analyzed are
    relatively closely aligned. Though it is somewhat robust to offsets of about 1 second,
    the optimal performance occurs at the perfect alignment (i.e., perfectly corresponding
    frames) between the full movie and the potentially infringing clip. This function
    serves to align the two as closely as possible based on the similarity in the feature
    vectors of 10 frames at the start of the video clip.
    '''
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
    
def subset_candidate_film(init_frame, orig_frames, rec_frames):
    '''
    Inputs: Predicted inital frame, list of encodings of the original movie, list of
    encodings of the potentially infringing clip.
    Outputs: Subset of the full list of encodings that matches the length of the
    potentially infringing clip.
    Purpose: Once the optimal start point is selected, the list of encodings of the full
    movie is subsetted so that the lengths match for further analysis.
    '''
    clip_length = len(rec_frames)
    subset = orig_frames[init_frame:(init_frame + clip_length + 1)]
    return subset
    
def run_model(input1, input2, model):
    '''
    Inputs: Two vectors of shape (4096,) and the trained model of the top layers of 
    the Siamese network.
    Outputs: Single match prediction based on the classification portion of the Siamese
    network.
    Purpose: The outputs of the VGG16 network from the two frames are compared and a
    single match probability is returned.
    '''
    prediction = model.predict([input1, input2])
    return prediction
    
def prob_determination(prob):
    '''
    Inputs: Vector of frame-by-frame predictions
    Outputs: Single probability of infringement value
    Purpose: Using the full set of match probabilities (i.e., one per second over the
    length of the potentially infringing clip), and overall probability of infringment 
    is calculated.
    '''
    thresh = [1 if x > 0.5 else 0 for x in prob]
    return sum(thresh)/float(len(thresh))
    
def infringement_probability(clip='None', candidate_film='None', test='None', DATA_DIR=os.getcwd()):
    '''
    Inputs: The file name of the potentially infinging clip, the pickled file containing
    the encoding of the full film whose copyright is potentially being violated, and the
    path to the directory containing both files. An option exists to run a test based on
    pre-loaded encoding data on an abridged full movie and two clips: one that matches
    (test = 'POS') and one that does not match (test = 'NEG'). If either of these are
    selected, the clip and candidate_film inputs will be ignored.
    Outputs: Single probability of infringement value
    Purpose: Using all of the functions above, this function returns a single probability
    of infringement based on the similarity of the potentially infringing clip and the set
    of previously extracted feature encodings of the full movie. The function consists of
    an alignment process using the LSH forest and the classification process using the
    Siamese network structure. Note: warnings concerning the deprecation of the LSHForest
    function and the inability to load the optimizer state of the top model layers can be
    safely ignored.
    '''
    # Clip Alignment Processes
    vgg16_model = load_vgg16()

    if test == 'POS':
        print('Loading encodings from abridged Test film ...')
        TEST_FULL = pkg_resources.resource_filename('cimese_net', 'data/Test_AllFrames.data')
        orig_frames = dill.load(gzip.open(TEST_FULL))
        print('Loading encodings from positive match clip ...')
        TEST_POS = pkg_resources.resource_filename('cimese_net', 'data/Pos_AllFrames.data')
        rec_frames = dill.load(gzip.open(TEST_POS))
        
    elif test == 'NEG':
        print('Loading encodings from abridged Test film ...')
        TEST_FULL = pkg_resources.resource_filename('cimese_net', 'data/Test_AllFrames.data')
        orig_frames = dill.load(gzip.open(TEST_FULL))
        print('Loading encodings from negative match clip ...')
        TEST_NEG = pkg_resources.resource_filename('cimese_net', 'data/Neg_AllFrames.data')
        rec_frames = dill.load(gzip.open(TEST_NEG))
        
    else:
        print('Extracting frames from video clip ...')
        rec_frames = extract_clip_encodings(clip, vgg16_model)
        print("Extracted {n_frames} frames from {clip_file}".format(n_frames = len(rec_frames), clip_file=clip))
        orig_frames = load_candidate_encodings(candidate_film, DATA_DIR)
        
    print('Building LSH Forest ...')
    lshf = build_LSH_Forest(orig_frames)
    print('Aligning clip with candidate film ...')
    init_frame = clip_alignment(lshf, rec_frames)
    
    # Classification Model Processes
    print('Subsetting candidate film ...')
    subset = subset_candidate_film(init_frame, orig_frames, rec_frames)
    top_model = load_top_model()
    print('Conducting Classification via CNN ...')
    prob = []
    for i in range(len(subset)-1):
        encoding1 = subset[i].reshape(1,-1)
        encoding2 = rec_frames[i].reshape(1,-1)
        prediction = run_model(encoding1, encoding2, top_model)
        prob.append(prediction[0][1])
    return prob_determination(prob)
