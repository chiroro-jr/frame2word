import numpy as np
import importlib
from keras.preprocessing.image import img_to_array
from keras.saving import load_model
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.sequence import pad_sequences
from tempfile import NamedTemporaryFile
from pickle import load


def get_frames(uploaded_video):
    cv2 = importlib.import_module("cv2")

    # store the uploaded video contents in a temporary file
    temp_file = NamedTemporaryFile(delete=False)
    with temp_file as file:
        file.write(uploaded_video.getvalue())

    # open the temporary video using OpenCV
    cap = cv2.VideoCapture(temp_file.name)

    # Select 5 frames evenly spaced throughout the video
    frame_indices = np.linspace(0, cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1, num=5, dtype=int)

    # get 10 frames from the video to use to generate a summary
    frames = list()
    for index in frame_indices:
        # set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)

        # read the frame
        ret, frame = cap.read()

        if ret:
            frames.append(frame)

    # release video capture object
    cap.release()
    del cv2
    return frames


def index_to_word(provided_idx, tokenizer):
    for word, idx in tokenizer.word_index.items():
        if idx == provided_idx:
            return word

    return None


def cleanup_caption(caption: str):
    # remove the start of sequence token
    index = caption.find('startseq ')
    if index > -1:
        caption = caption[len('startseq'):]

    # remove the end of sequence token
    index = caption.find(' endseq')
    if index > -1:
        caption = caption[:index]

    return caption


def generate_caption(frame_features, max_length):
    # load the image captioning model
    image_captioner = load_model("./models/image_captioner.h5")

    # load the tokenizer
    tokenizer = load(open('./models/tokenizer.pkl', 'rb'))

    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for _ in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = image_captioner.predict([frame_features, sequence], verbose=0)
        # convert probability to integer
        yhat = np.argmax(yhat)
        # map integer to word
        word = index_to_word(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break

    del tokenizer
    del image_captioner
    return in_text


def get_summary(frame_features, max_length):
    frame_captions = []

    for feature_vector in frame_features:
        caption = generate_caption(feature_vector, max_length)
        caption = cleanup_caption(caption)
        frame_captions.append(caption)

    return frame_captions


def get_frame_features(frames, target_size):
    cv2 = importlib.import_module("cv2")

    # load the feature extraction model
    model = load_model("./models/feature_extractor.h5")

    features = []

    for frame in frames:
        # resize the frame to (224, 244) which is the required shape for VGG16
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
        frame = img_to_array(frame)
        frame = frame.reshape((1, frame.shape[0], frame.shape[1], frame.shape[2]))
        frame = preprocess_input(frame)

        feature_vector = model.predict(frame, verbose=0)

        features.append(feature_vector)
    del model
    del cv2
    return features


def format_summary(summary):
    formatted_sentences = [sentence.strip().capitalize() + '.' if sentence else '' for sentence in summary]
    return '\n'.join(formatted_sentences)
