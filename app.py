import streamlit as st
from helpers import get_frames, get_frame_features, get_summary, format_summary

# define the maximum sequence length
max_length = 34

# target image size expected by feature extractor
target_size = (224, 224)

"""
# Frame2Word
An model that summarizes videos using image captioning. Built using Tensorflow (VGG16 + LSTMs) and
OpenCV (for reading video frames)
> **Disclaimer**
> Due to memory limits on Streamlit (1GB RAM per application), I was only able to summarize the video using 2 frames.
> Additional frames would cause the application to crash as it ran out of memory. I tried to add some memory optimization
> techniques with things such as dynamic imports and `del` on unused objects.
"""

st.divider()

"""
## Submit A Video
"""

video_types = [".mp4", ".mkv", "mov", "m4a", "3gp", "3g2", "mj2"]
uploaded_video = st.file_uploader(label="Choose a file(less than 2MB)", type=video_types)
if st.button("Summarize Video", type="secondary"):
    if uploaded_video is not None:
        video_size = file_size = len(uploaded_video.read())
        max_video_size = 2 * 1024 * 1024   # 2MB in bytes
        # validate the size of the video
        if video_size <= max_video_size:
            """
            ## Generated Summary
            """

            frames = get_frames(uploaded_video)
            frame_features = get_frame_features(frames, target_size)
            summary = get_summary(frame_features, max_length)
            summary = format_summary(summary)
            for line in summary.split('\n'):
                st.write(line)
        else:
            """
            ## Upload Error: Video Size
            You need to submit a video that is equal to or less than 2MB. Summary could not be
            generated. Try another video.
            """
    else:
        """
        ## Generated Summary
        You need to submit a video to get a summary.
        """
else:
    """
    ## Generated Summary
    You need to submit a video to get a summary.
    """

st.divider()
"""
## Reference Links
- [Github repository for project](https://github.com/chiroro-jr/frame2word)
- [Colab notebook used to train the model]("https://colab.research.google.com/drive/1yt6999Pba9VY1UebsKOxbiZFfjw_vnlc?usp=sharing")
"""
