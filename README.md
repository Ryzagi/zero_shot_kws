# zero_shot_kws
Keyword Spotting (KWS) is an essential component of voice-assist technologies, where the user
speaks a predefined keyword to wake up a system
before speaking a complete command or query to
the device. The project aims to create a zero-shot
keyword spotter and compare it with a non-zeroshot (static) keyword spotter trained to recognize
a specific set of keywords.
![assfasfsa drawio](https://user-images.githubusercontent.com/56489328/207005539-d2939f30-63fa-423a-abeb-e4cfcb306bbe.png)

# The result model is suggested to be achieved by the following steps:
1. Train the baseline model on labeled Google Commands
data.
2. Design and development of a zero-shot KWS model
that will detect any word (/phrase):
(a) Create and label a training and testing dataset
from ASR data. Come up with ways to create
negative samples.
(b) Create a basic pipeline on graphemes, which will
match the encoded sequence of graphemes and
the encoded piece of audio (waves).
(c) Create a pipeline on phonemes, i.e. add a phonemization stage before text encoder.
(d) Create a pipeline on ASR features, instead of text
encoder.
(e) Compare the results 2.1 - 2.3 on the dataset from
point 2.0.
(f) Present the results of comparing models in the
form of precision-recall curves and metrics (accuracy, f1, etc.)
3. Compare the best zero-shot model from point 2 with
the model from point 1
(a) Prepare Google Commands in the same format as
the datasets for point 2
(b) Compare the model from point 1 and point 2
(c) Present the results of the model comparison in
the form of precision-recall curves and metrics
(accuracy, f1, etc.)
