# Handwriten-digit-recognition
# Handwritten digit recognition using various deep learning models
Many uses for digit recognition can be found in
processing bank checks, identifying licence plates,
sorting mail, and more . Due to the fact that
handwritten digit recognition is not optical
character recognition, there are numerous
difficulties arising from the various writing styles
used by various persons. For the goal of
handwritten digit recognition, this research offers a
thorough comparison of various machine learning
and deep learning techniques.
# Used models
MLP Classifier,LeNet, Le Net5, ResNet,and Convolutional Neural
Network have all been utilized for this. A machine
is more effective at pattern or text recognition
thanks to machine learning technologies. Because
of the continuous accumulation and incremental
development of handwritten digit sample
collections for identification precision, the temporal
complexity of current algorithms or models is
actually very high.
Any model must be accurate in order to produce
reliable results; this is crucial. Low precision
models are unsuitable for real-world applications.To
an ex-High precision is essential in an automated
bank check processing system that can read the
amount and date on the check. It is undesirable if
the system mistakenly interprets a digit since it
could cause serious harm.
Because of this, these real-world applications need
an algorithm with great precision. So that the most
accurate
the method with the lowest likelihood of errors can
be used in various handwritten digit recognition
applications, we are offering a comparison of
several algorithms based on their accuracy.
Since the 1980s, handwriting recognition
software has been available. The task of
handwritten digit recognition, using a classifier, has
extraordinary significance and use, including online
digit recognition on PC tablets, identifying zip
codes on mail, processing bank check amounts, and
processing numeric sections in structures filled out
by hand (for example, tax forms). While attempting
to address this problem, numerous difficulties arise.
There are variations in the size, thickness,
orientation, and placement of the handwritten digits
in relation to the margins. The primary goal was to
implement a method of pattern characterisation to
recognise the handwritten numbers offered in the
MINIST data collection of photographs of
handwritten digits (0–9)
# Methodology 
first we have run our 5 model at 10 epoch 
then at 50 and after that at 100. with a
batch size of 128.we have used relu
Activation function and softMax to.
for evaluation training loss and
traninng accurcy is considerd. 
#  mlp consist following architecture
 Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
# CNN  
   Dense(64, activation='relu'),
    Dense(10, activation='softmax')
# LeNet 
 Dense(120, activation='relu'),
    Dense(84, activation='relu'),
    Dense(10, activation='softmax')
# LeNet 5 
Dense(120, activation='relu'),
    Dense(84, activation='relu'),
    Dense(10, activation='softmax')
# ResNet 
   Dense(units=256, activation='relu')
   Dense(units=128, activation='relu')
