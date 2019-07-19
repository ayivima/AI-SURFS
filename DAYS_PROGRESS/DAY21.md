
DAY 21:
=======

ACTIVITIES
---------------------------------------------------------------------------------------------------------------
### #SG_NOVICE-AI, BUILT A SIMPLE PYTORCH WRAPPER

1. Rolled out PYBYTES in #sg_novice-ai, which led us to demonstrate, and self implement tensor flattening. 
It is part of a series of explorations of seemingly scary AI concepts. 
Special shout out to @Olivia, @Eman, @Hung, @Aarthi Alagammai, and @Nirupama Singh for their amazing contributions to today's activities.

2. Still reading this book, "The Mathematics of Language": http://linguistics.ucla.edu/people/Kracht/html/formal.alt.pdf.

3. Built a minimalistic wrapper around Pytorch for creating networks. Network creation gets as easy as: 
`network(sequence of node counts for each layer, hidden nodes activation function, output function)`
```
>>> model3 = network((784,256,128,64,32,16,10,5,3), ReLU(), logsoftmax(0))
>>> model3
Sequential(
  (0): Linear(in_features=784, out_features=256, bias=True)
  (1): ReLU()
  (2): Linear(in_features=256, out_features=128, bias=True)
  (3): ReLU()
  (4): Linear(in_features=128, out_features=64, bias=True)
  (5): ReLU()
  (6): Linear(in_features=64, out_features=32, bias=True)
  (7): ReLU()
  (8): Linear(in_features=32, out_features=16, bias=True)
  (9): ReLU()
  (10): Linear(in_features=16, out_features=10, bias=True)
  (11): ReLU()
  (12): Linear(in_features=10, out_features=5, bias=True)
  (13): ReLU()
  (14): Linear(in_features=5, out_features=3, bias=True)
  (15): ReLU()
  (16): LogSoftmax()
)
```
Snapshots of `quicknets` in action attached below.

4. @Stark's passion for #sg_novice-ai is amazing...We keep planning on stirring up empowering activity in #sg_novice-ai. 
We believe in building from the ground right up! @Stark says we will become "Pytorch beasts"...Haha!

5. ...And, I am still at implementing concepts at a bare-bones level: Matrix tools and its "spin-offs"


REPO FOR 60DAYSOFUDACITY:
-------------------------
https://github.com/ayivima/AI-SURFS/

PROGRESS:
---------
https://github.com/ayivima/AI-SURFS/blob/master/DAYS_PROGRESS


ENCOURAGEMENTS
--------------
Cheers to  @Eman, @Aarthi Alagammai, @Olivia, @Sharim, @Hung, @Varez.W, @THIYAGARAJAN R, @LauraT , @Anna Scott, @Nirupama Singh, @Frida, @Lisa Crossman, @Stark, @Samuela Anastasi, @nabhanpv, @Nana Aba T, @geekykant, @Shaam, @EPR, @Anshu Trivedi, @George Christopoulos, @Vivank Sharma, @Heather A, @Joyce Obi, @Aditya kumar, @vivek, @Florence Njeri, @Jess, @J. Luis Samper, @gfred, @Erika Yoon.
