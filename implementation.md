
## Plan

* ~~Basic structure~~
* Save loss history to compare between experiments
* Way to visualize loss over epochs
* Add proper calculation of true-positives-road, false-positives-non-road
* augment images with brightness for better detection of varios lit segments of the road
* Model optimization
    * ~~L2 regularization~~
    * ~~Skip layers~~
    * Advanced Faster-RCNN (from kaggle competition winners)
* Apply model to video

Log:
- adding stop gradient propagation increased training speed and same accuracy can be reached in shorter time (3 epochs, lr=1e-3)
 Accuracy=0.65; loss=0.76
 (10 epochs, lr=1e-2) acc=0.67, loss=~0.53
- Skip layers added. lr=1e-3
acc=0.84, l=0.44
- propagation stopped, epochs=5 (run=1537606761.08072)
acc=0.88, l=0.32
- additional 1x1 removed, epochs=5, lr-1e-3 (run=1537646319.892609)
acc=0.88, loss=0.35
- train longer: epochs=10, lr=1e-3 (run=)
acc=0.91, loss=0.20


