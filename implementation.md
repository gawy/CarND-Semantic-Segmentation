
## Plan

* ~~Basic structure~~
* Way to visualize loss over epochs
* Save loss history to compare between experiments
* Model optimization
    * ~~L2 regularization~~
    * ~~Skip layers~~
    * Advanced Faster-RCNN (from kaggle competition winners)


Log:
- adding stop gradient propagation increased training speed and same accuracy can be reached in shorter time (3 epochs, lr=1e-3)
 Accuracy=0.65; loss=0.76
 (10 epochs, lr=1e-2) acc=0.67, loss=~0.53
- Skip layers added. lr=1e-3
acc=0.84, l=0.44
- propagation stopped, epochs=5
acc=0.88, l=0.32


