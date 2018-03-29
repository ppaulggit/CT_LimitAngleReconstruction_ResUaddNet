# CT_LimitAngleReconstruction_ResUaddNet

This is a ResUaddNet basic using UNet but
the concat operation change to add operatin,
and combing the residual block.

We set the angle interval is 10 degree, so 
the total projection number is 18(because 
the total projection angle is 180 degree).

The residual block is two 3X3 convolution
like the resnet_v2's residual block:

     bn->relu->conv->bn->relu->conv

Here is the loss curve.The begin loss is 
0.6779, the final loss is 0.0005.
![image](https://github.com/PaulGitt/CT_LimitAngleReconstruction_ResUaddNet/blob/master/loss.jpg)

The final reconstruction result is below.
There are three row, each column represent
each model, the first one in one row
is origin model, second one is LimitAngleRecon,
the third one is the reuslt use ResUaddNet model
to reconstruct.
![image](https://github.com/PaulGitt/CT_LimitAngleReconstruction_ResUaddNet/blob/master/origin_limitangle_recon1.jpg)
![image](https://github.com/PaulGitt/CT_LimitAngleReconstruction_ResUaddNet/blob/master/origin_limitangle_recon2.jpg)
![image](https://github.com/PaulGitt/CT_LimitAngleReconstruction_ResUaddNet/blob/master/origin_limitangle_recon3.jpg)
