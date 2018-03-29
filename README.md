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

