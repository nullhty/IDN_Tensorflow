# IDN_Tensorflow
Tensorflow implementation of "Fast and Accurate Single Image Super-Resolution via Information Distillation Network

I only training the X2 model and the PSNR on Set5 (37.70+) is lower than the result in paper.

## Update
### 2019-05-27
The old code use the low.shape to calculate the psnr.

Add a new line of code in "test.py"(78 line) to use the original.shape.
