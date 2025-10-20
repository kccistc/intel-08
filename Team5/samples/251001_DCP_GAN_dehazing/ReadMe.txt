============================================================================================
A prior guided conditional generative adversarial network for single image dehazing

--------------------------------------------------------------------------------------------
Permission to use, copy, or modify this software and its documentation for educational
and research purposes only and without fee is hereby granted, provided that this copyright
notice and the original authors' names appear on all copies and supporting documentation.
This software shall not be used, redistributed, or adapted as the basis of a commercial
software or hardware product without first obtaining permission of the authors. The authors
make no representations about the suitability of this software for any purpose. It is
provided "as is" without express or implied warranty.
--------------------------------------------------------------------------------------------

This is a method for single image dehazing by using the prior-based dehazing results as guidance
described in the following paper:

Yan Zhao Su, Zhi Gao Cui, Chuan He, Ai Hua Li, Tao Wang, Kun Cheng,
"Prior guided conditional generative adversarial network for single image dehazing,"
Neurocomputing, 2021.https://doi.org/10.1016/j.neucom.2020.10.061

Please contact Yan Zhao Su (syzlhh@163.com) if you have any questions.

--------------------------------------------------------------------------------------------

DCP_GAN is proposed for single image dehazing by using a prior-based dehazing image as a guidance image for the conditional
generative adversarial network. It is a combination with the prior information and image translation method.

The code is edit and test on windows 10 system with the pytorch(version 1.8.0+cu101) framework and a 2070S GPU.
For the usage, you need prepare the datasets, then using the train.py to train the generative model, in the end you can use
test.py to test the model on your hazy images.



Please refer to the above paper for more details of the methods.

============================================================================================
