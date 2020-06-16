# Proxy Anchor Loss

## Overview

This repository contains a Keras implementation of the loss function introduced in [Proxy Anchor Loss for Deep Metric Learning](https://arxiv.org/abs/2003.13911).
Alternatively, you may find the official PyTorch implementation [here](https://github.com/tjddus9597/Proxy-Anchor-CVPR2020/blob/master/code/losses.py).

## Example of usage

```python
# Define the input and output tensors
input_tensor = Input(shape=(embedding_size))
output_tensor = ProxyAnchor(units=class_num)(input_tensor)

# Define the model and compile it
model = Model(inputs=input_tensor, outputs=output_tensor)
model.compile(loss=proxy_anchor_loss, optimizer="adam")
```

## Notes

* Tested on TensorFlow 1.15.3 and 2.2.0.
* Create an issue should any questions arise.