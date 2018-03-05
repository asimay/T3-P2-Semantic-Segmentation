# Semantic Segmentation
### Goals
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

---

[image1]: ./output/fianl1.gif
[image11]: ./output/final2.gif
[image2]: ./output/trajectory.PNG
[image3]: ./output/1.PNG
[image4]: ./output/2.PNG
[image5]: ./output/best.PNG
[image6]: ./output/trap.PNG

### Final result

Final testing on images, take 4 pics as example:

![alt text][image1]
![alt text][image2]

![alt text][image3]
![alt text][image4]



### Procedure:

#### 1. Load VGG Model.

The model is a fully convolutional version model. I first used `tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)` load it from vgg path, and then use `graph = tf.get_default_graph()` to get the tensor from model,
we can get `input_tensor, keep_prob, layer3_out, layer4_out, layer7_out` hyper parameters from model. we will use these parameters to train the network.

The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.

#### 2. Data Augumentation

I wrote a helper function `helper.gen_augumentation_data(os.path.join(data_dir, 'data_road/training'), image_shape)` in `run()` to generate more data for better performance. in this method, because the pic is about road and environment, I only flip the original images and ground truth images, and output them into training data images dir. this will generate the data 2 times than original data, about `289*2=578` images to train. you can find the implementation in `helper.py` file.


#### 3. FCN-8 network to train and optimize.

The FCN-8 architecture is as below:

![alt text][image5]

I used `layers()` to create the layers for a fully convolutional network, and build skip-connction layers.

I used 1x1 convolution to replace with fully connection layer of vgg_layer7, and then use `tf.layers.conv2d_transpose` to upsample the data, and then, combined with convolution of vgg_layer4_out, this generate layer4 skip-connction layer, and then, the same as above, I use layer4_out to upsample the data, and then combined with convolution of vgg_layer3_out, this generate layer3 skip-connction layer, after these step, we finally upsample the layer3_out to output logits.

About optimizer, I first used `tf.nn.softmax_cross_entropy_with_logits()`, this is used to output logits with softmax, and calculate the cross entropy, then used `tf.train.AdamOptimizer()` to optimize the train. 

About learning rate, I set to `learning_rate=0.001`, and `keep_prob=0.5`

I also used `kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3)` to add l2-regularization, and with a smaller value : `STD_DEV = 0.001`.

#### 4. Train.
In training function `train_nn()`, set `feed_list` as feed_dict's input, and then run `train_op` tensor to train.

I also wrote a small piece of code to show training progress. This will give us a direct feeling of training.

![alt text][image6]

set training hyper parameters:
```
	epochs = 30
	batch_size = 32
```

#### 5. Training Loss

And the training loss shows as below:

![alt text][image7]

After training, save the model with `tf.train.Saver().save`, this produce about 2G model files.

#### 6. Testing

With the trained model, test the testing images, and output all results to  `runs` dir.

![alt text][image1]
![alt text][image3]

#### 7. Testing on videos:

I also wrote some function like `process_video()` to run prediction directly on video. This is same step with testing step.

## Tips

The final result show good prediction of road in many images, but there is still some result are not good, especiall with shadow, light, human foot and so on.

## Further to promote

Promote more accuracy about the result.