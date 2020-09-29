Using ![MIT's Indoor CVPR dataset](http://web.mit.edu/torralba/www/indoor.html), a deep learning pipeline is created which is able to automatically correct image orientation.
A labeled dataset was created where images were randomly rotated by {0, 90, 180, 270} degrees

Transfer learning via feature extraction was then applied and the VGG16
network architecture was used to extract features from the final max-pooling layer in the network. These features
were fed into a Logistic Regression classifier, enabling to correctly predict the orientation of
an image with 91% accuracy. 

![image_orientation_terminal](https://user-images.githubusercontent.com/56476887/94591197-5d1d6900-02a5-11eb-985c-7e7e288959a6.png)


Combining both the VGG16 network and our trained Logistic
Regression model enabled us to construct a pipeline that can automatically detect and correct image
orientation.

## Output
![hope2](https://user-images.githubusercontent.com/56476887/94584471-4aeafd00-029c-11eb-8397-199a44cfb3ce.gif)
![hope](https://user-images.githubusercontent.com/56476887/94584466-4888a300-029c-11eb-9ad8-69746321527b.gif)

this result would have been impossible if our
CNN was truly rotation invariant. CNNs are robust image classification tools and can correctly
classify images under a variety of orientations. However, the individual filters inside the CNN are
not rotation invariant.
