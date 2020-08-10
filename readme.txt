it’s a common misconception that the individual filters learned by a CNN are invariant to rotation – the filters themselves are not. 
Instead, the CNN is able to learn a set of filters that activate when they see a particular object under a given
rotation. Therefore, the filters themselves are not rotation invariant, but the CNN is able to obtain
some level of rotation invariance due to the number of filters it has learned, with ideally some of
these filters activating under various rotations. As we’ll see in this chapter, CNNs are not fully
rotation invariant, otherwise, it would be impossible for us to determine the orientation of an image

1st using the Indoor Scene Recognition dataset, we create a dataset that consists of images of 4 groups, rotated by 0, 90, 180 and 270 degrees.
We then extract features using VGGNet and then train a logistic regressor and we obtain the following result:
              precision    recall  f1-score   support

           0       0.94      0.92      0.93       633
         180       0.92      0.92      0.92       637
         270       0.88      0.91      0.90       608
          90       0.90      0.89      0.89       617

    accuracy                           0.91      2495
   macro avg       0.91      0.91      0.91      2495
weighted avg       0.91      0.91      0.91      2495

Using this, we can correct image orientations.Algorithms and deep learning pipelines such as these can be used to process large datasets,
perhaps of scanned images, where the orientation of the input images is unknown. Applying such a
pipeline would dramatically reduce the amount of time taken for a human to manually correct the
orientation prior to storing the image in an electronic database.

Finally, it’s once again worth mentioning that this result would have been impossible if our
CNN was truly rotation invariant. CNNs are robust image classification tools and can correctly
classify images under a variety of orientations; however, the individual filters inside the CNN are
not rotation invariant.

There is certainly a degree of generalization here as well; however, if these filters were truly rotation
invariant the features we extracted from our input images would be near identical to each other.
If the features were near identical, then the Logistic Regression classifier would not be able to
discriminate between our image orientations. This is an important lesson to keep in mind when
developing your own deep learning applications – if you expect input objects to exist under many
orientations, make sure your training data reflects this requirement.
