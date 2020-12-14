# The Effect of Image Resolution on Neural Networks

Please view paper_cs231n.pdf for a more detailed write up of this project.

## 

Neural networks are becoming deeper and more costly to evaluate. One potential way to reduce the computational cost of neural networks is to use images with lower resolutions. The impact of using lower resolution training and testing images on the accuracy of image classification has not previously been studied. We train image classification CNN to accomplish standard fine tuning tasks, using images from several data sets (to represent tasks of various difficulties) at several resolutions, evaluating the CNN using their testing accuracies. We find that the more difficult a task is, the higher the resolution the images the model uses should be: ranging from 64x64 for easier tasks to 224x224 for difficult tasks. 