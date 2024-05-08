# The Effect of Super Resolution Method on Classification Performance

This repository is dedicated to exploring the impact of various super-resolution (SR) methods on the performance of image classification models. By enhancing low-resolution images using different SR techniques, we aim to investigate how the quality of image resolution affects the accuracy of classifying images into predefined categories.

## Introduction

Image super-resolution (ISR) is a technique aimed at converting low-resolution images into high-resolution counterparts, ideally preserving content and details. Image classification, on the other hand, involves assigning images to predefined categories based on their content. This project seeks to bridge these two areas by examining the effect of super-resolution on classification accuracy. We utilize a subset of the ImageNet dataset, applying various SR methods to low-resolution images, and then assess classification performance using these enhanced images.

## Dataset

The ImageNet dataset, containing over 14 million annotated images across more than 21 thousand classes, serves as the foundation for our experiments. We focus on the validation set of the ImageNet dataset to assess the classification performance, using top-1 accuracy as our primary metric.

## Data Preparation

We employ the Laplacian function from the OpenCV library to measure the sharpness of images, identifying those that could benefit from super-resolution techniques. This step helps in selecting suitable candidates for the SR process to improve image quality.

## Methodology

Our methodology encompasses several key SR techniques:

- **Real-ESRGAN**: Utilizes a second-order degradation process to model practical degradations, including blur, resize, noise, and JPEG compression.
- **BSRGAN**: Introduces a practical degradation model that randomly shuffles degradations to synthesize low-resolution images, aiming to improve the generalization ability of SR models.
- **Swin2SR**: A variant of the Swin Transformer V2 tailored for image super-resolution tasks, focusing on multi-scale feature extraction and efficient long-range dependency capture.

## Classification

We assess the impact of SR on classification performance using a systematic approach. Low-resolution images enhanced by SR techniques are classified using the VGG19 algorithm to evaluate the effectiveness of SR in improving classification accuracy.

## Contact

For any inquiries or further discussion, please contact the authors:

- Ömer Faruk Aydın (aydinome21@itu.edu.tr)
- Hasan Taha Bağcı (bagcih21@itu.edu.tr)

## Acknowledgments

This project is conducted as part of the YZV416E Computer Vision Project at Istanbul Technical University, under the guidance of our professors and mentors in the Artificial Intelligence and Data Engineering department.

