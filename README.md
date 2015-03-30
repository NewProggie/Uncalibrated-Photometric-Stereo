# Uncalibrated-Photometric-Stereo

Photometric-Stereo under unknown light source directions using svd

This is a short demo showing the creation of a 3D mesh using uncalibrated photometric-stereo made only with a laptop screen and a webcamera (Macbook Pro in this case). I'm using the integrated iSight camera together with the laptop screen to light a model from different angles. The outcome is a 3D mesh of the object with imagewidth x imageheight vertices which can be further used to feed a 3D printer for example.

[![Demo on YouTube](http://img.youtube.com/vi/qlq3n5r1Xy0/mqdefault.jpg)](http://www.youtube.com/watch?v=qlq3n5r1Xy0)

The idea for this setup is based on the following paper: Schindler, G. (2008). Photometric Stereo via Computer Screen Lighting for Real-time Surface Reconstruction. International Symposium on 3D Data Processing, Visualization and Transmission, 1–6.
The source code is written in C++ and I'm using OpenCV for image processing.

![Screenshot Export](https://raw.githubusercontent.com/NewProggie/Uncalibrated-Photometric-Stereo/master/images/export.png)
