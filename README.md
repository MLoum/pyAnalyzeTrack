# pyAnalyzeTrack Fork : Gaétan Delhaye

Last change K-means 2D, GMM 2D, K-means 3D GMM 3D with absolute value of red and green (meaning the average value of each canal for each frame tracked of a particle). Added these values to the filters ans added a first script to treat multiple files automatically ( not very well made for the moment but it works).

In 3D the label tend to randomly switch so sometimes the mnoomeres is labelized as dimeres well have to find a fix ( could just try a test to see wiwh label has the biggest r_cov mean but thats not very elegant in the code it create a layer of test after the computation)
