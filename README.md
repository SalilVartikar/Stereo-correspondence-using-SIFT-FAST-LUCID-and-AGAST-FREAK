# Stereo correspondence using SIFT, FAST+LUCID and AGAST+FREAK

##Application: Stereo correspondence of images##

##Feature detectors and descriptors used:##
1. SIFT
2. FAST + LUCID
3. AGAST + FREAK

##Instructions:##
1. Add library and include dependencies for the project as below:
   C/C++ > Additional Include Libraries: Add "C:\OpenCV3.3\OpenCV\build\include".

2. Linker > General > Additional Library Directories: Add "C:\OpenCV3.3\OpenCV\build\x64\vc14\lib".

3. Linker > Input > Additional Dependencies: Add opencv_world330d.lib.

4. Confirm the configuration mode is Debug.

5. Copy the images "left.png", "right.png" and "truth.png" to the main project folder (where the source code is).

6. Open the source code in Visual Studio.

7. Run the code. You can find the output images in the main folder. Also, necessary details have been printed in the    console.

##Note: The project has been compiled and run on Windows 10 machine running Visual Studio 15 and OpenCV 3.3.##
