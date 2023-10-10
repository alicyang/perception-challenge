# perception-challenge
![image](https://github.com/alicyang/perception-challenge/assets/121693495/005c5891-e89b-441e-868b-6d9ffca5d901)
Libraries used: cv2, numpy, matplotlib

Final Methodology: I divided the image into a left and right side, generating mask images by isolating the cones on each side based on     
                   their red color, used the masks to generate contours, and extracted two sets of coordinate points for each side from the 
                   centroids of the contours. Then, I used the coordinate sets to draw two lines with numPy. 
                   
Previous Methodology: Applying the findContours() function on the entire image. I quickly realized that this approach was too broad, as 
                      many contours could be extracted from the image, and I was uncertain about how I would approach isolating any single                        one of them. 
