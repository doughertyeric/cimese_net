# cimese_net

Copyright Infringement (or CI) is a major issue for big media production companies. Recent estimates have placed the economic impact of copyright infringement on the big movie studios between 6.5 and 18 billion dollars per year.

With millions of video clips being uploaded to hosting sites like YouTube, an automated detection method is necessary. My goal here is to develop a means of efficiently and automatically checking each new video clip for copyright infringement. I decided to adopt a Siamese network structure to achieve this purpose; hence the name CI mese net.

Given a clip and a candidate film (i.e., one that we suspect the clip is infringeing upon), we will create a set of functions that will: a) align the clip with the most probable starting frame in the candidate film, b) using that segment of the film, extract a set of frames for comparison with the clip and perform the classification of each frame as a match or not a match, and c) output a final probability of infringement based on the a 0.5 threshold for each of the tested frames. Thus, the input to the full script should be the clip itself, the name of the candidate film (we can attempt to extend the process so that a candidate film is selected automatically), and the directory in which the trained top model is stored. The output should be a single probability of infringement value.
