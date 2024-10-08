"""
Kneeliverse: A Universal Knee/ElbowDetection Library for Performance Curves

Estimating the knee/elbow point in performance curves is a challenging task.
However, most of the time these points represent ideal compromises between cost and performance.

This library implements several well-known knee detection algorithms:
1. Discrete Curvature 
2. DFDT
3. Kneedle
4. L-method
5. Menger curvature

Furthermore, the code in this library expands the ideas on these algorithms to 
detect multi-knee/elbow points in complex curves.
We implemented a recursive method that allows each of the previously mentioned methods
to detect multi-knee and elbow points.
Some methods natively support multi-knee detection, such as:
1. Kneedle
2. Fusion
3. Z-method

Finally, we also implemented additional methods that help with knee detection tasks.
As a preprocessing step, we develop a custom RDP algorithm that reduced a discrete 
set of points while keeping the reconstruction error to a minimum.
As a post-processing step we implemented several algorithms:
1. 1D dimensional clustering, is used to merge close knee points
2. Several filters out non-relevant knees
3. Knee ranking algorithms that used several criteria to assess the quality of a knee point
"""

import kneeliverse.clustering
import kneeliverse.convex_hull
import kneeliverse.curvature
import kneeliverse.dfdt
import kneeliverse.evaluation
import kneeliverse.knee_ranking
import kneeliverse.kneedle
import kneeliverse.linear_fit
import kneeliverse.lmethod
import kneeliverse.menger
import kneeliverse.metrics
import kneeliverse.multi_knee
import kneeliverse.postprocessing
import kneeliverse.rdp
import kneeliverse.zmethod