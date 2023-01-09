import rplidar
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint, Point
from scipy.spatial import Voronoi, voronoi_plot_2d

# Initialize the RPLIDAR
lidar = rplidar.RPLidar('/dev/ttyUSB0')

# Retrieve the distance measurements from the RPLIDAR
measurements = []
for i, scan in enumerate(lidar.iter_scans()):
    measurements.extend([(float(r[0]), float(r[1])) for r in scan])

# Convert the distance measurements to a 2D point cloud
points = []
for r, theta in measurements:
  x = r * np.cos(theta)
  y = r * np.sin(theta)
  points.append((x, y))

# Use DBSCAN to cluster the points into groups
clustering = DBSCAN(eps=0.2, min_samples=10).fit(points)
cluster_labels = clustering.labels_
num_clusters = len(set(cluster_labels))
clusters = []
for i in range(num_clusters):
    clusters.append([points[j] for j in range(len(points)) if cluster_labels[j] == i])

# Select the clusters that are likely to be pots
pot_clusters = []
for cluster in clusters:
  # Convert the cluster to a MultiPoint object
  cluster_points = MultiPoint(cluster)
  
  # Calculate the convex hull of the cluster
  cluster_hull = cluster_points.convex_hull
  
  # Check if the area of the convex hull is small enough to be a pot
  if cluster_hull.area < 1:
    pot_clusters.append(cluster)

# Initialize the minimum distance to a large value
minDistance = float("inf")

# For each pot cluster:
for pot_cluster in pot_clusters:
  # Calculate the distance between the pot cluster and the road
  distance = calculateDistanceToRoad(pot_cluster, measurements)
  
  # Update the minimum distance if necessary
  if distance < minDistance:
    minDistance = distance

# Output the minimum distance
print("The minimum distance between the road and the pots is:", minDistance)

# Generate a Voronoi diagram of the environment
vor = Voronoi(points)

# Select the Voronoi edges that are within a certain distance of a pot
pot_edges = []
for pot_cluster in pot_clusters:
  for v in vor.vertices:
    for p in pot_cluster:
      if np.linalg.norm(v-p) < 0.5:
        pot_edges.append((v,p))

# Plot the environment
fig = voronoi_
