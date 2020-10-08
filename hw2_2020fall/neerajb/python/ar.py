import numpy as np
import cv2
#Import necessary functions
from matchPics import matchPics
from planarH import compositeH
from planarH import computeH_ransac
from loadVid import loadVid
from opts import get_opts

opts = get_opts()

#Write script for Q3.1

# import videos
panda_video = loadVid('../data/ar_source.mov')
book_video = loadVid('../data/book.mov')
cv_cover = cv2.imread('../data/cv_cover.jpg')

# create VideoWriter Object
writer = cv2.VideoWriter('../result/ar.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20, (book_video.shape[2], book_video.shape[1]))

for i in range(0, book_video.shape[0]):
    # continue running panda video if it ends
    ii = i%panda_video.shape[0]
    
    # print(i)
    matches, locs1, locs2 = matchPics(book_video[i], cv_cover, opts)
    new_locs1 = []
    new_locs2 = []
    for match in matches:
        new_locs1.append(locs1[match[0]])
        new_locs2.append(locs2[match[1]])

    # flip x and y columns
    new_locs1 = np.asarray(new_locs1)
    new_locs1[:,[0, 1]] = new_locs1[:,[1, 0]]
    new_locs2 = np.asarray(new_locs2)
    new_locs2[:,[0, 1]] = new_locs2[:,[1, 0]]

    # compute best H
    bestH2to1, inliers = computeH_ransac(new_locs1, new_locs2, opts)

    # trim off black edges 
    L = int((book_video[i].shape[0] * panda_video[ii].shape[1])/(book_video[i].shape[1]))
    D = int(abs(panda_video[ii].shape[1] - L)/2)

    L2 = int((book_video[i].shape[1] * panda_video[ii].shape[0])/(book_video[i].shape[0]))
    D2 = int(abs(panda_video[ii].shape[1] - L2)/2)

    trimmed_panda = panda_video[ii][D:D+cv_cover.shape[1],D2:D2+cv_cover.shape[0]] 
    trimmed_panda = cv2.resize(trimmed_panda, (cv_cover.shape[1], cv_cover.shape[0]))

    video_frame = compositeH(bestH2to1, trimmed_panda, book_video[i])
    
    # write to avi files
    
    writer.write(video_frame)

writer.release()   

