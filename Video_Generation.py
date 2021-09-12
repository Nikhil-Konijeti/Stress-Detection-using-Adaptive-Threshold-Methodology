#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from moviepy.editor import *
clip1 = VideoFileClip("Negative.mp4",audio=True)
clip2 = VideoFileClip("Negative.mp4",audio=True)
clip3 = VideoFileClip("Negative.mp4",audio=True)
clip4 = VideoFileClip("Negative.mp4",audio=True)
clip5 = VideoFileClip("Negative.mp4",audio=True)
clip6 = VideoFileClip("Negative.mp4",audio=True)
final_clip = concatenate_videoclips([clip1,clip2,clip3,clip4,clip5,clip6],method="compose")
final_clip.write_videofile("Final.mp4")