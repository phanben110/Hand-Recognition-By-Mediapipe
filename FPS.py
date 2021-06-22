# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 15:29:32 2017

@author: geaxx
"""
import time
import cv2

def now():
    return time.perf_counter()
    
class FPS: # To measure the number of frame per second
    def __init__(self, mean_nb_frames=10):
        self.nbf=0
        self.fps=0
        self.start=0
        self.mean_nb_frames = mean_nb_frames
        
    def update(self):
        if self.nbf%self.mean_nb_frames==0:
            if self.start != 0:
                self.stop=now()
                self.fps=self.mean_nb_frames/(self.stop-self.start)
                self.start=self.stop
            else :
                self.start=now()    
        self.nbf+=1
    
    def get(self):
        return self.fps

    def display(self, win, orig=(10,50), font=cv2.FONT_HERSHEY_PLAIN, size=3, color=(255,0,255), thickness=3):
        #cv2.putText( frame , f"Status: {label} {conf}" , (200,50) , cv2.FONT_HERSHEY_PLAIN,3, (255, 0, 255  ) ,thickness=3)
        cv2.putText(win,f"FPS={self.get():.2f}",orig,font,size,color,thickness)

