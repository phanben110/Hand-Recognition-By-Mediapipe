import cv2 
import numpy as np 
import numpy

class DrawFinger:
    def __init__(self, draw = True ): 
        self.draw = draw 
        
        self.box = [] 
    
    def boudingBox ( self, point  ):
        xList = [] 
        yList = [] 
        
        for i in range( len ( point ) ) : 
            xList.append ( point[i][0] ) 
            yList.append ( point[i][1] ) 
        xMin, xMax = min(xList), max(xList) 
        yMin, yMax = min(yList), max(yList) 
        self.box = [xMin, yMin , xMax, yMax]
    def drawAndResize( self, img , point  , size = 100 ): 
        self.boudingBox( point  )  
        img2 = numpy.ones((img.shape[0], img.shape[1], 3), numpy.uint8) * 0
        img3 = numpy.ones((size, size, 3), numpy.uint8) * 0

        if len ( self.box ) != 0 :

            y = self.box[1]
            x = self.box[0]
            w = self.box[2] - self.box[0]
            h = self.box[3] - self.box[1] 
            #note here can change because can't suits for this situation 
 

            #s =int ( w*h*0.0004) 
            #print ( w*h ) 
            #if s > 20 :
            #    s = 20
            #elif s < 3 :
            #    s = 1
            s = int ( w*h/2000 ) 
            #print ( s ) 
            if s > 30 : 
                s = 26 
            elif s < 1:
                s = 0
            #print (f"s: {s}" )

            self.box.append(s) 
            #cv2.rectangle( img2 , ( self.box[0] - s , self.box[1] - s ) , ( self.box[2] + s , self.box[3]+ s  ) , (0,255,0),2 )

            list_connections = [[0, 1, 2, 3, 4],
                                [0, 5, 6, 7, 8],
                                [5, 9, 10, 11, 12],
                                [9, 13, 14 , 15, 16],
                                [13, 17],
                                [0, 17, 18, 19, 20]]


            lines = [np.array([point[i] for i in line]) for line in list_connections]
            #print ( f"print line {lines}" )
             
            crop_img = img2[y-s:y + h + s , x-s:x + w + s ]
            cv2.polylines(img2, lines, False, (255,255, 255), s, cv2.LINE_AA)
            for a,b in point :
                cv2.circle(img2, (a, b), int ( s/2 ) , (255,255,255), -1)

            # resize 130 x 130
            k=1
            #print ( f"print crop {crop_img.shape}" ) 
            if w > h :
                k = w/(size - 4) 
            else :
                k = h/(size - 4) 
            
            w1 = int ( w/k )
            h1 = int ( h/k )
            dim = (w1 ,h1)
            #print (f" size {w}, {h} , dim {dim} ")
            try:
                crop_img  = cv2.cvtColor(crop_img , cv2.COLOR_BGR2GRAY)
                img3  = cv2.cvtColor(img3 , cv2.COLOR_BGR2GRAY)
                crop_img = cv2.resize(crop_img , dim)
                x_offset=int ( (size - crop_img.shape[0] )/2)
                y_offset=int ( (size - crop_img.shape[1] )/2)
                #print ( y_offset )
                img3[x_offset:crop_img.shape[0]+x_offset , y_offset:crop_img.shape[1]+y_offset] = crop_img
                return True , self.box , img3
            except : 
                return False, False, False 
        return False ,False , False 




