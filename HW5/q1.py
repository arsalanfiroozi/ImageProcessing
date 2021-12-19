import cv2
import numpy as np
from scipy.signal import convolve2d
from scipy.sparse.linalg import spsolve

imgs = cv2.imread('1.source.jpg');
imgt = cv2.imread('2.target.jpg');
mask = cv2.imread('Mask_i2.jpg');
mask = (mask[:,:,0]>100)*255;

ps = [26,26];
pt = [50,50];
w = 215;
h = 360;
warpm = [[1,0,pt[1]-ps[1]],[0,1,pt[0]-ps[0]]];

#maski = np.zeros((imgt.shape[0],imgt.shape[1]));
#maski[pt[0]:pt[0]+w,pt[1]:pt[1]+h] = 255;
maski = cv2.warpAffine(np.uint8(mask),np.float64(warpm),(imgt.shape[1],imgt.shape[0]));
maskiL = maski.reshape((maski.shape[0]*maski.shape[1],));

n = np.int32(np.sum(maski/255));
val = np.int32([i for i in range(1,n+1)]);
Map = np.zeros((maski.shape[1]*maski.shape[0],1),dtype=np.int32);
Map[maskiL == 255] = val.reshape((val.shape[0],1));
Map = Map.reshape((maski.shape[0],maski.shape[1])) - 1;


gradKernel = np.float32([[0,1,0],[1,-4,1],[0,1,0]]);

res = np.zeros(imgt.shape);
for k in range(3):
	imgsc = imgs[:,:,k];
	imgtc = imgt[:,:,k];
#	grad = cv2.filter2D(imgsc,-1,gradKernel);
	grad = convolve2d(imgsc,gradKernel,mode='same');
	grad = cv2.warpAffine(grad,np.float64(warpm),(imgtc.shape[1],imgtc.shape[0]));
	gradSource = convolve2d(imgtc,gradKernel,mode='same');
#	cv2.imwrite('g'+np.str(k)+'.jpg',grad);
	
	A = np.zeros((n,n),dtype=np.int8);
	C = np.zeros((n,1),dtype=np.int32);
	for i in range(imgtc.shape[0]):
		for j in range(imgtc.shape[1]):
			if(Map[i,j]!=-1):
				
				t = Map[i,j];
				A[t,t] = -4;
				
				if(Map[i+1,j] == -1):
					C[t] = C[t] - imgtc[i+1,j];
				else:
					A[t,Map[i+1,j]] = 1;
					
				if(Map[i-1,j] == -1):
					C[t] = C[t] - imgtc[i-1,j];
				else:
					A[t,Map[i-1,j]] = 1;
					
				if(Map[i,j+1] == -1):
					C[t] = C[t] - imgtc[i,j+1];
				else:
					A[t,Map[i,j+1]] = 1;
					
				if(Map[i,j-1] == -1):
					C[t] = C[t] - imgtc[i,j-1];
				else:
					A[t,Map[i,j-1]] = 1;
					
				C[t] = C[t] + grad[i,j];
#				if(np.max(np.abs([grad[i,j], gradSource[i,j]])) == np.abs(grad[i,j])):
#					C[t] = C[t] + grad[i,j];
#				else:
#					C[t] = C[t] + gradSource[i,j];
				
	sol = spsolve(A,C);
	sol = sol - np.min(sol);
	sol = sol / np.max(sol) * 255;
	sol = np.uint8(sol);
	
	c = imgtc.copy();
	c = c.reshape((imgtc.shape[0]*imgtc.shape[1],1));
	c[maskiL == 255] = sol.reshape((sol.shape[0],1));
	res[:,:,k] = c.reshape((imgtc.shape[0],imgtc.shape[1]));
cv2.imwrite('res1.jpg',res);

#imgs = cv2.imread('1.source.jpg');
#imgt = cv2.imread('2.target.jpg');
#imgs = cv2.warpAffine(np.uint8(imgs),np.float64(warpm),(imgt.shape[1],imgt.shape[0]));
#imgt[maski == 255,:] = imgs[maski == 255,:];
#cv2.imwrite('nochnages.jpg',imgt);