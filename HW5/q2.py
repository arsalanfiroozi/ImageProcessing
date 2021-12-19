import cv2
import numpy as np
from scipy.signal import convolve2d

def gauss2d(x,sigma):
	return np.exp(-(x[0]**2+x[1]**2)/(2*sigma**2)) / np.sqrt(2) / sigma;

def kernel_gauss(n,sigma):
	k = np.zeros([n,n]);
	for i in range(n):
		for j in range(n):
			k[i,j] = gauss2d([i - (n+1)/2,j- (n+1)/2],sigma);
	return k/np.sum(k);

def lappyr(img1,img2,mask,kernel,n):
	if(n == depth):
		nFilter = convolve2d(mask,kernel,mode='same')/255;
		nFilter = nFilter/np.max(nFilter);
		res = nFilter*img1+(1-nFilter)*img2;
#		cv2.imwrite('Filter-Level'+np.str(n)+'.jpg',nFilter*255);
		return res;
	res = lappyr(cv2.pyrDown(img1),cv2.pyrDown(img2),cv2.pyrDown(mask),kernel,n+1);
	res = cv2.pyrUp(res,dstsize=(img1.shape[1],img1.shape[0]));
	kb = np.float64([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]])/256;
	img1b = convolve2d(img1,kb,mode='same',boundary='symm');
	img1L = img1 - img1b;
	img2b = convolve2d(img2,kb,mode='same',boundary='symm');
	img2L = img2 - img2b;
	nFilter = convolve2d(mask,kernel,mode='same')/255;
	nFilter = nFilter/np.max(nFilter);
	res = nFilter*img1L+(1-nFilter)*img2L + res;
#	print(type(nFilter))
#	cv2.imwrite('Filter-Level'+np.str(n)+'.jpg',nFilter*255);
	return res;

depth = 4;

imgs = cv2.imread('1.source.jpg');
imgt = cv2.imread('2.target.jpg');
maski = cv2.imread('Mask_i2.jpg');
maski = maski[:,:,0];
maski = (maski > 100)*255;

ps = [26,26];
pt = [50,50];
w = 215;
h = 360;

warpm = [[1,0,pt[1]-ps[1]],[0,1,pt[0]-ps[0]]];

#Mask = np.zeros((imgt.shape[0],imgt.shape[1]));
#Mask[pt[0]:pt[0]+w,pt[1]:pt[1]+h] = 255;
Mask = cv2.warpAffine(np.uint8(maski),np.float64(warpm),(imgt.shape[1],imgt.shape[0]));

Kernel = kernel_gauss(41,3);
#Kernel = np.float32([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]])/254;

#cv2.imwrite('Mask.jpg',Mask);
#cv2.imwrite('Kernel.jpg',Kernel * 255 / np.max(Kernel));
#cv2.imwrite('PyrTest.jpg',cv2.pyrUp(imgs[:,:,0], dstsize=(500,500)));


res = imgt.copy();
for i in range(3):
	imgsco = imgs[:,:,i];
	imgtc = imgt[:,:,i];
	imgsc = cv2.warpAffine(imgsco,np.float64(warpm),(imgtc.shape[1],imgtc.shape[0]));
	res[:,:,i] = lappyr(imgsc,imgtc,Mask,Kernel,0);

cv2.imwrite('res2.jpg',res);