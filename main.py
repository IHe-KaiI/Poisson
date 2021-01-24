from PIL import Image
import numpy as np 

def calc(x,y):
    global w1,h1
    x-=1
    y-=1
    return x*h1+y

dx=[1,-1,0,0]
dy=[0,0,-1,1]

p_img1=Image.open('1.jpg')
p_img2=Image.open('2.jpg')
img1=np.array(p_img1).transpose(1,0,2)
img2=np.array(p_img2).transpose(1,0,2)
print(img1.shape)

img1_pred=np.zeros(img1.shape)

w1=img1.shape[0]-2
h1=img1.shape[1]-2
#x,y=input().split()
x=10
y=10

for i in range(1,img1.shape[0]-1):
    for j in range(1,img1.shape[1]-1):
        for k in range(3):
            img1_pred[i][j][k]=img1[i][j][k]*-4
            for fx in range(4):
                img1_pred[i][j][k]+=img1[i+dx[fx]][j+dy[fx]][k]

for k in range(3):
    pixel_num=w1*h1
    B=np.zeros([pixel_num])
    A=np.zeros([pixel_num,pixel_num])
    for i in range(1,img1.shape[0]-1):
        for j in range(1,img1.shape[1]-1):
            if(i==1):
                B[calc(i,j)]-=img2[x][y+j][k]
            if(j==1):
                B[calc(i,j)]-=img2[x+i][j][k]
            if(i==img1.shape[0]-2):
                B[calc(i,j)]-=img2[x+i+1][y+j][k]
            if(j==img1.shape[1]-2):
                B[calc(i,j)]-=img2[x+i][y+j+1][k]

          
            B[calc(i,j)]+=img1_pred[i][j][k]
            print(i,j,k,img1_pred[i][j][k],B[calc(i,j)])
            A[calc(i,j)][calc(i,j)]=-4
            for fx in range(4):
                nx=i+dx[fx]
                ny=j+dy[fx]
                if(nx>w1 or ny>h1 or nx<1 or ny<1):
                    continue
                A[calc(nx,ny)][calc(i,j)]=1
                A[calc(i,j)][calc(nx,ny)]=1
    inv_A=np.linalg.inv(A)
    print(inv_A)
    X=inv_A.dot(B)
    for i in range(1,img1.shape[0]-1):
        for j in range(1,img1.shape[1]-1):
            img2[x+i][y+j][k]=min(max(X[calc(i,j)],0),255)
img2=img2.transpose(1,0,2)
res=Image.fromarray(img2)
res.show()