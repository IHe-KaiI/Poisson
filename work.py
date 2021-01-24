import cv2
import numpy as np
import numpy.linalg as lg

src_path='./4.jpg'
dst_path='./3.jpg'
res_path='./test.jpg'

pos_x=50
pos_y=50

dx=np.array([0,0,1,-1])
dy=np.array([1,-1,0,0])

def Get_id(x, y):
    return x*H+y

def Get_MatrixA():
    num=W*H
    A=np.zeros([num,num])
    for x in range(W):
        for y in range(H):
            id=Get_id(x,y)
            A[id][id]=-4
            for i in range(4):
                x1=x+dx[i]
                y1=y+dy[i]
                if (x1>=0 and x1<W and y1>=0 and y1<H):
                    id1=Get_id(x1,y1)
                    A[id][id1]=1
    return A

def Get_MatrixB(src, dst):
    num=W*H
    Kernel=np.array([[0,1,0],[1,-4,1],[0,1,0]])
    conv=cv2.filter2D(src,-1,Kernel)
    Br=np.zeros(num)
    Bg=np.zeros(num)
    Bb=np.zeros(num)

    for x in range(W):
        for y in range(H):
            id=Get_id(x,y)
            for i in range(4):
                x1=x+dx[i]
                y1=y+dy[i]
            #Br[id]=conv[x][y][0]
            #Bg[id]=conv[x][y][1]
            #Bb[id]=conv[x][y][2]
            for i in range(4):
                x1=x+dx[i]
                y1=y+dy[i]
                if (x1<0 or x1>=W or y1<0 or y1>=H):
                    Br[id]-=dst[pos_x+x1][pos_y+y1][0]
                    Bg[id]-=dst[pos_x+x1][pos_y+y1][1]
                    Bb[id]-=dst[pos_x+x1][pos_y+y1][2]
                else:
                    if (abs(src[x1][y1][0]-src[x][y][0])>abs(dst[pos_x+x1][pos_y+y1][0]-dst[pos_x+x][pos_y+y][0])):
                        Br[id]+=src[x1][y1][0]-src[x][y][0]
                    else:
                        Br[id]+=dst[pos_x+x1][pos_y+y1][0]-dst[pos_x+x][pos_y+y][0]
                    if (abs(src[x1][y1][1]-src[x][y][1])>abs(dst[pos_x+x1][pos_y+y1][1]-dst[pos_x+x][pos_y+y][1])):
                        Bg[id]+=src[x1][y1][1]-src[x][y][1]
                    else:
                        Bg[id]+=dst[pos_x+x1][pos_y+y1][1]-dst[pos_x+x][pos_y+y][1]
                    if (abs(src[x1][y1][2]-src[x][y][2])>abs(dst[pos_x+x1][pos_y+y1][2]-dst[pos_x+x][pos_y+y][2])):
                        Bb[id]+=src[x1][y1][2]-src[x][y][2]
                    else:
                        Bb[id]+=dst[pos_x+x1][pos_y+y1][2]-dst[pos_x+x][pos_y+y][2]
    return Br, Bg, Bb

    
if __name__=="__main__":
    img_src=cv2.imread(src_path).astype('float32')
    img_dst=cv2.imread(dst_path).astype('float32')


    size=img_src.shape
    global W,H
    W=size[0]
    H=size[1]


    A=Get_MatrixA()
    Br, Bg, Bb=Get_MatrixB(img_src, img_dst)

    INV=lg.inv(A)
    Br=INV.dot(Br)
    Bg=INV.dot(Bg)
    Bb=INV.dot(Bb)
    for x in range(W):
        for y in range(H):
            id=Get_id(x,y)
            img_dst[pos_x+x][pos_y+y]=[Br[id],Bg[id],Bb[id]]
            for k in range(3):
                img_dst[pos_x+x][pos_y+y][k]=min(max(img_dst[pos_x+x][pos_y+y][k],0),255)
            

    cv2.imwrite(res_path, img_dst)
    

