import numpy as np
import cv2
import scipy.optimize 

def get_corners(img):
    total_corners=[]
    for i in range(13):


        found, corners = cv2.findChessboardCorners(img[i], (6,9), flags=cv2.CALIB_CB_ADAPTIVE_THRESH)
        corners=corners.reshape(-1,2)

        total_corners.append(corners)
    return total_corners

def get_world_points():
    for i in range(54):
        Yi, Xi = np.indices((9,6))
        world_points = np.stack(((Xi.ravel()) * 21.5, (Yi.ravel()) * 21.5)).T
    return world_points


    
def get_all_h(total_corners):
    world_points = get_world_points()
    all_h = []
    
    for corners in total_corners:

        X = world_points[:, 0]
        Y = world_points[:, 1]
        x = corners[:, 0]
        y = corners[:,1]
        Eq = []
        for i in range(corners.shape[0]):
            eq1 = np.array([-X[i],- Y[i], -1, 0, 0, 0, X[i]*x[i], Y[i]*x[i], x[i]])
            Eq.append(eq1)
            eq2 = np.array([0, 0, 0, -X[i], -Y[i], -1, X[i]*y[i], Y[i]*y[i], y[i]])
            Eq.append(eq2)

        Eq = np.array(Eq)
        U, E, V = np.linalg.svd(Eq, full_matrices=True)
        H = V[-1,:].reshape((3, 3))
        H = H / H[2,2] 
        all_h.append(H)
    return all_h



def get_B(H):
    V=[]
    for h in H:
        h1=h[:,0]
        h2=h[:,1]   
        V11=np.array([h1[0]*h1[0], h1[0]*h1[1] + h1[1]*h1[0], h1[1]*h1[1], h1[2]*h1[0] + h1[0]*h1[2], h1[2]*h1[1] + h1[1]*h1[2], h1[2]*h1[2]])
        V12=np.array([h1[0]*h2[0], h1[0]*h2[1] + h1[1]*h2[0], h1[1]*h2[1], h1[2]*h2[0] + h1[0]*h2[2], h1[2]*h2[1] + h1[1]*h2[2], h1[2]*h2[2]])
        V22=np.array([h2[0]*h2[0], h2[0]*h2[1] + h2[1]*h2[0], h2[1]*h2[1], h2[2]*h2[0] + h2[0]*h2[2], h2[2]*h2[1] + h2[1]*h2[2], h2[2]*h2[2]])
        V.append(np.transpose(V12))
        V.append(np.transpose(V11-V22))
        
    V=np.array(V)
    U, sigma, V = np.linalg.svd(V)
    b = V[-1, :]    

    B = np.zeros((3,3))
    B[0,0] = b[0]
    B[0,1] = b[1]
    B[0,2] = b[3]    #b = [B11, B12, B22, B13, B23, B33]
    B[1,0] = b[1]
    B[1,1] = b[2]
    B[1,2] = b[4]
    B[2,0] = b[3]
    B[2,1] = b[4]
    B[2,2] = b[5]

    return B

def get_K(B):
    v0=(B[0,1]*B[0,2]-B[0,0]*B[1,2])/(B[0,0]*B[1,1]-B[0,1]**2)
    lambd = B[2,2] - (B[0,2]**2 + v0 * (B[0,1] * B[0,2] - B[0,0] * B[1,2]))/B[0,0]
    alpha = np.sqrt(lambd/B[0,0])
    beta = np.sqrt(lambd * (B[0,0]/(B[0,0] * B[1,1] - B[0,1]**2)))
    gamma = -(B[0,1] * alpha**2 * beta) / lambd 
    u0 = (gamma * v0 / beta) - (B[0,2] * alpha**2 / lambd)

    K = np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]])
    return K



def get_RT(K,H):
    RT=[]
    for h in H:
        h1=h[:,0]
        h2=h[:,1]
        h3=h[:,2]
        h1_norm = np.linalg.norm(np.dot(np.linalg.inv(K), h1),2)
        lambd = 1 / h1_norm
        r1 = np.dot(lambd, np.linalg.inv(K).dot(h1))
        r2 = np.dot(lambd, np.linalg.inv(K).dot(h2))
        r3 = np.cross(r1, r2)
        t = np.dot(lambd, np.linalg.inv(K).dot(h3))
        result = np.column_stack((r1, r2, r3, t))
        RT.append(result)
    return RT


def K_k_values(K,k):
    alpha = K[0,0]
    gamma = K[0,1]
    beta = K[1,1]
    u0 = K[0,2]
    v0 = K[1,2]
    k1 = k[0]
    k2 = k[1]

    x0 = np.array([alpha, gamma, beta, u0, v0, k1, k2])
    return x0




def reprojection(K,k,total_RT,world_points,total_corners):
    alpha = K[0,0]
    gamma = K[0,1]
    beta = K[1,1]
    u0 = K[0,2]
    v0 = K[1,2]
    k1 = k[0]
    k2 = k[1]

    
    error_list_images=[]
    rep_points_images=[]
    for i in range(np.shape(total_corners)[0]):

        image_error=0
        rep_points=[]
        
        RT=total_RT[i]
        RT_3=np.delete(RT, 2, 1)  #No Z in RT
        KRT_3=np.dot(K,RT_3)
        
        for j in range(len(world_points)):
            world_points_4=np.zeros((4,1))
            world_points_4[0]=(world_points[j][0])
            world_points_4[1]=(world_points[j][1])
            world_points_4[2]=0
            world_points_4[3]=1
            world_points_3=np.zeros((3,1))
            world_points_3[0]=(world_points[j][0])
            world_points_3[1]=(world_points[j][1])
            world_points_3[2]=1
            
            RTX = np.dot(RT, world_points_4)
            x =  RTX[0] / RTX[2]                  #normalized world coordinates of image
            y = RTX[1] / RTX[2]

            uvw=np.dot(KRT_3,world_points_3)       #World points to image points
            u=uvw[0]/uvw[2]
            v=uvw[1]/uvw[2]
            
            u_dash=u+(u-u0)*(k1*(x**2 + y**2)+ k2*(x**2 + y**2)**2)
            v_dash=v+(v-v0)*(k1*(x**2 + y**2)+ k2*(x**2 + y**2)**2)
            
            rep_points.append([u_dash,v_dash])
            
            mij=np.array([total_corners[i][j][0],total_corners[i][j][1],1], dtype = 'float').reshape(3,1)
            mij_dash=np.array([u_dash,v_dash,1], dtype = 'float').reshape(3,1)
            error=np.linalg.norm((mij-mij_dash),2)
            image_error=image_error+error
            
        image_average_error=image_error/(54)
        error_list_images.append(image_average_error)
        rep_points_images.append(rep_points)
    average_error_total=np.mean(error_list_images)
                     
    return average_error_total,rep_points_images




def loss(x0,total_RT, total_corners, world_points):
    alpha, gamma, beta, u0, v0, k1, k2 = x0
    K = np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]]).reshape(3,3)
    k = np.array([k1, k2]).reshape(2,1)
    
    error_list_images=[]
    
    for i in range(np.shape(total_corners)[0]):

        image_error=0
        RT=total_RT[i]
        RT_3=np.delete(RT, 2, 1)  #No Z in RT
        KRT_3=np.dot(K,RT_3)
        
        for j in range(len(world_points)):
            world_points_4=np.zeros((4,1))
            world_points_4[0]=(world_points[j][0])
            world_points_4[1]=(world_points[j][1])
            world_points_4[2]=0
            world_points_4[3]=1
            world_points_3=np.zeros((3,1))
            world_points_3[0]=(world_points[j][0])
            world_points_3[1]=(world_points[j][1])
            world_points_3[2]=1
            
            RTX = np.dot(RT, world_points_4)
            x =  RTX[0] / RTX[2]                  #normalized world coordinates of image
            y = RTX[1] / RTX[2]

            uvw=np.dot(KRT_3,world_points_3)       #World points to image points
            u=uvw[0]/uvw[2]
            v=uvw[1]/uvw[2]
            
            u_dash=u+(u-u0)*(k1*(x**2 + y**2)+ k2*(x**2 + y**2)**2)
            v_dash=v+(v-v0)*(k1*(x**2 + y**2)+ k2*(x**2 + y**2)**2)
            
            mij=np.array([total_corners[i][j][0],total_corners[i][j][1],1], dtype = 'float').reshape(3,1)
            mij_dash=np.array([u_dash,v_dash,1], dtype = 'float').reshape(3,1)
            error=np.linalg.norm((mij-mij_dash),2)
            image_error=image_error+error
        image_average_error=image_error/54
        error_list_images.append(image_average_error)
                     
    return np.array(error_list_images)
    
    
    
    
    
    
    
    


img = [cv2.imread(f'./Calibration_Imgs/{i}.jpg') for i in range(13)]
total_corners=get_corners(img)
world_points=get_world_points()
H=get_all_h(total_corners)
B=get_B(H)
print(B)
K=get_K(B)
print(K)

total_RT=get_RT(K,H)

k=np.array([0,0]).reshape(2,1)
x0 = K_k_values(K, k)
res = scipy.optimize.least_squares(loss, x0=x0, method="lm", args=[total_RT, total_corners, world_points])

alpha, gamma, beta, u0, v0, k1, k2 = res.x
K_rec = np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]]).reshape(3,3)
k_rec = np.array([k1, k2]).reshape(2,1)
total_RT_rec=get_RT(K_rec,H)
print(K_rec)
print(k_rec)

error_images,reproj_init_points=reprojection(K,k,total_RT,world_points,total_corners)
rec_error_images,reproj_points=reprojection(K_rec,k_rec,total_RT_rec,world_points,total_corners)

print(error_images)

print(rec_error_images)
D=np.array([k_rec[0],k_rec[1],0,0],np.float32)
K=np.array(K_rec,np.float32).reshape(3,3)

# for i, image_points in enumerate(reproj_points):
#     image=cv2.undistort(img[i], K, D)
#     for point in image_points:
#         x = int(point[0])
#         y = int(point[1])
#         image=cv2.circle(image,(x,y),5,(255,0,0),3)
#     cv2.imwrite('./results/'+str(i)+'.jpg',image)
        
    
    

