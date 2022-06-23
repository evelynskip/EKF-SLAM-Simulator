from cmath import pi
from turtle import shape
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, atan2,sqrt
import time

class Plotting:
    
    def __init__(self):
        self.true_x, self.true_y, self.true_theta = [], [], []
        self.pred_x, self.pred_y, self.pred_theta = [], [], []
        self.pred_lm_x, self.pred_lm_y = [], []
        self.time = []

    def update(self, true_states, pred_states, time):
        self.true_x.append(true_states[0])
        self.true_y.append(true_states[1])
        self.true_theta.append(true_states[2])

        self.pred_x.append(pred_states[0])
        self.pred_y.append(pred_states[1])
        self.pred_theta.append(pred_states[2])

        self.pred_lm_x.append(pred_states[3])
        self.pred_lm_y.append(pred_states[4])

        self.time.append(time)

    @staticmethod
    def computeTriangle(rob,strRobType):
        if strRobType == 'Predicted':
            pos=rob.pred_pos
        elif strRobType == 'True':
            pos=rob.true_pos

        x = pos[0]
        y = pos[1]
        theta = pos[2]
        p=[]
        p.append([-rob.length/3, -rob.wheelbase/2]) 
        p.append([2*rob.length/3, 0]) 
        p.append([-rob.length/3, rob.wheelbase/2])
        p.append([-rob.length/3, -rob.wheelbase/2])
        R = np.asarray([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]).reshape([2,2])
        p = np.asarray(p)
        p= R.dot(p.T)
        p[0][:]=p[0][:]+x
        p[1][:]=p[1][:]+y
        p=p.T
        return p
    
    @staticmethod
    def get_covariance_ellipse_points(mu, P, base_circ=[]):

        if len(base_circ) == 0:
            N = 20
            phi = np.linspace(0, 2*np.pi, N)
            x = np.reshape(np.cos(phi), (-1,1))
            y = np.reshape(np.sin(phi), (-1,1))
            base_circ.extend(np.hstack((x,y)).tolist())

        vals, _ = np.linalg.eigh(P)

        offset = 1e-6 - min(0, vals.min())

        G = np.linalg.cholesky(P + offset * np.eye(mu.shape[0]))

        # 3 sigma bound
        circ = 3*np.matmul(np.array(base_circ), G.T) + mu

        return circ

    def plot_covariance(self,mean,cov,ax,N):
        
        for i in range(N):
            idx = 3 + 3*i
            if not (mean[idx+2]==0 and mean[idx]==0):
                P = cov[idx:idx+2,idx:idx+2]
                circ = self.get_covariance_ellipse_points(mean[idx:idx+2].reshape([-1]), P)
                ax.plot(circ[:,0],circ[:,1],color='silver')


    def show(self,rob,landmarks,mean,cov,N,window,ax):
        p_pred = self.computeTriangle(rob,'Predicted')
        p_true = self.computeTriangle(rob,'True')
        
        ax.clear()
        ax.plot(self.true_x, self.true_y, label='True Robot')
        ax.plot(self.pred_x, self.pred_y, label='Predicted Robot')
        ax.plot([mark.x for mark in landmarks], [mark.y for mark in landmarks], 'gX', label='True Landmarks')
        ax.plot([mean[3 + 3 * idx, 0] for idx in range(N)],
                 [mean[4 + 3 * idx, 0] for idx in range(N)], 'rX', label='Predicted Landmarks')
        self.plot_covariance(mean,cov,ax,N)
        #draw the robot
        for i in range(3):
            ax.plot([p_pred[i][0],p_pred[i+1][0]],[p_pred[i][1],p_pred[i+1][1]],color = 'orange')
            ax.plot([p_true[i][0],p_true[i+1][0]],[p_true[i][1],p_true[i+1][1]],color = 'cornflowerblue')
        #draw the observation line
        ax.legend()
        ax.grid()
        ax.set_xlabel("X/m")
        ax.set_ylabel("Y/m")
        window.canvas.draw()
        if flag == 0:
            window.canvas.flush_events() 
         # plt.pause(0.001)

class Landmark:
    def __init__(self, x_pos, y_pos, sig):
        self.x = x_pos
        self.y = y_pos
        self.s = sig
        self.seen = False

        self.x_hat = 0.
        self.y_hat = 0.
        self.s_hat = 0.


class Measurement:
    def __init__(self, rng, ang, j,landmarks):
        self.rng = rng
        self.ang = ang
        self.id = j
        self.landmark = lm_from_id(j, landmarks)

class Robot:
    def __init__(self,vt,wt):
        self.wheelbase = 1
        self.length = 1.5
        self.true_pos = np.array([0.,0.,0.])
        self.pred_pos =np.array([0.,0.,0.])
        self.vt = vt  #linear velocity
        self.wt = wt  #angular velocity

    def motion(self, thet, DT):
        # Avoid divide by zero
        if self.wt == 0.:
            self.wt += 1e-5

        # Motion without noise/errors
        theta_dot = self.wt * DT
        x_dot = (-self.vt/self.wt) * sin(thet) + (self.vt/self.wt) * sin(thet + self.wt*DT)
        y_dot = (self.vt/self.wt) * cos(thet) - (self.vt/self.wt) * cos(thet + self.wt*DT)
        a = np.array([x_dot, y_dot, theta_dot]).reshape(-1, 1)

        # Derivative of above motion model
        b = np.zeros((3, 3))
        b[0, 2] = (-self.vt/self.wt) * cos(thet) + (self.vt/self.wt) * cos(thet + self.wt*DT)
        b[1, 2] = (-self.vt/self.wt) * sin(thet) + (self.vt/self.wt) * sin(thet + self.wt*DT)

        return a, b    

    def pos_update(self,Rt,DT,mean,flag):
            # motion with guassian noise
            x, y, theta = self.true_pos
            v=self.vt
            w=self.wt

            
            theta_dot = self.wt * DT
            x_dot = (-self.vt/self.wt) * sin(theta) + (self.vt/self.wt) * sin(theta + self.wt*DT)
            y_dot = (self.vt/self.wt) * cos(theta) - (self.vt/self.wt) * cos(theta + self.wt*DT)

            theta_error= (np.random.normal(0., sqrt(Rt[2, 2]))) * DT/5
            x_error=(np.random.normal(0., sqrt(Rt[0, 0]))) * DT
            y_error=(np.random.normal(0., sqrt(Rt[1, 1]))) * DT
            theta += theta_dot +theta_error
            if theta<-pi:
                theta = theta+2*pi
            if theta>pi:
                theta = theta-2*pi
            x += x_dot + x_error
            y += y_dot+ y_error

            if flag==0:
                self.true_pos=np.array([x, y, theta])
            if flag==1:
                self.pred_pos=mean[0:3]
    
        

class EKFSLAM:


    def predict(self, rob,N,Rt,Qt,landmarks,rng_max,prev_mean=None, prev_cov=None, ut=None,DT=None):
        Fx = np.eye(3, 3*N+3)

        f, g = rob.motion(prev_mean[2, 0], DT)
        mean = prev_mean + Fx.T @ f
        rob.pos_update(Rt,DT,mean,0)

        #observe
        zs = []
        for lm in landmarks:
            z = sensor(Qt,rob.true_pos,lm,landmarks)
            if (z.rng<rng_max):
                zs.append(z)

        Gt = Fx.T @ g @ Fx + np.eye(3*N+3)
        cov = Gt @ prev_cov @ Gt.T + Fx.T @ Rt @ Fx

        for obs in zs:
            j = obs.landmark.s
            zi = np.array([obs.rng, obs.ang, obs.id]).reshape(-1, 1)
            if not obs.landmark.seen:
                mean[3+3*j, 0] = mean[0, 0] + obs.rng * cos(obs.ang + mean[2, 0])  # x
                mean[4+3*j, 0] = mean[1, 0] + obs.rng * sin(obs.ang + mean[2, 0])  # y
                mean[5+3*j, 0] = obs.landmark.s  # s
                obs.landmark.seen = True

            delt_x = mean[3+3*j, 0] - mean[0, 0]
            delt_y = mean[4+3*j, 0] - mean[1, 0]
            delt = np.array([delt_x, delt_y]).reshape(-1, 1)
            q = delt.T @ delt

            zi_hat = np.zeros((3, 1))
            zi_hat[0, 0] = np.sqrt(q)
            zi_hat[1, 0] = atan2(delt_y, delt_x) - mean[2, 0]
            if zi_hat[1, 0] <-pi:
                zi_hat[1, 0] = zi_hat[1, 0]+2*pi
            if zi_hat[1, 0] > pi:
                zi_hat[1, 0] = zi_hat[1, 0]-2*pi
            zi_hat[2, 0] = obs.landmark.s

            Fxj_a = np.eye(6, 3)
            Fxj_b = np.zeros((6, 3*N))
            Fxj_b[3:, 3*j:3+3*j] = np.eye(3)
            Fxj = np.hstack((Fxj_a, Fxj_b))

            h = np.zeros((3, 6))
            h[0, 0] = -1/np.sqrt(q) * delt_x
            h[0, 1] = -1/np.sqrt(q) * delt_y
            h[0, 3] = 1/np.sqrt(q) * delt_x
            h[0, 4] = 1/np.sqrt(q) * delt_y
            h[1, 0] = 1/q*delt_y#delt_y
            h[1, 1] = -1/q*delt_x
            h[1, 2] = -1
            h[1, 3] = -1/q*delt_y
            h[1, 4] = 1/q*delt_x
            h[2, 5] = 1

            Hti =  (h @ Fxj)
            Kti = cov @ Hti.T @ np.linalg.inv((Hti @ cov @ Hti.T + Qt))

            dzi=zi-zi_hat
            if dzi[1]<-pi:
                dzi[1] = dzi[1]+2*pi
            if dzi[1]>pi:
                dzi[1] = dzi[1]-2*pi
            mean = mean + (Kti @ (dzi))
            cov = (np.eye(cov.shape[0]) - Kti @ Hti) @ cov

        return mean, cov


def lm_from_id(lm_id, lm_list):
    for lmrk in lm_list:
        if lmrk.s == lm_id:
            return lmrk
    return None



def sensor(Qt,states,lm,landmarks):
    #generate Guassian noise,mean=0,standard deviation=0.05
    rng_noise = np.random.normal(0., sqrt(Qt[0, 0]))
    ang_noise = np.random.normal(0., sqrt(Qt[1, 1]))
    z_rng = np.sqrt((states[0] - lm.x) ** 2 + (states[1] - lm.y) ** 2) + rng_noise
    z_ang = atan2(lm.y - states[1], lm.x - states[0]) - states[2] + ang_noise
    if z_ang <-pi:
        z_ang = z_ang+2*pi
    if z_ang > pi:
        z_ang = z_ang-2*pi
    z_j = lm.s
    z = Measurement(z_rng, z_ang, z_j,landmarks)
    return z

class Error:
    def __init__(self):
        # k elements for k errors at k moments
        self.error_lmk = [] 
        self.t = []
        self.error_rob_pos = [] 
        self.error_rob_angle = [] 

    @staticmethod
    def sum_sqrt(*args):
        sum = 0
        for arg in args:
            sum += arg**2
        return sqrt(sum)

    def update(self,lmks,rob,mean,t):
        self.t.append(t)
        self.error_rob_pos.append(self.sum_sqrt(rob.true_pos[0]-rob.pred_pos[0],rob.true_pos[1]-rob.pred_pos[1]))
        self.error_rob_angle.append(self.sum_sqrt(rob.true_pos[2]-rob.pred_pos[2]))
        temp = 0
        n = 0
        for lmk in lmks:
            if lmk.seen == True:
                dx = lmk.x - mean[lmk.s*3+3]
                dy = lmk.y - mean[lmk.s*3+4]
                n+=1
                temp = temp + dx**2 +dy**2
        e_lmks = sqrt(temp)/n
        self.error_lmk.append(e_lmks)

    def plot(self,window,ax):
        ax.clear()
        ax.plot([t for t in self.t],[e_pos for e_pos in self.error_rob_pos],color = 'cornflowerblue')
        ax.grid()
        ax.set_ylabel("Error/m")
        ax.set_xlabel("T/s")
        window.canvas.draw()

def slam_function(window,Plot_flag,DT,rt,qt,v,w,rng_max):
    global flag
    flag=Plot_flag
    '''initialize parameters'''
    if DT == '':
        DT=0.1
    else:
        DT=float(DT)
    if v == '':
        v=2.
    else:
        v=float(v)
    if w == '':
        w=0.2
    else:
        w=float(w)
    if rng_max == '':
        rng_max=15
    else:
        rng_max=float(rng_max)
    # Rt standard deviation of motion noise
    # Qt standard deviation of measurement noise
    if rt == '':
        rt=.01
    else:
        rt=float(rt)

    if qt == '':
        qt=.05
    else:
        qt=float(qt)
    Rt = rt**2*np.eye(3)
    Qt = qt**2*np.eye(3)#.05*np.eye(3)

    t = 0.
    tf = 2*3.05/w
    INF = 1000.
    # set landmarks
    lm1 = Landmark(2., 3., 0)
    lm2 = Landmark(13., 13., 1)
    lm3 = Landmark(-5., 12., 2)
    lm4 = Landmark(-14., 5., 3)
    lm5 = Landmark(0., 25., 4)
    lm6 = Landmark(0.,10,5)
    lm7 = Landmark(3.,15.,6)
    lm8 = Landmark(4.,5.,7)
    lm9 = Landmark(7.,10.,8)
    lm10 = Landmark(6.,7.,9)
    lm11 = Landmark(-8.,15.,10)
    lm12 = Landmark(-5.,0.,11)
    lm13 = Landmark(-7.5,3.,12)
    lm14 = Landmark(-12.,5.,13)
    lm15 = Landmark(-14.,15.,14)
    lm16 = Landmark(-10.,20.,15)
    lm17 = Landmark(-7.5,21.,16)
    lm18 = Landmark(-5.,-5.,17)
    
    landmarks = [lm1, lm2, lm3, lm4, lm5,lm6,lm7]
    N = len(landmarks)

    ekf = EKFSLAM()
    vis = Plotting()
    ave_error = Error()
    #initialize robot
    # u velocity of the robot
    u = np.array([v, w])
    rob=Robot(u[0],u[1])

    # create mean and cov matrix
    xr = rob.true_pos.reshape(-1, 1) #to a column vector
    #print("xr",xr.shape)
    xm = np.zeros((3 * N, 1))
    #print("xm",xm.shape)
    mean = np.vstack((xr, xm))
    #print("mean",mean.shape)
    cov = 1 * np.eye(len(mean))
    cov[:3, :3] = np.zeros((3, 3))
    #print("cov",cov.shape)

    # plt.ion()
    ax=window.ax
    ax1=window.ax1
    if flag == 0:
        ax1.clear()
    while t <= tf:
    
        #EKF-SLAM algorithm
        mean, cov = ekf.predict(rob,N,Rt, Qt,landmarks,rng_max, mean, cov, u,DT)
        #update predicted and true pose
        rob.pos_update(Rt,DT,mean,1)
        #update data for plotting
        vis.update(rob.true_pos.flatten().copy(), mean.flatten().copy(), t)# deep copy
        ave_error.update(landmarks,rob,mean,t)
        if flag == 0:
            vis.show(rob,landmarks,mean,cov,N,window,ax)
            ave_error.plot(window,ax1)
        t += DT

    # performance(np.round(mean, 3),N)
    # plt.ioff()
    # plt.show()
    if flag == 1:
        vis.show(rob,landmarks,mean,cov,N,window,ax)
        ave_error.plot(window,ax1)
    
    return