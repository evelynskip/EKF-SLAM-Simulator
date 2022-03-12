import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, atan2
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

    def show(self,landmarks,mean,N,window):
        
        ax=window.figure.add_axes([0.1,0.1,0.8,0.8])
        ax.cla()
        ax.plot(self.true_x, self.true_y, label='True')
        ax.plot(self.pred_x, self.pred_y, label='Predicted')
        ax.plot([mark.x for mark in landmarks], [mark.y for mark in landmarks], 'gX', label='True Landmarks')
        ax.plot([mean[3 + 3 * idx, 0] for idx in range(N)],
                 [mean[4 + 3 * idx, 0] for idx in range(N)], 'rX', label='Predicted Landmarks')
        ax.legend()
        ax.grid()
        #time.sleep(0.05) 
        plt.pause(0.001)
        window.canvas.draw()


        

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
        self.wheel = 1
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

    def pos_update(self,Rt,DT,mean):
            # motion with guassian noise
            x, y, theta = self.true_pos
            v=self.vt
            w=self.wt

            theta_dot = w
            x_dot = v*cos(theta)
            y_dot = v*sin(theta)

            theta += (theta_dot + np.random.normal(0., Rt[2, 2])) * DT
            x += (x_dot + np.random.normal(0., Rt[0, 0])) * DT
            y += (y_dot + np.random.normal(0., Rt[1, 1])) * DT

            self.true_pos=np.array([x, y, theta])
            self.pred_pos=mean[0:3]
    # def cal_triangle():
        

class EKFSLAM:


    def predict(self, rob,N,Rt,Qt, prev_mean=None, prev_cov=None, ut=None, zt=None,DT=None):
        Fx = np.eye(3, 3*N+3)

        f, g = rob.motion(prev_mean[2, 0], DT)
        # f, g = self.motion(ut[0], ut[1], prev_mean[2, 0], DT)
        mean = prev_mean + Fx.T @ f

        Gt = Fx.T @ g @ Fx + np.eye(3*N+3)
        cov = Gt @ prev_cov @ Gt.T + Fx.T @ Rt @ Fx

        for obs in zt:
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
            zi_hat[2, 0] = obs.landmark.s

            Fxj_a = np.eye(6, 3)
            Fxj_b = np.zeros((6, 3*N))
            Fxj_b[3:, 3*j:3+3*j] = np.eye(3)
            Fxj = np.hstack((Fxj_a, Fxj_b))

            h = np.zeros((3, 6))
            h[0, 0] = -np.sqrt(q) * delt_x
            h[0, 1] = -np.sqrt(q) * delt_y
            h[0, 3] = np.sqrt(q) * delt_x
            h[0, 4] = np.sqrt(q) * delt_y
            h[1, 0] = delt_y
            h[1, 1] = -delt_x
            h[1, 2] = -q
            h[1, 3] = -delt_y
            h[1, 4] = delt_x
            h[2, 5] = q

            Hti = (1/q) * (h @ Fxj)
            Kti = cov @ Hti.T @ np.linalg.inv((Hti @ cov @ Hti.T + Qt))

            mean = mean + (Kti @ (zi-zi_hat))
            cov = (np.eye(cov.shape[0]) - Kti @ Hti) @ cov

        return mean, cov


def lm_from_id(lm_id, lm_list):
    for lmrk in lm_list:
        if lmrk.s == lm_id:
            return lmrk
    return None


def performance(pred,N):
    pred_dict = dict()
    pred_dict['X'] = pred[0, 0]
    pred_dict['Y'] = pred[1, 0]
    pred_dict['THETA'] = pred[2, 0]
    for n in range(N):
        pred_dict['LM_' + str(n) + ' X'] = pred[3 + 3 * n, 0]
        pred_dict['LM_' + str(n) + ' Y'] = pred[4 + 3 * n, 0]
        pred_dict['LM_' + str(n) + ' ID'] = pred[5 + 3 * n, 0]

    print('PREDICTED STATES')
    print(pred_dict)


def sensor(Qt,states,lm,landmarks):
    #generate Guassian noise,mean=0,standard deviation=0.05
    rng_noise = np.random.normal(0., Qt[0, 0])
    ang_noise = np.random.normal(0., Qt[1, 1])
    z_rng = np.sqrt((states[0] - lm.x) ** 2 + (states[1] - lm.y) ** 2) + rng_noise
    z_ang = atan2(lm.y - states[1], lm.x - states[0]) - states[2] + ang_noise
    z_j = lm.s
    z = Measurement(z_rng, z_ang, z_j,landmarks)
    return z



def slam_function(window,DT):
    if DT == '':
        DT=0.1
    else:
        DT=float(DT)
    t = 0.
    tf = 30.
    INF = 1000.
    # set landmarks
    lm1 = Landmark(2., 3., 0)
    lm2 = Landmark(13., 13., 1)
    lm3 = Landmark(-5., 12., 2)
    lm4 = Landmark(-12., 12., 3)
    lm5 = Landmark(0., 25., 4)
    lm6 = Landmark(0.,10,5)
    lm7 = Landmark(5.,15.,6)
    landmarks = [lm1, lm2, lm3, lm4, lm5,lm6,lm7]
    N = len(landmarks)

    ekf = EKFSLAM()
    vis = Plotting()

    # Rt standard deviation of motion noise
    # Qt standard deviation of measurement noise
    Rt = .1*np.eye(3)
    Qt = .05*np.eye(3)

    #initialize robot
    # u velocity of the robot
    u = np.array([2., 0.2])
    rob=Robot(u[0],u[1])

    # create mean and cov matrix
    xr = rob.true_pos.reshape(-1, 1) #to a column vector
    xm = np.zeros((3 * N, 1))
    mean = np.vstack((xr, xm))
    cov = INF * np.eye(len(mean))
    cov[:3, :3] = np.zeros((3, 3))

    # plt.ion()
    
    while t <= tf:
        #update data for plotting
        vis.update(rob.true_pos.flatten().copy(), mean.flatten().copy(), t)# deep copy
        #observe
        zs = []
        for lm in landmarks:
            z = sensor(Qt,rob.true_pos,lm,landmarks)
            zs.append(z)
        #EKF-SLAM algorithm
        mean, cov = ekf.predict(rob,N,Rt, Qt, mean, cov, u, zs,DT)
        #update predicted and true pose
        rob.pos_update(Rt,DT,mean)
        #plot 
        vis.show(landmarks,mean,N,window)
        t += DT

    # performance(np.round(mean, 3),N)
    # plt.ioff()
    # plt.show()
    return