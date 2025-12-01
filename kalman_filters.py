import numpy as np

# ---------- Constant Velocity (6-state) ----------
class KalmanFilter3D_CV:
    # State: [x,y,z, vx,vy,vz]^T
    def __init__(self, sigma_vel=3.0, init_x=None, init_P=None):
        self.x = np.zeros((6,1)) if init_x is None else init_x.reshape((6,1)).copy()
        self.P = np.eye(6) * 1000 if init_P is None else init_P.copy()
        self.H = np.zeros((3,6)); self.H[:3,:3] = np.eye(3)
        self.R = np.eye(3)  # measurement noise (tune per class)
        self.I = np.eye(6)
        self.sigma_vel = sigma_vel

    def predict(self, dt, occluded=False):
        F = np.eye(6); F[0,3] = F[1,4] = F[2,5] = dt
        self.x = F.dot(self.x)
        Q = self.build_process_noise(dt)
        # optionally inflate Q more when occluded
        if occluded:
            Q *= 3.0
        self.P = F.dot(self.P).dot(F.T) + Q

    def update(self, Z, meas_R=None):
        if meas_R is not None:
            self.R = meas_R
        y = Z.reshape((3,1)) - self.H.dot(self.x)
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))
        self.x = self.x + K.dot(y)
        self.P = (self.I - K.dot(self.H)).dot(self.P)

    def build_process_noise(self, dt):
        q = self.sigma_vel**2
        Q = np.zeros((6,6))
        for i in range(3):
            Q[i,i]     = 0.25 * dt**4 * q
            Q[i,i+3]   = 0.5  * dt**3 * q
            Q[i+3,i]   = 0.5  * dt**3 * q
            Q[i+3,i+3] = dt**2 * q
        return Q


# ---------- Constant Acceleration (9-state) ----------
class KalmanFilter3D_CA:
    # State: [x,y,z, vx,vy,vz, ax,ay,az]^T
    def __init__(self, sigma_acc=0.5, init_x=None, init_P=None):
        self.x = np.zeros((9,1)) if init_x is None else init_x.reshape((9,1)).copy()
        self.P = np.eye(9) * 1000 if init_P is None else init_P.copy()
        self.H = np.zeros((3,9)); self.H[:3,:3] = np.eye(3)
        self.R = np.eye(3)
        self.I = np.eye(9)
        self.sigma_acc = sigma_acc

    @staticmethod
    def build_transition(dt):
        return np.array([
            [1, 0, 0, dt, 0,  0,  0.5*dt*dt,       0,       0],
            [0, 1, 0, 0,  dt, 0,  0,       0.5*dt*dt,       0],
            [0, 0, 1, 0,  0,  dt, 0,             0, 0.5*dt*dt],
            [0, 0, 0, 1, 0,  0,  dt,            0,       0],
            [0, 0, 0, 0, 1,  0,  0,            dt,       0],
            [0, 0, 0, 0, 0,  1,  0,             0,      dt],
            [0, 0, 0, 0, 0,  0,  1,             0,       0],
            [0, 0, 0, 0, 0,  0,  0,             1,       0],
            [0, 0, 0, 0, 0,  0,  0,             0,       1]
        ])

    def build_process_noise(self, dt):
        q = self.sigma_acc**2
        Q = np.zeros((9,9))
        # block for x/v/a (same across axes)
        for i in range(3):
            # index mapping: pos=i, vel=i+3, acc=i+6
            p = i
            v = i+3
            a = i+6
            Q[p,p] = 0.25*dt**4*q
            Q[p,v] = 0.5*dt**3*q
            Q[p,a] = 0.5*dt**2*q
            Q[v,p] = 0.5*dt**3*q
            Q[v,v] = dt**2*q
            Q[v,a] = dt*q
            Q[a,p] = 0.5*dt**2*q
            Q[a,v] = dt*q
            Q[a,a] = q
        return Q

    def predict(self, dt, occluded=False):
        F = self.build_transition(dt)
        self.x = F.dot(self.x)
        Q = self.build_process_noise(dt)
        if occluded:
            Q *= 3.0
        self.P = F.dot(self.P).dot(F.T) + Q

    def update(self, Z, meas_R=None):
        if meas_R is not None:
            self.R = meas_R
        y = Z.reshape((3,1)) - self.H.dot(self.x)
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))
        self.x = self.x + K.dot(y)
        self.P = (self.I - K.dot(self.H)).dot(self.P)