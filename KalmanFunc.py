import numpy as np

def predict(X, A, P, Q):
    X = np.dot(A, X)
    P = np.add(np.dot(A, np.dot(P, np.transpose(A))), Q)

    return X, P

def KalmanGain(P, H, R):
    num = np.dot(P, np.transpose(H))
    denom = np.add(np.dot(H, np.dot(P, np.transpose(H))), R)
    K = np.dot(num, np.linalg.inv(denom))

    return K

def update(X, P, Z, K, H):
    Y = np.subtract(Z, np.dot(H, X))
    I = np.identity(P.shape[0])
    X = np.add(X, np.dot(K, Y))
    P = np.dot(np.subtract(I, np.dot(K, H)), P)

    return X, P

def processNoise(G, Qv):
    Q = np.dot(G, np.dot(Qv, np.transpose(G)))

    return Q



