



def L1to(into):
    if(into == 1):
        out = 0
    elif(into == 2):
        out = 0.25
    elif(into == 3):
        out = 0.75
    return out

def L2to(into):
    if(into == 1):
        out = 0.75
    elif(into == 2):
        out = 0
    elif(into == 3):
        out = 0.25
    return out

def L3to(into):
    if(into == 1):
        out = 0.25
    elif(into == 2):
        out = 0.75
    elif(into == 3):
        out = 0
    return out

def R1to(into):
    if(into == 1):
        out = 0
    elif(into == 2):
        out = 0.75
    elif(into == 3):
        out = 0.25
    return out

def R2to(into):
    if(into == 1):
        out = 0.25
    elif(into == 2):
        out = 0
    elif(into == 3):
        out = 0.75
    return out

def R3to(into):
    if(into == 1):
        out = 0.75
    elif(into == 2):
        out = 0.25
    elif(into == 3):
        out = 0
    return out

def L1(U1, U2, U3):
    out = L1to(1)*U1 + L1to(2)*U2 + L1to(3)*U3
    return out


def L2(U1, U2, U3):
    out = L2to(1)*U1 + L2to(2)*U2 + L2to(3)*U3
    return out

def L3(U1, U2, U3):
    out = L3to(1)*U1 + L3to(2)*U2 + L3to(3)*U3
    return out



def R1(U1, U2, U3):
    out = R1to(1)*U1 + R1to(2)*U2 + R1to(3)*U3
    return out


def R2(U1, U2, U3):
    out = R2to(1)*U1 + R2to(2)*U2 + R2to(3)*U3
    return out

def R3(U1, U2, U3):
    out = R3to(1)*U1 + R3to(2)*U2 + R3to(3)*U3
    return out



def U1Update(U1_old, U2_old, U3_old):
    gamma = 0.5
    L = L1(U1_old, U2_old, U3_old)
    R = R1(U1_old, U2_old, U3_old)
    U1 = gamma*max(L, R)
    return U1



def U2Update(U1_old, U2_old, U3_old):
    gamma = 0.5
    Reward = 1
    L = L2(U1_old, U2_old, U3_old)
    R = R2(U1_old, U2_old, U3_old)
    U2 = Reward + gamma*max(L, R)
    return U2



def U3Update(U1_old, U2_old, U3_old):
    gamma = 0.5
    L = L3(U1_old, U2_old, U3_old)
    R = R3(U1_old, U2_old, U3_old)
    U3 = gamma*max(L, R)
    return U3


U1_old = 0
U2_old = 0
U3_old = 0

for i in range(2):
    print("iter: ", i)
    print("\nold: ", U1_old, U2_old, U3_old)
    U1_new = U1Update(U1_old, U2_old, U3_old)
    U2_new = U2Update(U1_old, U2_old, U3_old)
    U3_new = U3Update(U1_old, U2_old, U3_old)
    print("\nnew: ",U1_new, U2_new, U3_new)
    U1_old = U1_new
    U2_old = U2_new
    U3_old = U3_new
