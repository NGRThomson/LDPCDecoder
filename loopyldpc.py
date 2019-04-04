import numpy as np

def loopyldpc(H, y, p, maxiter=20):
    # Initialisation
    # Store number of c and v nodes based on the shape of the parity check matrix H.
    
    num_check, num_v = H.shape
        
    # Creates vectors pxigy representing P(x(j) = i|y(j)) where y(j) is the jth element
    # of the received vector,  x(j) is the jth element of the transmitted vector and i is (0,1).
    # For example if y(2) = 0 element 2 of px1gy corresponds to P(x(2) = 1|y(2)) = 0.1
    # since the flip probability p is 0.1.
    p1 = abs(y-p) # gives 0.9 if y[i] = 1 and  p=0.1 and 0.1 if y[i] = 0
    p0 = 1-p1

    # Initialise q1, q0, r1 and r0 matrices
    q0 = np.multiply(p0, H)
    q1 = np.multiply(p1, H)
    
    r0 = np.zeros((num_check, num_v))
    r1 = np.zeros((num_check, num_v))

    for iteration in range(maxiter):
        
        # Horizontal step 
        # iterate through each row of the parity check matrix
        for i in range(num_check):
        # bits that are connected to parity check i (always has length 4) 
            bits = np.array(np.where(H[i,:] == 1))
            bits = bits.ravel()
            for b in bits:
                notbitb = np.setdiff1d(bits, b)
                # update r1
                sumprod = np.prod(q1[i,notbitb])
                for l in notbitb:
                    sumprod += q1[i,l]*np.prod(q1[i, np.setdiff1d(notbitb,l)])
                r1[i,b] = sumprod
                                
                #update r0
                sumprod = np.prod(q0[i,notbitb])
                for l in notbitb:
                    sumprod += q0[i,l] * np.prod(q0[i, np.setdiff1d(notbitb,l)])
                r0[i,b] = sumprod
        
        # Vertical step
        for j in range(num_v):
            checks = np.array(np.where(H[:,j] == 1))
            checks = checks.ravel()
            for c in checks:
                notcheckc = np.setdiff1d(checks, c)
                q1[c, j] = np.prod(r1[notcheckc, j]) * p1[j]
                q0[c, j] = np.prod(r0[notcheckc, j]) * p0[j]
                
                # normalise (equivalent to alpha constant on pg. 561 Mackay)
                const = q1[c,j] +  q0[c,j]
                q1[c,j] = q1[c,j]/ const
                q0[c,j] = q0[c,j]/ const
        
        # Calculate psuedoposterior probabilities
        qn1 = np.zeros((1, num_v))
        qn0 = np.zeros((1, num_v))
        for k in range(num_v):
            chk = np.array(np.where(H[:,k] == 1))
            qn0[0, k] = np.prod(r0[chk, k]) * p0[k]
            qn1[0, k] = np.prod(r1[chk, k]) * p1[k]
        
            
            # normalise (equivalent to alpha constant on pg. 561 Mackay)
            const = qn0[0, k] + qn1[0, k]
            qn0[0, k] = qn0[0, k]/const
            qn1[0, k] = qn1[0, k]/const
                
        xhat = np.array(qn1 > 0.5).astype(int)
        z = np.mod(H@xhat.T,2)
        
        failedchecks = sum(z)
        print("Iteration number:" + str(iteration + 1) + '\n' + "Parity check failures:" + str(failedchecks))
        if failedchecks == 0:
            success = 0
            decode = xhat
            break
        else:
            success = -1
            decode = xhat
    return success, decode

