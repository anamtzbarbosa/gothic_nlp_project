import numpy as np
import torch
import matplotlib.pyplot as plt
def synthesize(RNN, h0, x0, n, rng):
    h = h0.copy()
    x = x0.copy()
    indices = []

    for t in range(n):
        a = RNN['W'] @ h + RNN['U']@x + RNN['b']
        h = np.tanh(a)
        o = RNN['V']@h + RNN['c']
        p = np.exp(o) / np.sum(np.exp(o),axis=0,keepdims=True)
        cp = np.cumsum(p, axis=0)
        u = rng.uniform(size=1)
        ii = np.argmax(cp-u>0)
        indices.append(ii)
        x = np.zeros((K,1))
        x[ii] = 1
    Y  = np.eye(K)[:, indices]
    return Y

def forward_pass(RNN, X, Y, h0):
    h = h0.copy()
    loss = 0
    h_list = [h0]
    p_list = []

    for t in range(X.shape[1]): # same as seq length
        x = X[:, t:t+1]  # get current input column
        a = RNN['W'] @ h + RNN['U']@x + RNN['b']
        h = np.tanh(a)
        h_list.append(h)
        o = RNN['V']@h + RNN['c']
        p = np.exp(o) / np.sum(np.exp(o),axis=0,keepdims=True)
        p_list.append(p)
        loss += np.sum(-Y[:, t:t+1] * np.log(p))
    loss = loss / X.shape[1]  # average over sequence
    return loss, h_list, p_list

def backward_pass(RNN, X, Y, h_list, p_list):
    W = RNN['W']
    V = RNN['V']
    grads = {}
    grads['W'] = np.zeros_like(RNN['W'])
    grads['U'] = np.zeros_like(RNN['U'])
    grads['V'] = np.zeros_like(RNN['V'])
    grads['b'] = np.zeros_like(RNN['b'])
    grads['c'] = np.zeros_like(RNN['c'])
    g_list = []
    for t in range(SEQ_LENGTH):
        g = -(Y[:,t:t+1] - p_list[t]) #dL/dot
        g_list.append(g)

    dLdh = V.T @ g_list[-1]
    #dLda = dLdh * (1-h_list[-1]**2) # h is just tanh(a), can use it
    # commented out, it cant be used in last t
    dLda = np.zeros_like(h_list[0])
    for t in range(SEQ_LENGTH-1,-1,-1):
        grads['V'] += g_list[t] @ h_list[t+1].T
        grads['c'] += g_list[t]
        dLdh = V.T @ g_list[t] + W.T @ dLda
        dLda = dLdh * (1 - h_list[t+1]**2)

        grads['W'] += dLda @ h_list[t].T
        grads['U'] += dLda @ X[:, t:t+1].T
        grads['b'] += dLda
    
    for grad in grads.keys():
        grads[grad] = grads[grad] / SEQ_LENGTH
    return grads

# assumes X has size d x tau, h0 has size m x 1, etc
def ComputeGradsWithTorch(X, y, h0, RNN):

    tau = X.shape[1]

    Xt = torch.from_numpy(X)
    ht = torch.from_numpy(h0)

    torch_network = {}
    for kk in RNN.keys():
        torch_network[kk] = torch.tensor(RNN[kk], requires_grad=True)


    ## give informative names to these torch classes        
    apply_tanh = torch.nn.Tanh()
    apply_softmax = torch.nn.Softmax(dim=0) 
    
    # create an empty tensor to store the hidden vector at each timestep
    Hs = torch.empty(h0.shape[0], X.shape[1], dtype=torch.float64)
    
    hprev = ht
    for t in range(tau):

        #### BEGIN your code ######

        # Code to apply the RNN to hprev and Xt[:, t:t+1] to compute the hidden scores "Hs" at timestep t
        # (ie equations (1,2) in the assignment instructions)
        # Store results in Hs

        # Don't forget to update hprev!
        x = Xt[:,t:t+1]        
        a = torch_network['W'] @ hprev + torch_network['U'] @ x + torch_network['b']
        ht = apply_tanh(a)
        hprev = ht
        Hs[:,t:t+1] = ht
        #### END of your code ######            

    Os = torch.matmul(torch_network['V'], Hs) + torch_network['c']        
    P = apply_softmax(Os)    
    
    # compute the loss
    
    loss = torch.mean(-torch.log(P[y, np.arange(tau)]))
    
    # compute the backward pass relative to the loss and the named parameters 
    loss.backward()

    # extract the computed gradients and make them numpy arrays
    grads = {}
    for kk in RNN.keys():
        grads[kk] = torch_network[kk].grad.numpy()

    return grads

def train_RNN(RNN, book_data, rng, n_epochs=3):
    m_dict = {kk: np.zeros_like(RNN[kk]) for kk in RNN.keys()}
    v_dict = {kk: np.zeros_like(RNN[kk]) for kk in RNN.keys()}
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    e = 0
    t = 1
    hprev = np.zeros((m, 1))
    smooth_loss = -np.log(1/K)
    loss_hist = []
    synth_samples = []
    for epoch in range(n_epochs):
        while e < len(book_data) - SEQ_LENGTH - 1:
            X_chars = book_data[e:e+SEQ_LENGTH]
            label = book_data[e+1:e+1+SEQ_LENGTH]
            X = np.eye(K)[:, [char_to_ind[c] for c in X_chars]]
            Y = np.eye(K)[:, [char_to_ind[c] for c in label]]
            loss, h_list, p_list = forward_pass(RNN, X, Y, hprev)
            grads = backward_pass(RNN, X, Y, h_list, p_list)
            for kk in RNN.keys():
                m_dict[kk] = beta1*m_dict[kk] + (1-beta1)*grads[kk]
                v_dict[kk] = beta2*v_dict[kk] + ((1-beta2)*(grads[kk]**2))
                m_hat = m_dict[kk]/(1-beta1**t)
                v_hat = v_dict[kk]/(1-beta2**t)
                RNN[kk] = RNN[kk] - (ETA*m_hat/ (np.sqrt(v_hat) + epsilon) )
            
            hprev = h_list[-1]
            smooth_loss = 0.999*smooth_loss + 0.001*loss
            if (t%100 == 0):
                print(f"Step {t}, smooth loss = {smooth_loss:.4f}")
                loss_hist.append(smooth_loss)
            if (t==1 or t%10000 == 0):
                Y_synthetic = synthesize(RNN,hprev,X[:,0:1],200,rng)
                synth_text = "".join([ind_to_char[np.argmax(Y_synthetic[:, i])] for i in range(Y_synthetic.shape[1])])
                synth_samples.append(f"iter={t}\n{synth_text}")
                print("\n***** SYNTHESIZED TEXT *****")
                print(synth_text)
                print("******************************\n")
            e += SEQ_LENGTH
            t +=1
        e = 0
        hprev = np.zeros((m, 1))
    return loss_hist, synth_samples