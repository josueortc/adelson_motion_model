import numpy as np
import torch


def filter_generation():
    #Define the space axis of the filters
    nx = 16.0
    max_x = 2.0
    dx = (max_x*2)/nx

    #A row vector holding spatial sampling intervals
    x_filt = np.linspace(-max_x, max_x, num=nx)

    #Spatial filter parameters

    sx = 0.5
    sf = 1.1

    #Spatial filter response

    gauss = np.exp(-x_filt**2/(sx**2))
    even_x = np.cos(2*np.pi*sf*x_filt)*gauss
    odd_x = np.sin(2*np.pi*sf*x_filt)*gauss

    #Define the time axis of the filters
    nt = 3
    max_t = 0.5
    dt = max_t/nt

    #A column vector holding the temporal sampling intervals

    t_filt = np.linspace(0, max_t, nt)

    #Temporal filter parameters

    k = 100
    slow_n = 9
    fast_n = 6
    beta = 0.9

    #Temporal filter response

    slow_t = (k*t_filt)*slow_n*np.exp(-k*t_filt)*(1/np.math.factorial(slow_n) - beta*((k*t_filt)**2)/np.math.factorial(slow_n+2))

    fast_t = (k*t_filt)*fast_n*np.exp(-k*t_filt)*(1/np.math.factorial(fast_n) - beta*((k*t_filt)**2)/np.math.factorial(fast_n+2))

    #Step 1b

    even_xx = np.outer(even_x, even_x)
    odd_xx = np.outer(odd_x, odd_x)

    e_slow = np.random.random((slow_t.shape[0], even_x.shape[0], even_x.shape[0]))
    e_fast = np.random.random((slow_t.shape[0], even_x.shape[0], even_x.shape[0]))
    o_slow = np.random.random((slow_t.shape[0], even_x.shape[0], even_x.shape[0]))
    o_fast = np.random.random((slow_t.shape[0], even_x.shape[0], even_x.shape[0]))

    #Step 1c

    for i in range(even_x.shape[0]):
        e_slow[:,:,i] = np.outer(slow_t, even_xx[:,i])
        e_fast[:,:,i] = np.outer(fast_t, even_xx[:,i])
        o_slow[:,:,i] = np.outer(slow_t, odd_xx[:,i])
        o_fast[:,:,i] = np.outer(fast_t, odd_xx[:,i])

    #Step 2

    left_1 = o_fast + e_slow
    left_2 = o_slow + e_fast
    right_1 = o_fast + e_slow
    right_2 = o_slow + e_fast


    left_1 = torch.from_numpy(left_1).float()
    left_2 = torch.from_numpy(left_2).float()
    right_1 = torch.from_numpy(right_1).float()
    right_2 = torch.from_numpy(right_2).float()

    left_1 = left_1.unsqueeze(0).unsqueeze(0)
    left_2 = left_2.unsqueeze(0).unsqueeze(0)
    right_1 = right_1.unsqueeze(0).unsqueeze(0)
    right_2 = right_2.unsqueeze(0).unsqueeze(0)


    return left_1, left_2, right_1, right_2


def motion_model(input):

    assert len(input.size()) == 5
    left_1, left_2, right_1, right_2 = filter()

    #Convolution with input
    resp_right_1 = torch.nn.functional.conv3d(input, right_1)
    resp_right_2 = torch.nn.functional.conv3d(input, right_2)
    resp_left_1 = torch.nn.functional.conv3d(input, left_1)
    resp_left_2 = torch.nn.functional.conv3d(input, left_2)

    resp_right_1 = resp_right_1 ** 2
    resp_right_2 = resp_right_2 ** 2
    resp_left_1 = resp_left_1 ** 2
    resp_left_2 = resp_left_2 ** 2

    energy_right= resp_right_1 + resp_right_2
    energy_left= resp_left_1 + resp_left_2

    total_energy = energy_right.sum() + energy_left.sum()

    #Normalization
    RR1 = resp_right_1/total_energy
    RR2 = resp_right_2/total_energy
    LR1 = resp_left_1/total_energy
    LR2 = resp_left_2/total_energy

    motion = torch.cat([RR1,RR2,LR1, LR2], 1)

    return motion
