import library_casual_bandit_paper as library
import matplotlib.pyplot as plt
import numpy as np
n_range = np.array(range(5,21,1))

# you can modify the graph density paramters rho and rho_L to reporduce results in main paper.
rho = 0.4
rho_L = 0.2
num_graphs = 50
s_o,s_p,s_t = library.comapre_alg4_with_full_disc(n_range,rho,rho_L,num_graphs)

# Code to reproduce Fig 2
color_line = np.array([179, 63, 64])/255 ;
plt.figure()
plt_label= 'Learning Observable Induced graph on An(Y)'
plt = library.plot(s_o,color_line,0,n_range,plt_label)
color_line = np.array([1, 119, 179])/255 ;
plt_label= 'Learning POMISs (Alg. 4)'
plt = library.plot(s_p,color_line,plt,n_range,plt_label)
color_line = np.array([72, 161, 77])/255 ;
plt_label= 'Learning Induced graph on An(Y) with all latents'
plt = library.plot(s_t,color_line,plt,n_range,plt_label)


# Code to reproduce Fig 
n_range = np.array(range(5,21,1))
rho = 0.4
rho_L = 0.4
num_graphs = 50
s_o,s_p,s_t = library.comapre_alg4_with_full_disc(n_range,rho,rho_L,num_graphs)

color_line = np.array([179, 63, 64])/255 ;
plt.figure()
plt_label= 'Learning Observable Induced graph on An(Y)'
plt = library.plot(s_o,color_line,0,n_range,plt_label)
color_line = np.array([1, 119, 179])/255 ;
plt_label= 'Learning POMISs (Alg. 4)'
plt = library.plot(s_p,color_line,plt,n_range,plt_label)
color_line = np.array([72, 161, 77])/255 ;
plt_label= 'Learning Induced graph on An(Y) with all latents'
plt = library.plot(s_t,color_line,plt,n_range,plt_label)


# Code to reproduce Fig 2
n_range = np.array(range(5,21,1))
rho = 0.4
rho_L = 0.6
num_graphs = 50
s_o,s_p,s_t = library.comapre_alg4_with_full_disc(n_range,rho,rho_L,num_graphs)

color_line = np.array([179, 63, 64])/255 ;
plt.figure()
plt_label= 'Learning Observable Induced graph on An(Y)'
plt = library.plot(s_o,color_line,0,n_range,plt_label)
color_line = np.array([1, 119, 179])/255 ;
plt_label= 'Learning POMISs (Alg. 4)'
plt = library.plot(s_p,color_line,plt,n_range,plt_label)
color_line = np.array([72, 161, 77])/255 ;
plt_label= 'Learning Induced graph on An(Y) with all latents'
plt = library.plot(s_t,color_line,plt,n_range,plt_label)



# Code to reproduce Fig 4(a)
nodes = 10
rho = 0.3
rho_L = 0.3
num_graphs = 50
Max_samples =  6 * 10**7
pd_a,fd_a = library.comapare_reg_fulldisc_vs_alg4(nodes,rho, rho_L , num_graphs,Max_samples)
plt.figure()
color_line = np.array([179, 63, 64])/255 ;
plt_label= 'Learning Induced graph on An(Y) with all latents + UCB '
plt = library.plot_reg(fd_a,color_line,0,Max_samples ,plt_label)
color_line = np.array([1, 119, 179])/255 ;
plt_label= 'Learning POMISs (Alg. 4) + UCB'
plt = library.plot_reg(pd_a,color_line,plt,Max_samples ,plt_label)



# Code to reproduce Fig 4(b)
nodes = 15
rho = 0.3
rho_L = 0.3
num_graphs = 50
Max_samples =  15 * 10**7
pd_a,fd_a = library.comapare_reg_fulldisc_vs_alg4(nodes,rho, rho_L , num_graphs,Max_samples)
plt.figure()
color_line = np.array([179, 63, 64])/255 ;
plt_label= 'Learning Induced graph on An(Y) with all latents + UCB '
plt = library.plot_reg(fd_a,color_line,0,Max_samples ,plt_label)
color_line = np.array([1, 119, 179])/255 ;
plt_label= 'Learning POMISs (Alg. 4) + UCB'
plt = library.plot_reg(pd_a,color_line,plt,Max_samples ,plt_label)



# Code to reproduce Fig 4(c)
nodes = 20
rho = 0.3
rho_L = 0.3
num_graphs = 50
Max_samples =  6 * 10**8
pd_a,fd_a = library.comapare_reg_fulldisc_vs_alg4(nodes,rho, rho_L , num_graphs,Max_samples)
plt.figure()
color_line = np.array([179, 63, 64])/255 ;
plt_label= 'Learning Induced graph on An(Y) with all latents + UCB '
plt = library.plot_reg(fd_a,color_line,0,Max_samples ,plt_label)
color_line = np.array([1, 119, 179])/255 ;
plt_label= 'Learning POMISs (Alg. 4) + UCB'
plt = library.plot_reg(pd_a,color_line,plt,Max_samples ,plt_label)