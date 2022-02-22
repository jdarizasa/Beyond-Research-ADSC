# coding=utf-8
from scipy import integrate
from xlrd import open_workbook
import scipy.stats as ss
import numpy as np
import scipy as sp
import pytwalk
import matplotlib.pyplot as plt

"""
Formulación de un modelo dinámico tipo SEIR con nacimientos y muertes, y un modelo bayesiano para calcular R0 
en la epidemia de influenza de 1918 usando datos de San Francisco
"""

class forward_mapping:
    """
    esta clase simula el modelo directo y el operador de observación
    """
    def __init__(self):
        """
        parámetros conocidos
        """
        self.mu = 1.0/(75.0*365.0)
        self.xi = 0.0#10.0**-5
        self.N = 550000.0
        self.sigma = 1.0/3.0
        self.gamma = 1.0/5.0
        self.Pobs = 0.75
        self.beta0 = 1.8*(self.sigma+self.mu)*(self.gamma+self.mu)/self.sigma
        
        """
        cargamos los datos de influenza de 1918 en San Francisco
        """
        flu_book = open_workbook('pandemic_infuenza_SF_1918.xls')
        flu_sheet = flu_book.sheet_by_index(0)
        self.flu_data = np.asarray(flu_sheet.col_values(2,6,69))
        self.flu_time = np.linspace(0.0,62.0,63)
    
        """
        pesos para la cuadratura trapezoidal
        """
        self.weigths = np.ones(11)
        self.weigths[0] = 0.5
        self.weigths[-1] = 0.5
    
    def rhs(self,x,t,p):    
        """
        Lado derecho del modelo de epidemias 
        """
        fx = np.zeros(3)    
        inc = p[0]*x[1]     #cuadrar parametros que se postulan (dejarlos quietos) y los que se infieren si molestarlos
        fx[0] = (inc+self.xi)*(1.0-x[0]-x[1]-x[2]) - (self.sigma + self.mu)*x[0]
        fx[1] = self.sigma*x[0] - (self.gamma + self.mu)*x[1]
        fx[2] = self.gamma*x[1] - self.mu*x[2]  #pronostico de sistema dinamico es el objetivo
        return fx

    def soln(self,p):
        """
        integra la ecuacion diferencial ordinaria
        """
        x0 = np.array([p[1],p[2],p[3]])
        return integrate.odeint(self.rhs,x0,self.flu_time,args=(p,))

    def solve(self,p):
        """
        evalua el valor esperado de la verosimilitud
        """
        x0 = np.array([p[1],p[2],p[3]])
        nn = len(self.flu_time)
        dt = 1.0/nn
        n_quad = 10*nn+1
        t_quad = np.linspace(self.flu_time[0],self.flu_time[-1],n_quad)
        soln = integrate.odeint(self.rhs,x0,t_quad,args=(p,))
        result = np.zeros(nn)
        for k in range(nn):
            x_e = soln[10*k:10*(k+1)+1,0]
            incidence = self.sigma*x_e*self.N
            result[k] = dt*np.dot(self.weigths,incidence)
        return self.Pobs*result
        
if __name__=="__main__":
    
    # inicializa la clase
    fm = forward_mapping()
    
    def energy(q):
        # verosimilitud binomial negativa
        mu = fm.solve(q)
        omega = 1.0
        theta = 2.0/3.0
        r = mu/(omega-1.0+theta*mu)
        p = 1.0/(omega+theta*mu)
        log_likelihood = np.sum(ss.nbinom.logpmf(fm.flu_data,r,p))
        log_prior = 0.0
        # distribucion gamma para beta
        log_prior += ss.gamma.logpdf(q[0],1.0,scale=fm.beta0)
        # distribucion beta para E(0)
        log_prior += ss.beta.logpdf(q[1],2.0,5.0)        
        # distribucion beta para I(0)
        log_prior += ss.beta.logpdf(q[2],2.0,5.0)                
        # distribucion beta para R(0)
        log_prior += ss.beta.logpdf(q[3],2.0,5.0)                
        
        print(-log_likelihood - log_prior)
        return -log_likelihood - log_prior    

    def support(q):    
        """
        soporte en el espacio de parametros
        """        
        rt = True
        rt &= (0.0 < q[0] < 5.0)
        rt &= (0.0 < q[1] < 1.0)
        rt &= (0.0 < q[2] < 1.0)
        rt &= (0.0 < q[3] < 1.0)
        rt &= (q[1]+q[2]+q[3]<1.0)
        return rt
    
    def init():
        """
        muestrea parámetros uniformes en el soporte
        """
        q = np.zeros(4)
        q[0] = np.random.uniform(low=0.0,high=1.0)
        q[1] = np.random.uniform(low=0.0,high=0.1)
        q[2] = np.random.uniform(low=0.0,high=0.1)
        q[3] = np.random.uniform(low=0.0,high=0.1)
        return q
    
    # haz una cadema de Markov con el twalk
    seir = pytwalk.pytwalk(n=4,U=energy,Supp=support)
    seir.Run(T=100000,x0=init(),xp0=init())
    # guarda la cadena y la energía en un archivo
    np.savetxt('chain.txt',seir.Output)

    
#GRAFICAS DE LA POSTERIOR
#verosimilitud
seir.Ana(start=1200)

#DISTRIBUCIONES POSTERIORES MARGINALES
#Beta
beta_chain = seir.Output[1200:, 0]
plt.hist(beta_chain)
plt.axvline(x = np.mean(beta_chain), color="green")
plt.axvline(x = np.percentile(beta_chain, 97.5), color="red")
plt.axvline(x = np.percentile(beta_chain, 2.5), color="red")
print(np.mean(beta_chain))
print(np.percentile(beta_chain, 2.5))
print(np.percentile(beta_chain, 97.5))

#E(0)
E_chain = seir.Output[1200:, 1]
plt.hist(E_chain)
plt.axvline(x = np.mean(E_chain), color="green")
plt.axvline(x = np.percentile(E_chain, 97.5), color="red")
plt.axvline(x = np.percentile(E_chain, 2.5), color="red")
print(np.mean(E_chain))
print(np.percentile(E_chain, 2.5))
print(np.percentile(E_chain, 97.5))

#I(0)
I_chain = seir.Output[1200:, 2]
plt.hist(I_chain)
plt.axvline(x = np.mean(I_chain), color="green")
plt.axvline(x = np.percentile(I_chain, 97.5), color="red")
plt.axvline(x = np.percentile(I_chain, 2.5), color="red")
print(np.mean(I_chain))
print(np.percentile(I_chain, 2.5))
print(np.percentile(I_chain, 97.5))

#R(0)
R_chain = seir.Output[1200:, 3]
plt.hist(R_chain)
plt.axvline(x = np.mean(R_chain), color="green")
plt.axvline(x = np.percentile(R_chain, 97.5), color="red")
plt.axvline(x = np.percentile(R_chain, 2.5), color="red")
print(np.mean(R_chain))
print(np.percentile(R_chain, 2.5))
print(np.percentile(R_chain, 97.5))

#ESTIMACIÓN DEL R_0
mu = 1.0/(75.0*365.0)
sigma = 1.0/3.0
gamma = 1.0/5.0
R0 = (sigma/(mu + sigma))*(seir.Output[1200:, 0]/(mu+gamma))
print(np.mean(R0))
plt.hist(R0)
plt.axvline(x = np.mean(R0), color="green")
plt.axvline(x = np.percentile(R0, 97.5), color="red")
plt.axvline(x = np.percentile(R0, 2.5), color="red")
print(np.mean(R0))
print(np.percentile(R0, 2.5))
print(np.percentile(R0, 97.5))

#c_i estimado
mu = 1.0/(75.0*365.0)
sigma = 1.0/3.0
gamma = 1.0/5.0
xi = 0.0
weigths = np.ones(11)
weigths[0] = 0.5
weigths[-1] = 0.5
N = 550000.0
Pobs = 0.75
flu_time = np.linspace(0.0,62.0,63)
seir = np.loadtxt("chain.txt")
t_data = np.loadtxt("fluspa.txt")
def rhs(x,t,p):    
        """
        Lado derecho del modelo de epidemias 
        """
        fx = np.zeros(3)    
        inc = p[0]*x[1]     #cuadrar parametros que se postulan (dejarlos quietos) y los que se infieren si molestarlos
        fx[0] = (inc+xi)*(1.0-x[0]-x[1]-x[2]) - (sigma + mu)*x[0]
        fx[1] = sigma*x[0] - (gamma + mu)*x[1]
        fx[2] = gamma*x[1] - mu*x[2]  #pronostico de sistema dinamico es el objetivo
        return fx

def solve(p):
        """
        evalua el valor esperado de la verosimilitud
        """
        x0 = np.array([p[1],p[2],p[3]])
        nn = len(flu_time)
        dt = 1.0/nn
        n_quad = 10*nn+1
        t_quad = np.linspace(flu_time[0],flu_time[-1],n_quad)
        soln = integrate.odeint(rhs,x0,t_quad,args=(p,))
        result = np.zeros(nn)
        for k in range(nn):
            x_e = soln[10*k:10*(k+1)+1,0]
            incidence = sigma*x_e*N
            result[k] = dt*np.dot(weigths,incidence)
        return Pobs*result


solns = np.zeros((500,63))
for k in np.arange(500):
    solns[k,:] = solve(seir[-k,(0,1,2,3)])
    
# find and plot the median
median_soln = np.median(solns,axis=0)
# find quantiles and plot probability region
q1 = np.quantile(solns,0.05,axis=0)
q2 = np.quantile(solns,0.95,axis=0)
# plot data and results
plt.plot(t_data,'r-o',label='data')
plt.plot(np.linspace(0,63,63),median_soln,'k',label='median')
plt.fill_between(np.linspace(0,63,63),q1,q2,color='k', alpha=0.5)
