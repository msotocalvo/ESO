import numpy as np

class complexity:
    name = "Complexity Evaluation"
    iterations = 1_000_000
    optimal = 0
    bounds = (-100,100)   
        
        
    @staticmethod
    def function(x):
        for i in range(1, 2 + 1):
            x = 0.55 + float(i)
            x += x + i
            x /= 2
            x = x * x
            x = np.sqrt(x)
            x = np.log(x)
            x = np.exp(x)
            x = x / (x + 2)
        return x    
            

class ackley1:
    name = 'Ackley 1'
    optimal = 0
    bounds = (-35,35)
    type = 'multimodal'
       
    @staticmethod
    def function(x):        
        return -20 * np.exp(-0.2 * np.sqrt(sum(x_i**2 for x_i in x) / len(x))) - np.exp(sum(np.cos(2 * np.pi * x_i) for x_i in x) / len(x)) + 20 + np.e

class ackley2:
    name = 'Ackley 2'
    optimal = -200
    bounds = (-32,32)
    type = 'unimodal'
       
    @staticmethod
    def function(x):
        
        return -200 * np.exp(-0.02 * np.sqrt(x[0] ** 2 + x[1] ** 2))
    
class ackley3:
    name = 'Ackley 3'
    optimal = -219.1418
    bounds = (-32,32)
    type = 'unimodal'
       
    @staticmethod
    def function(x):
        
        return -200 * np.exp(-0.02 * np.sqrt(x[0] ** 2 + x[1] ** 2)) + 5 * np.exp(np.cos(3*x[0]) + np.sin(3*x[1]))   

class ackley4:
    name = 'Ackley 4'
    optimal = -3.917275
    bounds = (-35,35)
    type = 'multimodal'
       
    @staticmethod
    def function(x):
        n = len(x)
        return np.sum([(np.exp(-0.2*np.sqrt(x[i]**2 + x[i+1]**2)) + 3*(np.cos(2*x[i]) + np.sin(2*x[i+1])))for i in range(n - 1)])

class adjiman:
    name = 'Adjiman'
    optimal = -2.0218
    bounds = (-1,2), (-1,1)
    type = 'multimodal'
    
    @staticmethod
    def function(x):
        return np.cos(x[0]) * np.sin(x[1]) - x[0] / (x[1]**2 + 1)
    
class alpine_1:
    name = 'Alpine 1'
    optimal = 0
    bounds = (-10,10)
    type = 'multimodal'
    
    @staticmethod
    def function(x):
        return sum([np.abs(xi * np.sin(xi) + 0.1 * xi) for xi in x])
    
class alpine_2:
    name = 'Alpine 2'
    optimal = 2.808 # *D
    bounds = (-0,10)
    type = 'multimodal'
    
    @staticmethod
    def function(x):
        return np.prod(np.sqrt(x)*np.sin(x))  
    
class bartels_conn:
    name = 'Bartels Conn'
    optimal = 0
    bounds = (-500,500)
    type = 'multimodal'
    
    @staticmethod
    def function(x):        
        return  abs(x[0]**2 + x[1]**2 + x[0] * x[1]) + abs(np.sin(x[0])) + abs(np.cos(x[1]))     

class Beale:
    name = 'Beale'
    optimal = 0
    bounds = (-4.5,4.5)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return (1.5 - x[0] + x[0] * x[1])**2 + (2.25 - x[0] + x[0] * x[1]**2)**2 + (2.625 - x[0] + x[0] * x[1]**3)**2

class Bohachevsky1:
    name = 'Bohachevsky 1'
    optimal = 0
    bounds = (-100,100)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return x[0]**2 + 2*x[1]**2 - 0.3*np.cos(3*np.pi*x[0]) - 0.4*np.cos(4*np.pi*x[1]) + 0.7
        
class BentCigar:
    name = 'Bent Cigar'
    optimal = 0.0
    bounds = (-100,100)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return x[0]**2 + 10**6 * sum([xi**2 for xi in x[1:]])

class Booth:
    name = 'Booth'
    optimal = 0.00000
    bounds = (-10,10)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2
    
class Branin01:
    name = 'Branin 01'
    optimal = 0.39788735772973816
    bounds = (-5,10), (0,15)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return ((x[1] - (5.1 / (4 * np.pi ** 2)) * x[0] ** 2 + 5 * x[0] / np.pi - 6) ** 2 + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x[0]) + 10)    

class Branin02:
    name = 'Branin 02'
    optimal = 5.559037
    bounds = (-5,15)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return ((x[1] - (5.1 / (4 * np.pi ** 2)) * x[0] ** 2 + 5 * x[0] / np.pi - 6) ** 2
                + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x[0]) * np.cos(x[1]) + np.log(x[0] ** 2.0 + x[1] ** 2.0 + 1.0) + 10)

class Brent:
    name = 'Brent'
    optimal = 0.00000
    bounds = (-10,10)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return (x[0]+10)**2 + (x[1]+10)**2 + np.exp(-x[0]**2-x[1]**2)
    
class Brown:
    name = 'Brown'
    optimal = 0.00000
    bounds = (-1,4)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return np.sum((x[0] ** 2.0) ** (x[1] ** 2.0 + 1.0) + (x[1] ** 2.0) ** (x[0] ** 2.0 + 1.0))   

class Bukin2:
    name = 'Bukin 2'
    optimal = -124.75
    bounds = (-15,-5), (-3,3)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return 100 * (x[1] ** 2 - 0.01 * x[0] ** 2 + 1.0) + 0.01 * (x[0] + 10.0) ** 2.0
    
class Bukin4:
    name = 'Bukin 4'
    optimal = 0.00000
    bounds = (-15,-5), (-3,3)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return 100 * x[1] ** 2 + 0.01 * abs(x[0] + 10)   
    
class Bukin6:
    name = 'Bukin 6'
    optimal = 0.00000
    bounds = (-15,-5), (-3,3)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return 100 * np.sqrt(abs(x[1] - 0.01 * (x[0]**2))) + 0.01 * abs(x[0] + 10)

class ChungReynolds:
    name = 'Chung Reynolds'
    optimal = 0.0
    bounds = (-100,100)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return (sum(x_i**2 for x_i in x))**2

class CrownedCross:
    name = 'Crowned cross'
    optimal = 0.0001
    bounds = (-10,10)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return 0.0001 * (np.abs(np.sin(x[0]) * np.sin(x[1]) * np.exp(np.abs(100 - np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi))) + 1) ** (0.1)
        
class CrossInTray:
    name = 'CrossInTray'
    optimal = -2.0626
    bounds = (-10,10)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return -0.0001 * (np.abs(np.sin(x[0]) * np.sin(x[1]) * np.exp(np.abs(100 - np.sqrt(x[0]**2 + x[1]**2)/np.pi))) + 1)**0.1

class CrossLegTable:
    name = 'Cross Leg Table'
    optimal = -1
    bounds = (-10,10)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return -(np.abs(np.sin(x[0]) * np.sin(x[1]) * np.exp(np.abs(100 - np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi))) + 1) ** (-0.1)

class Corana:
    name = 'Corana'
    optimal = 0.00000
    bounds = (-500,500)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return sum(0.15 * (0.2 * np.floor(abs(x[i]) / 0.2) + 0.49999 * np.sign(x[i]) - 0.05 * np.sign(x[i]))**2 * [1, 1000, 10, 100][i] if abs(x[i] - (0.2 * np.floor(abs(x[i]) / 0.2) + 0.49999 * np.sign(x[i]))) < 0.05 else [1, 1000, 10, 100][i] * x[i]**2 for i in range(4))

class Cola:
    name = 'Cola'
    optimal = 11.7464
    bounds = (0,4), (-4,4)
    type = 'multimodal'

    @staticmethod
    def function(x):
        d = np.asarray([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [1.27, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [1.69, 1.43, 0, 0, 0, 0, 0, 0, 0, 0],
                 [2.04, 2.35, 2.43, 0, 0, 0, 0, 0, 0, 0],
                 [3.09, 3.18, 3.26, 2.85, 0, 0, 0, 0, 0, 0],
                 [3.20, 3.22, 3.27, 2.88, 1.55, 0, 0, 0, 0, 0],
                 [2.86, 2.56, 2.58, 2.59, 3.12, 3.06, 0, 0, 0, 0],
                 [3.17, 3.18, 3.18, 3.12, 1.31, 1.64, 3.00, 0, 0, 0],
                 [3.21, 3.18, 3.18, 3.17, 1.70, 1.36, 2.95, 1.32, 0, 0],
                 [2.38, 2.31, 2.42, 1.94, 2.85, 2.81, 2.56, 2.91, 2.97, 0.]])
        
        xi = np.atleast_2d(np.asarray([0.0, x[0]] + list(x[1::2])))
        xj = np.repeat(xi, np.size(xi, 1), axis=0)
        xi = xi.T

        yi = np.atleast_2d(np.asarray([0.0, 0.0] + list(x[2::2])))
        yj = np.repeat(yi, np.size(yi, 1), axis=0)
        yi = yi.T
        inner = (np.sqrt(((xi - xj) ** 2 + (yi - yj) ** 2)) - d) ** 2
        inner = np.tril(inner, -1)
        return np.sum(np.sum(inner, axis=1))
    
class CosineMixture:
    name = 'Cosine Mixture'
    optimal = 0
    bounds = (-1,1)
    type = 'multimodal'

    @staticmethod
    def function(x):
        n = len(x)
        return -0.1 * sum([np.cos(5*np.pi*x[i]) for i in range(n)]) - sum([x[i]**2 for i in range(n)])

class Csendes:
    name = 'Csendes'
    optimal = 0.00000
    bounds = (-1,1)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return sum([xi**6 * (2 + np.sin(1/xi)) for xi in x])
        
class Cube:
    name = 'Cube'
    optimal = 0.0
    bounds = (-10,10)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return 100 * (x[1] - x[0]**3)**2 + (1 - x[0])**2

class Cone:
    name = 'Cone'
    optimal = 0.0
    bounds = (-100,100)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return np.sqrt(np.sum(np.array(x)**2))
        
class Damavandi:
    name = 'Damavandi'
    optimal = 0
    bounds = (0,14)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return (1 - (np.abs((np.sin(np.pi * (x[0] - 2)) * np.sin(np.pi * (x[1] - 2))) / (np.pi**2 * (x[0] - 2) * (x[1] - 2))))**5) * (2 + (x[0]-7)**2 + 2*(x[1]-7)**2)

class DeVilliersGlasser01:
    name = 'De Villiers Glasser 01'
    optimal = 0
    bounds = (-500,500)
    type = 'multimodal'
        
    @staticmethod
    def function(x):
        n = 24
        return np.sum((x[0] * x[1]**(0.1 * (i - 1)) * np.sin(x[2] * (0.1 * (i - 1)) + x[3]) - 60.137*(1.371)**(0.1 * (i - 1)) * np.sin(3.112*(0.1 * (i - 1)) + 1.761)) ** 2 for i in range(1, n + 1))

class DeVilliersGlasser02:
    name = 'De Villiers Glasser 02'
    optimal = 0
    bounds = (-10,10)
    type = 'multimodal'
        
    @staticmethod
    def function(x):
        n = 24
        return np.sum((x[0] * x[1]**(0.1 * (i - 1)) * np.tanh(x[2] * (0.1 * (i - 1)) + np.sin(x[3] * (0.1 * (i - 1)))) * np.cos((0.1 * (i - 1)) * np.exp(x[4] * (0.1 * (i - 1)))) - (53.81 * np.tanh(3.01 * (0.1 * (i - 1)) + np.sin(2.13 * (0.1 * (i - 1)))) * np.cos( np.exp(0.507) * (0.1 * (i - 1))))) ** 2 for i in range(1, n + 1))

class Deceptive:
    name = 'Deceptive'
    optimal = -1
    bounds = (0,1)
    type = 'multimodal'

    @staticmethod
    def function(x):
        n = len(x)
        alpha = np.arange(1.0, n + 1.0) / (n + 1.0)
        beta = 2.0
        g = np.zeros((n,))
        for i in range(n):
            if x[i] <= 0.0:
                g[i] = x[i]
            elif x[i] < 0.8 * alpha[i]:
                g[i] = -x[i] / alpha[i] + 0.8
            elif x[i] < alpha[i]:
                g[i] = 5.0 * x[i] / alpha[i] - 4.0
            elif x[i] < (1.0 + 4 * alpha[i]) / 5.0:
                g[i] = 5.0 * (x[i] - alpha[i]) / (alpha[i] - 1.0) + 1.0
            elif x[i] <= 1.0:
                g[i] = (x[i] - 1.0) / (1.0 - alpha[i]) + 4.0 / 5.0
            else:
                g[i] = x[i] - 1.0
        return -((1.0 / n) * np.sum(g)) ** beta

class Discus:
    name = 'Discus'
    optimal = 0
    bounds = (-100,100)
    type = 'Unimodal'

    @staticmethod
    def function(x):
        n = len(x)
        return (x[0] - 1)**2 + np.sum([i*(2*x[i]**2 - x[i-1])**2 for i in range (n-1)])


class Dolan:
    name = 'Dolan'
    optimal = 0
    bounds = (-100,100)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return (abs((x[0] + 1.7 * x[1]) * np.sin(x[0]) - 1.5 * x[2] - 0.1 * x[3] * np.cos(x[3] + x[4] - x[0]) + 0.2 * x[4] ** 2 - x[1] - 1))

class DropWave:
    name = 'Drop Wave'
    optimal = -1
    bounds = (-5.12,5.12)
    type = 'multimodal'

    @staticmethod
    def function(x):
        norm_x = np.sum(x ** 2)
        return -(1 + np.cos(12 * np.sqrt(norm_x))) / (0.5 * norm_x + 2)

class DixonPrice:
    name = 'Dixon Price'
    optimal = 0
    bounds = (-10,10)
    type = 'unimodal'

    @staticmethod
    def function(x):
        i = np.arange(2, len(x) + 1)
        s = i * (2.0 * x[1:] ** 2.0 - x[:-1]) ** 2.0
        return np.sum(s) + (x[0] - 1.0) ** 2.0

class Easom:
    name = 'Easom'
    optimal = -1.00000
    bounds = (-100,100)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-((x[0] - np.pi)**2 + (x[1] - np.pi)**2))    

class EggCrate:
    name = 'Egg Crate'
    optimal = 0.00000
    bounds = (-5,5)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return (x[0]**2 + x[1]**2 + 25) * (np.sin(x[0])**2 + np.sin(x[1])**2)
    
class EggHolder:
    name = 'Egg Holder'
    optimal = -959.640662711
    bounds = (-512,512)
    type = 'multimodal'

    @staticmethod
    def function(x):
        vec = (-(x[1:] + 47) * np.sin(np.sqrt(abs(x[1:] + x[:-1] / 2. + 47))) - x[:-1] * np.sin(np.sqrt(np.abs(x[:-1] - (x[1:] + 47)))))
        return np.sum(vec)    

class Ellipse:
    name = 'Ellipse'
    optimal = 0.0
    bounds = (-10,10)
    type = 'unimodal'
    
    @staticmethod
    def function(x):
        n = len(x)
        return sum((10**((i)/(n-1)) * x[i])**2 for i in range(len(x)))
    
class Exponential:
    name = 'Exponential'
    optimal = 0.0
    bounds = (-1,1)
    type = 'unimodal'
    
    @staticmethod
    def function(x):
        
        return -np.exp(0-.5*np.sum(x**2))
   
class ElAttarVidyasagarDutta:
    name = 'El Attar Vidyasagar Dutta'
    optimal = 1.712780354
    bounds = (-500,500)
    type = 'unimodal'

    @staticmethod
    def function(x):
        
        return ((x[0] ** 2 + x[1] - 10) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2 + (x[0] ** 2 + x[1] ** 3 - 1) ** 2)    
    
class Griewank:
    name = 'Griewank'
    optimal = 0.00000
    bounds = (-600,600)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return np.sum(x ** 2 / 4000) - np.prod(np.cos(x / np.sqrt(np.arange(1., np.size(x) + 1.)))) + 1
    
class GoldsteinPrice:
    name = 'Goldstein Price'
    optimal = 3
    bounds = (-2,2)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return (1 + (x[0] + x[1] + 1)**2 * (19 - 14*x[0] + 3*x[0]**2 - 14*x[1] + 6*x[0]*x[1] + 3*x[1]**2)) * (30 + (2*x[0] - 3*x[1])**2 * (18 - 32*x[0] + 12*x[0]**2 + 48*x[1] - 36*x[0]*x[1] + 27*x[1]**2))   
    
class Gulf:
    name = 'Gulf'
    optimal = 0
    bounds = (0,60)
    type = 'multimodal'

    @staticmethod
    def function(x):
        m = 99
        i = np.arange(1., m + 1)
        u = 25 + (-50 * np.log(i / 100.)) ** (2 / 3.)
        vec = (np.exp(-((np.abs(u - x[1])) ** x[2] / x[0])) - i / 100.)
        return np.sum(vec ** 2)    

class HappyCat:
    name = 'Happy Cat'
    optimal = 0
    bounds = (-2,2)
    type = 'multimodal'

    @staticmethod
    def function(x):
        alpha = 1.0/8
        return ((np.sum(x**2) - len(x))**2)**alpha + (0.5*np.sum(x**2)+np.sum(x))/len(x) + 0.5
    
class HighConditionedElliptic:
    name = 'High Conditioned Elliptic'
    optimal = 0
    bounds = (-100,100)
    type = 'unimodal'

    @staticmethod
    def function(x):
        D = len(x)  # Dimensionalidad de la entrada
        return np.sum([(10**6)**((i - 1)/(D - 1)) * (xi**2) for i, xi in enumerate(x, start=1)])    
    
class HgBat:
    name = 'HgBat'
    optimal = 0
    bounds = (-100,100)
    type = 'multimodal'

    @staticmethod
    def function(x):
        sum_squares = np.sum(x**2)
        sum_fourth_powers = np.sum(x**4)
        D = len(x)  # Dimensionalidad de la entrada
        term1 = (sum_fourth_powers - sum_squares**2)**0.5
        term2 = (0.5 * sum_squares + np.sum(x)) / D
        return term1 + term2 + 0.5  
    
class HimmelBlau:
    name = 'HimmelBlau'
    optimal = 0
    bounds = (-6,6)
    type = 'multimodal'

    @staticmethod
    def function(x):
        
        return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2    

class HolderTable:
    name = 'Holder Table'
    optimal = -19.208
    bounds = (-10,10)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return -np.abs(np.sin(x[0]) * np.cos(x[1]) * np.exp(np.abs(1 - np.sqrt(x[0]**2 + x[1]**2)/np.pi)))

class Kowalik:
    name = 'Kowalik'
    optimal = 0.00030748610
    bounds = (-5,5)
    type = 'multimodal'

    @staticmethod
    def function(x):
        a = np.asarray([4.0, 2.0, 1.0, 1 / 2.0, 1 / 4.0, 1 / 6.0, 1 / 8.0,
                          1 / 10.0, 1 / 12.0, 1 / 14.0, 1 / 16.0])
        b = np.asarray([0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627,
                          0.0456, 0.0342, 0.0323, 0.0235, 0.0246])
        vec = b - (x[0] * (a ** 2 + a * x[1]) / (a ** 2 + a * x[2] + x[3]))
        return np.sum(vec ** 2)
    
class Katsuura:
    name = 'Katsuura'
    optimal = 1
    bounds = (0,100)
    type = 'multimodal'

    @staticmethod
    def function(x):
        d = 32
        n = len(x)
        k = np.atleast_2d(np.arange(1, d + 1)).T
        idx = np.arange(0., n * 1.)
        inner = np.round(2 ** k * x) * (2. ** (-k))
        return np.prod(np.sum(inner, axis=0) * (idx + 1) + 1)
    
class Langermann:
    name = 'Langermann'
    optimal = -5.1621259
    bounds = (-1.2,1.2)
    type = 'multimodal'

    @staticmethod
    def function(x):
        a = np.array([3, 5, 2, 1, 7])
        b = np.array([5, 2, 1, 4, 9])
        c = np.array([1, 2, 5, 2, 3])
        return (-np.sum(c * np.exp(-(1 / np.pi) * ((x[0] - a) ** 2 +
                (x[1] - b) ** 2)) * np.cos(np.pi * ((x[0] - a) ** 2 + (x[1] - b) ** 2)))) 
    
class Leon:
    name = 'Leon'
    optimal = 0.00000
    bounds = (-1.2,1.2)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2
    
class LennardJones:
    name = 'Lennard Jones'
    optimal = [-1.0, -3.0, -6.0, -9.103852, -12.712062,
                       -16.505384, -19.821489, -24.113360, -28.422532,
                       -32.765970, -37.967600, -44.326801, -47.845157,
                       -52.322627, -56.815742, -61.317995, -66.530949,
                       -72.659782, -77.1777043]
    bounds = (-4,4)
    type = 'multimodal'

    @staticmethod
    def function(x):
        n = len(x)
        k = int(n / 3)
        s = 0.0
        for i in range(k - 1):
            for j in range(i + 1, k):
                a = 3 * i
                b = 3 * j
                xd = x[a] - x[b]
                yd = x[a + 1] - x[b + 1]
                zd = x[a + 2] - x[b + 2]
                ed = xd * xd + yd * yd + zd * zd
                ud = ed * ed * ed
                if ed > 0.0:
                    s += (1.0 / ud - 2.0) / ud
        return s    

class Levy13:
    name = 'Levy 13'
    optimal = 0.00000
    bounds = (-10,10)
    type = 'multimodal'

    @staticmethod
    def function(x):
         return (x[0] - 1)**2 * np.sin(3*np.pi*x[1])**2 + (x[1] - 1)**2 * (1 + np.sin(2*np.pi*x[1])**2) +  (np.sin(3*np.pi*x[0])**2)


class Matyas:
    name = 'Matyas'
    optimal = 0.00000
    bounds = (-10,10)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]
    
class Michaelwicz:
    name = 'Michaelwicz'
    optimal = -1.8013
    bounds = (0,np.pi)
    type = 'multimodal'

    @staticmethod
    def function(x):
        m = 10
        return -sum([np.sin(xi) * np.sin((i+1) * xi**2 / np.pi)**(2*m) for i, xi in enumerate(x)])  

class Mishra01:
    name = 'Mishra 01'
    optimal = 2
    bounds = (0,1)
    type = 'multimodal'

    @staticmethod
    def function(x):
        n = len(x)
        xn = n - np.sum((x[:-1] + x[1:]) / 2.0)
        return (1 + xn) ** xn     
    
class Mishra02:
    name = 'Mishra 02'
    optimal = 2
    bounds = (0,1)
    type = 'multimodal'

    @staticmethod
    def function(x):
        n = len(x)
        xn = n - - np.sum(x[0:-1])
        return (1 + xn) ** xn     
    
class Mishra03:
    name = 'Mishra 03'
    optimal = -0.19990562
    bounds = (-10,10)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return 0.01 * (x[0] + x[1]) + np.sqrt(np.abs(np.cos(np.sqrt(np.abs(x[0] ** 2 + x[1] ** 2)))))    
    
class Mishra04:
    name = 'Mishra 04'
    optimal = -0.17767
    bounds = (-10,10)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return 0.01 * (x[0] + x[1]) + np.sqrt(np.abs(np.sin(np.sqrt(abs(x[0] ** 2 + x[1] ** 2)))))  
    
class NewFuntion01:
    name = 'New Funtion 01'
    optimal = -0.18459899925
    bounds = (-10,10)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return ((np.abs(np.cos(np.sqrt(np.abs(x[0] ** 2 + x[1]))))) ** 0.5 + 0.01 * (x[0] + x[1]))  
    
class NewFuntion02:
    name = 'New Funtion 02'
    optimal = -0.19933159253
    bounds = (-10,10)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return ((np.abs(np.sin(np.sqrt(np.abs(x[0] ** 2 + x[1]))))) ** 0.5 + 0.01 * (x[0] + x[1]))   
    
class OddSquare:
    name = 'Odd Square'
    optimal = -1.0084
    bounds = (-5*np.pi,5*np.pi)
    type = 'multimodal'

    @staticmethod
    
    def function( x):
        a = np.array([1, 1.3, 0.8, -0.4, -1.3, 1.6, -0.2, -0.6, 0.5, 1.4, 1, 1.3, 0.8, -0.4, -1.3, 1.6, -0.2, -0.6, 0.5, 1.4])
        b = a[:19]
        d = 19 * np.max((x - b)**2)
        h = np.sum((x - b)**2)
        return -np.exp(-d/(2.0*np.pi))*np.cos(np.pi*d)*(1.0 + 0.02*h/(d + 0.01))          
    
class PowellSingular1:
    name = 'Powell Singular 1'
    optimal = 0.00000
    bounds = (-4.5,5)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return sum((x[4*i-4] + 10*x[4*i-3])**2 + 5*(x[4*i-2] - x[4*i-1])**2 + (x[4*i-3] - 2*x[4*i-2])**4 + 10*(x[4*i-4] - x[4*i-1])**4 for i in range(1, len(x)//4 + 1))

class PowellSingular2:
    name = 'Powell Singular 2'
    optimal = 0.00000
    bounds = (-4,5)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return sum((x[i-1]+10*x[i])**2 + 5*(x[i+1]-x[i+2])**2 + (x[i]-2*x[i+1])**4 + 10 * (x[i-1]-x[i+2])**4 for i in range(len(x)-2))

class PowellSum:
    name = 'Powell Sum'
    optimal = 0.00000
    bounds = (-1,1)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return sum(abs(x_i)**(i+1) for i, x_i in enumerate(x))

class Price1:
    name = 'Price 1'
    optimal = 0
    bounds = (-500,500)
    type = 'multimodal'
    
    @staticmethod
    def function(x):
        return (np.abs(x[0]) - 5)**2 + (np.abs(x[1]) - 5)**2
    
class Price2:
    name = 'Price 2'
    optimal = 0.9
    bounds = (-500,500)
    type = 'multimodal'
    
    @staticmethod
    def function(x):
        return 1 + np.sin(x[0])**2 + np.sin(x[1])**2 - 0.1*np.exp(-x[0]**2 - x[1]**2)    

class Quadric:
    name = 'Unimodal'
    optimal = 0.0
    bounds = (-1.28,1.28)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return sum((sum(x[j] for j in range(i))**2) for i in range(len(x)))
           
class Rastrigin:
    name = 'Rastrigin'
    optimal = 0.00000
    bounds = (-5,5)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return sum(x_i**2 - 10 * np.cos(2 * np.pi * x_i) + 10 for x_i in x)
    
class Ridge:
    name = 'Ridge'
    optimal = 0.00000
    bounds = (-5,5)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return x[0] + 2*np.sum(x[1:]**2)**0.5    
    
class Ripple01:
    name = 'Ripple 01'
    optimal = -2.2
    bounds = (0,1)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return  sum(-np.exp(-2 * np.log(2) * ((x[i]/0.8)**2)) * (np.sin(5 * np.pi * x[i])**6 + 0.1 * np.cos(500 * np.pi * x[i])**2) for i in range(2))   

class Rosenbrock:
    name = 'Rosenbrock'
    optimal = 0.0
    bounds = (-30,30)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return sum(100 * (x[i]**2 - x[i+1])**2 + (x[i] - 1)**2 for i in range(len(x) - 1))
    
class RosenbrockModified:
    name = 'Rosenbrock Modified'
    optimal = 34.37
    bounds = (-2,2)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return 74 + 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2 - 400 * np.exp(-((x[0] + 1)**2 + (x[1] + 1)**2) / 0.1)
    
class RotatedEllipse:
    name = 'Rotated Ellipse'
    optimal = 0.0
    bounds = (-500,500)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return 7*x[0]**2 - 6*np.sqrt(3)*x[0]*x[1] + 13*x[1]**2
    
class Rump:
    name = 'Rump'
    optimal = 0.0
    bounds = (-500,500)
    type = 'unimodal'

    @staticmethod
    def function(x):
        term1 = (333.75 - x[0]**2)*x[1]**6 + x[0]**2 * (11*x[0]**2 * x[1]**2 - 121*x[1]**4 - 2)
        term2 =5.5*x[1]**8 + (x[0]/(2*x[1]))

        # Sum the terms to get the result.
        return term1 + term2    

class Salomon:
    name = 'Salomon'
    optimal = 0.00000
    bounds = (-100,100)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return 1 - np.cos(2 * np.pi * np.sqrt(np.sum([xi**2 for xi in x]))) + 0.1 * np.sqrt(np.sum([xi**2 for xi in x]))

class Schaffer:
    name = 'Schaffer'
    optimal = 0.00000
    bounds = (-100,100)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return 0.5 + (np.sin(x[0]**2 - x[1]**2)**2 - 0.5) / (1 + 0.001*np.sum(x**2))**2

class Schaffer2:
    name = 'Schaffer 2'
    optimal = 0.00000
    bounds = (-100,100)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return 0.5 + (np.sin(x[0]**2 - x[1]**2)**2 - 0.5) / (1 + 0.001*np.sum(x**2))**2
    
class Schaffer3:
    name = 'Schaffer 3'
    optimal = 0.00156685
    bounds = (-100,100)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return 0.5 + (np.sin(np.cos(np.abs( x[0]**2 - x[1]**2 ))) - 0.5) / (1 + 0.001*np.sum(x**2))**2
    
class Schaffer4:
    name = 'Schaffer 4'
    optimal = 0.292579
    bounds = (-100,100)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return 0.5 + (np.cos(np.sin(np.abs( x[0]**2 - x[1]**2 ))) - 0.5) / (1 + 0.001*np.sum(x**2))**2

class Schwefel01:
    name = 'Schwefel 01'
    optimal = 0.00000
    bounds = (-100,100)
    type = 'unimodal'
    
    @staticmethod
    def function(x):
        return sum(x_i**2 for x_i in x)**2
    
class Schwefel02:
    name = 'Schwefel 02'
    optimal = 0.00000
    bounds = (-100,100)
    type = 'unimodal'    

    @staticmethod
    def function(x):
        total_sum = 0
        for i in range(len(x)):
            inner_sum = sum(x[j] for j in range(i + 1))
            total_sum += inner_sum ** 2
        return total_sum    

   
class Schwefel20:
    name = 'Schwefel 20'
    optimal = 0.0
    bounds = (-100,100)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return np.sum(abs(x))

class Schwefel21:
    name = 'Schwefel 21'
    optimal = 0
    bounds = (-100,100)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return max(abs(x))

class Schwefel22:
    name = 'Schwefel 22'
    optimal = 0.0
    bounds = (-100,100)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return np.sum(abs(x)) + np.prod(abs(x))
    
class Schwefel23:
    name = 'Schwefel 23'
    optimal = 0.0
    bounds = (-100,100)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return np.sum(x**10)    

class Sphere:
    name = 'Sphere'
    optimal = 0
    bounds = (0,10)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return np.sum(np.square(x))

class SineEnvelop:
    name = 'Sine Envelop'
    optimal = 0
    bounds = (-20,20)
    type = 'multimodal'

    @staticmethod
    def function(x):
        # term1 =  np.sin(np.sqrt(x[i+1]**2 + x[i]**2) - 0.5)**2
        # term2 = (0.001*(x[i+1]**2 + x[i]**2) + 1)**2
        return - np.sum((np.sin(np.sqrt(x[i+1]**2 + x[i]**2) - 0.5)**2) / ((0.001*(x[i+1]**2 + x[i]**2) + 1)**2) + 0.5 for i in range(len(x) - 1))

class Stochastic:
    name = 'Stochastic'
    optimal = 0.00000
    bounds = (-5,5)
    type = 'multimodal'

    @staticmethod
    def function(x):
        epsilon = np.random.uniform(0, 1, size=len(x))
        return np.sum(epsilon * np.abs(x - 1 / (np.arange(1, len(x) + 1))))


class Step1:
    name = 'Step 1'
    optimal = 0.00000
    bounds = (-100,100)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return sum([np.floor(abs(xi)) for xi in x])

class Step2:
    name = 'Step 2'
    optimal = 0.00000
    bounds = (-100,100)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return np.sum(np.floor(x + 0.5)**2)

class Step3:
    name = 'Step 3'
    optimal = 0.00000
    bounds = (-100,100)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return np.sum(np.floor(x**2))
    
class Stepint:
    name = 'Stepint'
    optimal = 0.00000
    bounds = (-5.12,5.12)
    type = 'unimodal'

    @staticmethod
    def function(x):
        n = len(x)
        return  np.sum(np.abs(x[i]) for i in range(n))    

class StretchedVSineWave:
    name = 'Stretched V Sine Wave'
    optimal = 0
    bounds = (-10,10)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return sum(((x[i]**2 + x[i+1]**2)**0.25) * (np.sin(50 * (x[i]**2 + x[i+1]**2)**0.1)**2 + 0.1) for i in range(len(x) - 1))
    
class SumSquares:
    name = 'Sum squares'
    optimal = 0.00000
    bounds = (-10,10)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return sum((i+1)*x[i]**2 for i in range(len(x)))

class Trefethen:
    name = 'Trefethen'
    optimal = -3.3068
    bounds = (-10,10)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return np.exp(np.sin(50 * x[0])) + np.sin(60 * np.exp(x[1])) + np.sin(70 * np.sin(x[0])) + np.sin(np.sin(80 * x[1])) - np.sin(10 * (x[0] + x[1])) + (x[0]**2 + x[1]**2) / 4

class Trid06:
    name = 'Trid 06'
    optimal = -50
    bounds = (-20,20)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return sum([(x[i] - 1)**2 for i in range(len(x))]) - sum([x[i] * x[i - 1]  for i in range(len(x)-1)])
    
class Tripod:
    name = 'Tripod'
    optimal = 0
    bounds = (-100,100)
    type = 'multimodal'

    @staticmethod
    
    def p(x):
        return 1 if x >= 0 else 0
    
    def function(x):
        return Tripod.p (x[1]) * (1 + Tripod.p(x[0])) + abs(x[0] + 50*Tripod.p(x[1]*(1-2*Tripod.p(x[0]))))  + abs(x[1] - 50*(1 - 2* Tripod.p (x[1])))    
    
class Ursem1:
    name = 'Ursem 1'
    optimal = -4.81681406371
    bounds = (-2.5,3), (-2,2)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return -np.sin(2 * x[0] - 0.5 * np.pi) - 3.0 * np.cos(x[1]) - 0.5 * x[0]

class Ursem3:
    name = 'Ursem 3'
    optimal = -1.285
    bounds = (-2,2), (-1.5,1.5)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return -0.9*x[0]**2 + (x[1]**2 - 4.5*x[1]**2)*x[1]*x[0] + 4.7*np.cos(2*x[0] - x[1]**2*( 2 + x[0]))*np.sin(2.5*np.pi*x[0])
    
class UrsemWaves:
    name = 'Ursem Waves'
    optimal = -8.5536
    bounds = (-0.9,12), (-1.2,1.2)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return -np.sin(2.2*np.pi*x[0] + 0.5) * ((2 - abs(x[0])) / 2)**3 - np.sin(0.5*np.pi*x[1]**2 + 0.5) * ((2 - abs(x[1])) / 2)**3    

class Weierstrass:
    name = 'Weierstrass'
    optimal = 0.00000
    bounds = (-0.5, 0.5)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return np.sum([np.sum([0.5**(i+1) * np.cos(2 * np.pi * 3**i * (xi + 0.5)) for i in range(20)]) for xi in x]) - len(x) * np.sum([np.sum([0.5**(i+1) * np.cos(2 * np.pi * 3**i * 0.5) for i in range(20)])])

class Watson:
    name = 'Watson'
    optimal = 0.002288
    bounds = (-5,5)
    type = 'unimodal'

    @staticmethod
    def function(x):
        total_sum = 0
        for i in range(30):  # i goes from 0 to 29
            a_i = i / 29.0
            sum1 = 0
            sum2 = 0
            
            for j in range(1, 6):  # Assuming x has at least 6 elements
                sum1 += j * x[j] * (a_i ** (j - 1))
                sum2 += x[j] * (a_i ** j)
            
            total_sum += (sum1 - sum2**2 - 1)**2
            
        total_sum += x[0]**2
        return total_sum
    
class WayburnSeader1:
    name = 'Wayburn Seader 1'
    optimal = 0.0
    bounds = (-5,5)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return (x[0]**6 + x[1]**4 - 17)**2 + (2*x[0] + x[1] - 4)**2   
    
class WayburnSeader2:
    name = 'Wayburn Seader 2'
    optimal = 0.0
    bounds = (-50,50)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return (1.613 - 4*(x[0] - 0.3125)**2 - 4*(x[1] - 1.625)**2)**2 + (x[1] - 1)**2
    
class Whitley:
    name = 'Whitley'
    optimal = 0.0
    bounds = (-10.24,10.24)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return sum(sum(((100 * (x[i]**2 - x[j])**2 + (1 - x[j])**2)**2 / 4000 - np.cos(100 * (x[i]**2 - x[j])**2 + (1 - x[j])**2) + 1) for j in range(len(x))) for i in range(len(x)))   
    
class XinSheYang1:
    name = 'Xin-She Yang 1'
    optimal = 0
    bounds = (-5,5)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return sum(np.random.uniform(0, 1) * abs(x[i])**i for i in range(len(x)))

class XinSheYang2:
    name = 'Xin-She Yang 2'
    optimal = 0
    bounds = (-2*np.pi,2*np.pi)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return np.sum(np.abs(x[i]) for i in range(len(x))) / np.exp(sum(np.sin(x[i]**2) for i in range(len(x))))

class XinSheYang3:
    name = 'Xin-She Yang 3'
    optimal = -1
    bounds = (-20,20)
    type = 'multimodal'

    @staticmethod
    def function(x):
        return np.exp(-np.sum((x / 15.0)**(2.0 * 5))) - 2.0 * np.exp(-np.sum(x**2)) * np.prod(np.cos(x)**2)
    
class XinSheYang4:
    name = 'Xin-She Yang 4'
    optimal = -1
    bounds = (-10,10)
    type = 'multimodal'

    @staticmethod
    def function(x):
        t1 = np.sum(np.sin(x)**2)
        t2 = -np.exp(-np.sum(x**2))
        t3 = -np.exp(np.sum(np.sin(np.sqrt(np.abs(x)))**2))
        return (t1 + t2) * t3    

class Zackarov:
    name = 'Zackarov'
    optimal = 0
    bounds = (-5,10)
    type = 'multimodal'

    @staticmethod
    def function(x):
        n = len(x)
        return np.sum(np.square(x)) + (np.sum(0.5 * np.arange(1, n+1) * x))**2 + (np.sum(0.5 * np.arange(1, n+1) * x))**4 
    
class ZeroSum:
    name = 'Zero Sum'
    optimal = 0
    bounds = (-10,10)
    type = 'multimodal'

    @staticmethod
    def function(x):
        if np.abs(np.sum(x)) < 3e-16:
            return 0.0
        return 1.0 + (10000.0 * np.abs(np.sum(x))) ** 0.5  
    
class Zirilli:
    name = 'Zirilli'
    optimal = -0.3523
    bounds = (-10,10)
    type = 'unimodal'

    @staticmethod
    def function(x):
        return 0.25*x[0]**4 - 0.5*x[0]**2 + 0.1*x[0] +0.5*x[1]**2  
    
class Zimmerman:
    name = 'Zimmerman'
    optimal = 0
    bounds = (0,100)
    type = 'multimodal'

    @staticmethod
    def function(x):
        Zh1 = lambda x: 9.0 - x[0] - x[1]
        Zh2 = lambda x: (x[0] - 3.0) ** 2.0 + (x[1] - 2.0) ** 2.0 - 16.0
        Zh3 = lambda x: x[0] * x[1] - 14.0
        Zp = lambda x: 100.0 * (1.0 + x)
        return max(Zh1(x), Zp(Zh2(x)) * np.sign(Zh2(x)), Zp(Zh3(x)) * np.sign(Zh3(x)), Zp(-x[0]) * np.sign(x[0]),Zp(-x[1]) * np.sign(x[1]))      




























        
        