import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from numpy.core.fromnumeric import trace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.mixture import GaussianMixture

class GeometricMethod():
    def __init__(self,  distance = None, threshold = None, discriminator_type = 'EM'):
        self._means = []
        self._std = []
        self._threshold = threshold
        self.discriminator_type = discriminator_type
        if distance is None:
            self.distance0 = distance
        elif any(isinstance(el, list) for el in distance):
            if len(distance) == 1:
                self.distance0 = distance[0]
                self.distance1 =  distance[0]
            elif len(distance)==2:
                self.distance0 = distance[0]
                self.distance1=distance[1]
            else:
                ValueError(' distance must be a list or array of float')
        elif isinstance(distance, list):
            self.distance0 = distance
            self.distance1=distance
        else:
            raise ValueError(' distance must be a list or array of float')
        
    def fit(self, y_data, x_data):
        state0=[value for index,value in enumerate(y_data) if x_data[index]==0]
        state1=[value for index,value in enumerate(y_data) if x_data[index]==1]
        self._means= np.array([np.mean(state0, axis=0), np.mean(state1, axis=0)])
        self._std= np.array([np.std(state0, axis=0), np.std(state1, axis=0)])
    
    def predict(self, y_data, return_proba = False):
        x_data=[]
        if self.distance0 is None:
            self.distance0 = 2 * self._std[0]
            self.distance1 = 2 * self._std[1]
        for value in y_data:
            a = value[0]
            b = value[1]
            if self.discriminator_type == 'EM':
                condition0 = (a-self._means[0][0])**2/(self.distance0[0])**2 + (b-self._means[0][1])**2/(self.distance0[1])**2 < 1
                condition1 = (a-self._means[1][0])**2/(self.distance1[0])**2 + (b-self._means[1][1])**2/(self.distance1[1])**2 < 1
            elif self.discriminator_type == 'RM':
                condition0 = int(np.abs(a-self._means[0][0]) < self.distance0[0] and np.abs(b-self._means[0][1]) < self.distance0[1])
                condition1 = int(np.abs(a-self._means[1][0]) < self.distance1[0] and np.abs(b-self._means[1][1]) < self.distance1[1])
            if condition0 :
                if return_proba:
                    x_data.append([1,0])
                else:
                    x_data.append(0)
            elif condition1:
                if return_proba:
                    x_data.append([0,1])
                else:
                    x_data.append(1)
            else:
                total_distance = np.linalg.norm(value-self._means[0]) + np.linalg.norm(value-self._means[1])
                if return_proba:
                    if np.linalg.norm(value-self._means[0]) <= np.linalg.norm(value-self._means[1]):
                        distance0 = np.linalg.norm(value-self._means[0])
                        x_data.append(np.array([1-distance0/total_distance,distance0/total_distance]))
                    else:
                        distance1 = np.linalg.norm(value-self._means[1])
                        x_data.append(np.array([distance1/total_distance,1-distance1/total_distance]))
                else:
                    x_data.append(2)
        return np.array(x_data)
    
    def plot(self, y_data, x_data, ax):
        x_data_pred=self.predict(y_data, return_proba=False)
        good_pred0 = np.all([x_data_pred == x_data,x_data_pred == [0]*len(x_data_pred)], axis=0)
        good_pred1 = np.all([x_data_pred == x_data ,x_data_pred == [1]*len(x_data_pred)], axis=0)
        mixed_pred = [2]*len(x_data_pred) == x_data_pred
        bad_pred = np.all([x_data_pred != x_data, mixed_pred == [False]*len(mixed_pred)], axis=0)
        ax.scatter(y_data[good_pred0,0],y_data[good_pred0,1], color='indigo', label='0')
        ax.scatter(y_data[good_pred1,0],y_data[good_pred1,1], color='green', label='1')
        ax.scatter(y_data[mixed_pred,0],y_data[mixed_pred,1], color='yellow', label='Mixed state')
        ax.scatter(y_data[bad_pred,0],y_data[bad_pred,1], color='red', marker = 'x')
        if self.discriminator_type == 'EM':
            # ellipse zone
            zone0 = patches.Ellipse(self._means[0], 2*self.distance0[0], 2*self.distance0[1], fill = True, color="indigo")
            zone1 = patches.Ellipse(self._means[1], 2*self.distance1[0], 2*self.distance1[1], fill = True, color="green")
            ax.set_title('Ellipse method')
        elif self.discriminator_type == 'RM':
            print(self._means[0] - self.distance0)
            zone0 = patches.Rectangle(self._means[0] - self.distance0, 2*self.distance0[0], 2*self.distance0[1], fill = True, color="indigo")
            zone1 = patches.Rectangle(self._means[1] - self.distance1, 2*self.distance1[0] , 2*self.distance1[1], fill = True, color="green")
            ax.set_title('Rectangle method')
        zone0.set_clip_box(ax.bbox)
        zone0.set_alpha(0.5)
        ax.add_artist(zone0)
        zone1.set_clip_box(ax.bbox)
        zone1.set_alpha(0.5)
        ax.add_artist(zone1)
        ax.set_xlabel('I data')
        ax.set_ylabel('Q data')
        ax.legend()


    def score(self, y_data, x_data):
        x_data_pred= np.array(self.predict(y_data))
        good_pred = np.count_nonzero(x_data_pred == [x_data])/len(x_data)
        mixed_states = np.count_nonzero(x_data_pred == [2]*len(x_data_pred))/len(x_data)
        return {'good_predictions': good_pred, 'mixed_states': mixed_states}

class GeometricMethodSklearn(GeometricMethod):
    def __init__(self,  threshold = None, discriminator_type = 'LDA'):
        self._threshold = threshold
        self.discriminator_type = discriminator_type
        if discriminator_type == 'LDA':
            self._discriminator = LinearDiscriminantAnalysis()
        elif discriminator_type == 'QDA':
            self._discriminator = QuadraticDiscriminantAnalysis()
        else:
            ValueError('Wrong type of discriminator')
    
    def fit(self, y_data, x_data):
        self._discriminator.fit(y_data,x_data)

    def predict(self, y_data, return_proba = False):
        if self._threshold is None:
            if return_proba:
                return self._discriminator.predict_proba(y_data)
            else:
                return self._discriminator.predict(y_data)
        if return_proba:
            x_data = np.zeros((len(y_data),2))
        else:
            x_data = np.zeros(len(y_data))
        for i in range(len(y_data)):
            proba = self._discriminator.predict_proba([y_data[i]])[0]
            if return_proba:
                if proba[0] > 1 - self._threshold[0]:
                    x_data[i] = np.array([1,0])
                elif proba[1] > 1 - self._threshold[1]:
                    x_data[i] = np.array([0,1])
                else:
                    x_data[i] = proba
            else:
                if proba[0] > 1 - self._threshold[0]:
                    pass
                elif proba[1] > 1 - self._threshold[1]:
                    x_data[i] = 1
                else:
                    x_data[i]= 2
        return x_data

    def plot(self, y_data, x_data, ax):
        xx, yy = np.meshgrid(
                    np.arange(
                        min(y_data[:, 0]),
                        max(y_data[:, 0]),
                        (max(y_data[:, 0]) - min(y_data[:, 0])) / 500,
                    ),
                    np.arange(
                        min(y_data[:, 1]),
                        max(y_data[:, 1]),
                        (max(y_data[:, 1]) - min(y_data[:, 1])) / 500,
                    ),
                )
        x_data_pred = self.predict(y_data)
        scatter = ax.scatter(y_data[:, 0], y_data[:, 1], c=x_data_pred)
        zz = self.predict(np.c_[xx.ravel(), yy.ravel()])
        zz = np.array(zz).astype(float).reshape(xx.shape)
        ax.contourf(xx, yy, zz, alpha=0.2)
        ax.set_xlabel("I data")
        ax.set_ylabel("Q data")
        ax.set_title(self.discriminator_type)
        if len(scatter.legend_elements()[0]) == 2:
            ax.legend(*scatter.legend_elements())
        else:
            scatter2 = scatter.legend_elements()
            scatter2[1][-1] = 'Mixed state'
            ax.legend(*scatter2)
       
class GaussianMixtureModel():
    def __init__(self,  threshold = None):
        self._threshold = threshold
        self._discriminator = [None] *2
        self.gaussian01 = []
    
    def fit(self, y_data, x_data):
        for i in range(2):
            state = np.array([value for index,value in enumerate(y_data) if x_data[index]==i])
            means = np.mean(state, axis=0)
            self._discriminator[i] = GaussianMixture(n_components=2, covariance_type='spherical', max_iter=50)
            self._discriminator[i].means_init = np.array([[-means[0], means[1]],[means[0], means[1]]])
            self._discriminator[i].fit(state)
            estimators = []
            best = []
            for ind in range(self._discriminator[i].n_components):
                sigma = np.sqrt(self._discriminator[i].covariances_[ind])
                means = self._discriminator[i].means_[ind]
                weight = self._discriminator[i].weights_[ind]
                best= best +[weight]
                estimators.append([means, sigma, weight])
            self.gaussian01.append(estimators[best.index(max(best))])
    
    def predict(self, y_data, return_proba = False):
        x_data=[]
        for value in y_data:
            proba = self.compute_probability(value)
            if self._threshold is None:
                if return_proba:
                    x_data.append(proba)
                else:
                    if proba[0] > proba[1]:
                        x_data.append(0)
                    else:
                        x_data.append(1)
            else:
                if return_proba:
                    if proba[0] > 1 - self._threshold[0]:
                        x_data.append(np.array([1,0]))
                    elif proba[1] > 1 - self._threshold[1]:
                        x_data.append(np.array([0,1]))
                    else:
                        x_data.append(proba)
                else:
                    if proba[0] > 1 - self._threshold[0]:
                        x_data.append(0)
                    elif proba[1] > 1 - self._threshold[1]:
                        x_data.append(1)
                    else:
                        x_data.append(2)
        return np.array(x_data)
    
    # Compute the normal distribution p(u/0)
    @staticmethod
    def gaussian_distribution(state, mean, sd):
        if isinstance(state, complex):
            state=np.array([state.real,state.imag])
        return 0.5/(np.pi*sd**2)*np.exp(-0.5*np.linalg.norm(state-mean)**2/(sd**2))
    # The probability of all data points using bayes theorem

    def compute_probability(self, value):
        if len(value)==2 or isinstance(value, complex):
            gaussian0 = self.gaussian_distribution(value,self.gaussian01[0][0],self.gaussian01[0][1])
            gaussian1=self.gaussian_distribution(value,self.gaussian01[1][0],self.gaussian01[1][1])
            prob0=gaussian0/(gaussian0+gaussian1)
        else:
            ValueError('unsupported value')
        return np.array([prob0,1-prob0])
    
    def score(self, y_data, x_data):
        x_data_pred= np.array(self.predict(y_data))
        good_pred = np.count_nonzero(x_data_pred == [x_data])/len(x_data)
        mixed_states = np.count_nonzero(x_data_pred == [2]*len(x_data_pred))/len(x_data)
        return {'good_predictions': good_pred, 'mixed_states': mixed_states}
    
    def plot(self, y_data, x_data, axs):
        markers=['o', 's']
        colors = ['green', 'y']
        for i in range(2):
            data = np.array([value for index,value in enumerate(y_data) if x_data[index]==i])
            means = np.mean(data, axis=0)
            ax = axs[i]
            ax.scatter(data[:,0], data[:,1], s = 10, cmap = 'viridis', alpha = 0.5,
                   c = colors[i], marker = markers[i])
            ax.scatter(means[0], means[1], s = 100, marker = markers[i], cmap = 'viridis', c = 'k',
                   alpha = 1.0)
            ax.set_ylabel('I data', fontsize = 14)
            ax.set_xlabel('Q data', fontsize = 14)
            ax.set_title('state ' + str(i))
            estimators = []
            for ind in range(self._discriminator[i].n_components):
                sigma = np.sqrt(self._discriminator[i].covariances_[ind])
                means = self._discriminator[i].means_[ind]
                estimators.append([means, sigma])
            n_sigmas = 4
            for state in range(self._discriminator[i].n_components):
                xy, sigma = estimators[state]
                for i_s in range(n_sigmas):
                    ell = patches.Ellipse(xy, i_s * sigma , i_s * sigma, 0, lw = 1, fill = False)
                    ell.set_clip_box(ax.bbox)
                    ell.set_alpha(1)
                    ax.add_artist(ell)
    