"""Alex Davies, [other people who end up adding code here], 2021
This is the backend for Data HarPY
The module is built around the class "analyser", which takes a dataframe the name of the target column as init variables
Dependencies aren't huge, but more specialised modules are imported when methods are called"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from numpy.linalg import norm
import pandas as pd
from scipy.stats import chi2
from matplotlib.patches import Ellipse as el


# import math
# def normpdf(x, mean, sd):
#     var = float(sd)**2
#     denom = (2*math.pi*var)**.5
#     num = math.exp(-(float(x)-float(mean))**2/(2*var))
#     return num/denom
#
#
# def gamma_factor(mu1, mu2, dev1, dev2):
#     r = mu1 - mu2
#
#     mod_r = np.sum(np.abs(r))
#     rhat = r / mod_r
#
#
#     from1 = np.abs(np.dot(rhat, dev1))
#     from2 = np.abs(np.dot(rhat, dev2))
#     bottom = from1+from2
#
#
#
#     r = mu1 - mu2
#     r_hat = r / np.sum(np.abs(r))
#
#     dev1_norm = dev1 / np.sum(dev1)
#     dev2_norm = dev2 / np.sum(dev2)
#
#     r_signs = np.sign(r)
#
#     match_below = np.dot(r_signs*r_hat, dev1+dev2)
#     # match_above = np.abs(np.dot(r_signs*r_hat, dev2))
#
#     length_r = norm(r)
#
#     return length_r / (match_below)
#
#
#
#
#
#     # return mod_r / bottom


def distances(mu1, mu2):
    """Simple function, returns the euclidean
    distance between two given vector coordinates"""
    r = mu1 - mu2
    return norm(r)





class analyser:

    """Initialisation of the analyser takes a dataframe and a label column as input.
    The user can optionally force a regression problem using the *regression kwarg.
    The user can also force the analyser to use all of the data provided, instead of the default 1000 sub-sample"""
    def __init__(self, data, label_col,
                 regression = False, limit_size = True, normalise = True):
        print('analyser active') #Probably going to be removed in a later iteration

        #This part keeps processing fast, as we limit larger sets to 1000 instances
        #This tool isn't intended to actually build ML models, only provide useful insights
        if limit_size == True:
            if data.shape[0] > 1000:
                data = data.shuffle.sample(n=1000)

        #Define the core class attributes
        self.data = data



        self.column_names = data.columns.to_list()
        self.target = label_col
        datacols = self.column_names
        datacols.remove(label_col)  #This is now a list of all the non-target features

        self.datacols = datacols

        self.Y = data[label_col].to_numpy()  #Array of class labels

        print(self.data.head())
        if normalise:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            self.data[datacols] = self.scaler.fit_transform(self.data[datacols])
        print(self.data.head())
        self.X_df = self.data[datacols]   #Dataframe only of features

        self.X = self.X_df.to_numpy()
        print(self.X)


        if regression:
            self.problem = 'regression'
        else:
            if np.unique(self.Y).shape[0] >= 10:  #If the number of classes is greater than 10 assume regression
                self.problem = 'regression'
            else:
                self.problem = 'classification'

    """Function calculates the given metric,
    ideally a numpy function, on an overall or by-class basis"""
    def dist(self, classes = True, calculation = np.mean, return_labels = True):

        #If the user has specified that the metric is for the whole dataset
        if classes == False:
            return calculation(self.X, axis=0)

        #Get unique class labels
        class_labels = np.unique(self.Y)

        #Get the given metric for each class
        values = []
        for label in class_labels:
            values.append(calculation(self.X[self.Y == label, :], axis=0))


        if return_labels == True:
            return np.array(values), class_labels
        else:
            return np.array(values)


    """This function calculates the separation of the mean centres on a per-class basis
    There is a good deal of redundant code here from attempts to do some FANCY MATHS (probably failed attempt)
    Redundant code is flagged - i'm not going to document it as its not useable atm
    
    kwargs:
    compare_devs: currently redundant, pending FANCY MATHS
    dataframe: specifies the form of the returned results - if True, results are given row and column labels in a dataframe
    display: prints a nice heatmap of mean separations from the dataframe when True (ignored if dataframe == False)"""
    def mean_separations(self, compare_devs = False, dataframe = True, display=True):

        #Use the .dist method to get class means
        class_means, class_labels = self.dist()

        #The only if statement that is relevant in this current iteration
        if compare_devs == False:

            #Number of classes
            n_classes = class_labels.shape[0]

            #Gamma is just a name, picture the hulk doing calculations. Here we store the mean separations.
            gamma_array = np.zeros((n_classes, n_classes), dtype=np.float64)
            #O(n^2) calculations - iterate over each class
            for i, c1 in enumerate(class_labels):
                for n, c2 in enumerate(class_labels):
                    if i != n:
                        gamma_array[i, n] = distances(class_means[i, :], class_means[n, :])

        #REDUNDANT CODE
        #===================================================================================================================
        else:
            n_classes = class_labels.shape[0]

            gamma_array = np.zeros((n_classes, n_classes), dtype=np.float64)
            for i, c1 in enumerate(class_labels):
                for n, c2 in enumerate(class_labels):
                    if i != n:
                        gamma_array[i, n] = gamma_factor(class_means[i,:], class_means[n,:], class_devs[i,:], class_devs[n,:])

        # ===================================================================================================================

        #Adds the results array as a class attribute if *dataframe is specified as False
        if dataframe == False:
            self.separations = gamma_array
            return

        #Initialise pandas dataframe
        dataframe = pd.DataFrame(gamma_array, class_labels, class_labels)

        #Simple seaborn heatmap - using Seaborn here as its heatmap function allows automatic annotation
        if display:
            import seaborn as sn
            sn.heatmap(dataframe, annot=True)
            plt.show()

        #Add dataframe of results as a class attribute
        self.separations = dataframe



    """Can't lie, I really like this function
    Plots two features against each other from a classification perspective, with histograms for each feature
    along with confidence intervals for each class
    
    OR
    
    A similar thing for regression (but this needs some work)
    
    Args:
    Takes two feature names as argument"""
    def compare(self, x1, x2):

        #Only want the data for these two columns
        data_here = self.data.copy()[[x1, x2]]
        X = data_here.to_numpy()

        if self.problem == 'classification':

            # definitions for the axes
            left, width = 0.1, 0.65
            bottom, height = 0.1, 0.65
            spacing = 0.005

            rect_scatter = [left, bottom, width, height]
            rect_histx = [left, bottom + height + spacing, width, 0.2]
            rect_histy = [left + width + spacing, bottom, 0.2, height]

            # start with a rectangular Figure
            plt.figure(figsize=(12, 12))


            #Initial formatting for the axes
            ax_scatter = plt.axes(rect_scatter)
            ax_scatter.tick_params(direction='in', top=True, right=True)
            ax_histx = plt.axes(rect_histx)
            ax_histx.tick_params(direction='in', labelbottom=False)
            ax_histy = plt.axes(rect_histy)
            ax_histy.tick_params(direction='in', labelleft=False)

            #Scatter each class on the axis
            for l in np.unique(self.Y):
                ax_scatter.scatter(X[self.Y == l, 0], X[self.Y == l, 1], alpha=0.5, edgecolor='0')

            #Colors and intervals for the confidence intervals (inverval p containts (100*p) % of the data)
            colors = ['green', 'blue', 'orange', 'red']
            ps = [0.75, 0.9, 0.99, 0.999]

            #Iterate over unique classes
            for i, cl in enumerate(np.unique(self.Y)):

                #Iterate over confidence intervals
                for n, p in enumerate(ps):

                    #Only need to have a label for one of each interval
                    if i == 0:
                        label = ps[n]
                    else:
                        label = None

                    #Call a method to construct the confidence interval (below) and add it to the scatter axis
                    el2 = self.plot_confidence_interval(X, p, colors[n], label, y_class=cl)
                    ax_scatter.add_patch(el2)

                #For each class plot the (mean) centre
                cent = np.mean(X[self.Y == cl], axis=0)

                ax_scatter.scatter(cent[0], cent[1], marker='o', c='0')
                ax_scatter.scatter(cent[0], cent[1], marker='x', c='1')

            #Add a legend and axis labels to the scatter axis
            ax_scatter.legend(shadow=True, title="Confidence intervals")

            ax_scatter.set_xlabel(x1)
            ax_scatter.set_ylabel(x2)


            #Labels for the histograms
            labels = np.unique(self.Y)

            #Iterate over the classes, adding a pandas series to a list for each class and feature (x1,x2)
            datalist_x1 = []
            datalist_x2 = []
            for label in labels:
                datalist_x1.append(data_here[x1].loc[self.Y == label])
                datalist_x2.append(data_here[x2].loc[self.Y == label])


            #Get the axis limits from the scatter plot (histograms will be over the same range)
            y_lim = ax_scatter.get_ylim()
            x_lim = ax_scatter.get_xlim()

            #Histogram for x1 is horizontal, above the scatter
            ax_histx.hist(datalist_x1, bins=30, label=labels, histtype='bar', stacked=True)
            ax_histx.set_ylabel('N instances')
            ax_histx.legend(shadow=True)
            ax_histx.set_xlim(x_lim)

            #Same for x2, which is vertical, to the right of the scatter plot. NB: no need to add a legend here too
            ax_histy.hist(datalist_x2, bins=30, label=labels, histtype='bar', stacked=True, orientation = 'horizontal')
            ax_histy.set_xlabel('N instances')
            ax_histy.set_ylim(y_lim)

            plt.show()

        elif self.problem == 'regression':
            """Scatters x1 vs x2 (heatmap shows target value), x1 vs target, x2 vs target
            Definitely needs some work - plotting x vs y isn't actually that useful from a regression perspective"""

            X = self.data.copy()[[x1, x2]].to_numpy()

            fig, axes = plt.subplots(ncols=3, figsize=(27, 9))

            (ax1, ax2, ax3) = axes

            ax1.scatter(X[:, 0], X[:, 1], alpha=0.25, edgecolor='0', c=self.Y)  # x1 vs x2, color shows target value
            ax2.scatter(X[:, 0], self.Y, alpha=0.5, edgecolor='0')  # x1 vs target
            ax3.scatter(X[:, 1], self.Y, alpha=0.5, edgecolor='0')  # x2 vs target

            ax1.set_xlabel(x1)
            ax1.set_ylabel(x2)

            ax2.set_xlabel(x1)
            ax2.set_ylabel(self.target)

            ax3.set_xlabel(x2)
            ax3.set_ylabel(self.target)
            plt.show()

    """
    Generates a matplotlib.ellipse confidence interval for a distribution of points
    args:
    X - Collection of points, (2, n_instances)
    interval - confidence interval, value between 0 and 1, the proportion of data within the ellipse
    c - the colour of the generated ellipse
    label - class label, allows plotting of each confidence interval separately
    
    *kwargs:
    y_class - the label of the class to generate an ellipse for. If == None, generates an ellipse for all X
    """
    def plot_confidence_interval(self, X, interval, c, label, y_class = None):
        
        #If a class is specified, limit data to this class
        if y_class != None:
            X = X[self.Y == y_class]
        else:
            X = X
        
        #A value arising from some chi-squared maths, using a scipy function
        score = chi2.isf(1 - interval, 2)

        #The mean centre of the points
        centre = np.mean(X, axis=0)
        
        #Generate a covariance matrix for X, then find eigenvalues and vectors
        covariance_matrix = np.cov(X, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        #Eigenvalues are the semi-major and semi-minor axes of the ellipse
        #Eigenvectors show the direction of semi-minor and semi-major axes
        
        
        #Sort values and vectors according to eigenvalues
        sort_inds = np.argsort(eigenvalues)[::-1]

        eigenvalues = eigenvalues[sort_inds]
        eigenvectors = eigenvectors[:, sort_inds]
        
        
        #The angle of the ellipse to the horizontal axis
        angle = np.arctan(eigenvectors[1, 0] / eigenvectors[0, 0])
        angle = np.degrees(angle)
        
        #Generate the matplotlib.ellipse
        el2 = el(centre, width=2 * np.sqrt(score * eigenvalues[0]), height=2 * np.sqrt(score * eigenvalues[1]),
                 fill=False, color=c, angle=angle, label=label, alpha=0.7 + 0.3*(1-interval))

        return el2

    """
    Use a random forest to find the most "important" features in the data
    NB: Importance here is calculated using OOB scores, see SKLearn documentation
    
    Returns sorted list of feature importances
    """
    def feature_importances(self, display=True):
        #Import forests, and instantiate a forest according to the analyser problem
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        if self.problem == 'classification':
            forest = RandomForestClassifier(n_estimators=250,verbose=1)
        elif self.problem == 'regression':
            forest = RandomForestRegressor(n_estimators=250,verbose=1)

        #Fit a forest to the data
        forest.fit(self.X_df, self.Y)
        importances = forest.feature_importances_

        #SKLearn feature importances are by index
        #So we find the indexwise sort order and apply it to our feature names
        sort_inds = np.argsort(importances)
        sorted_features = np.array(self.datacols)[sort_inds[::-1]]

        if display:
            self.compare(sorted_features[0], sorted_features[1])


        return sorted_features

    """The next few functions are very similar
    They all apply a dimensional reduction technique to the data
    Then add that reduction to the class dataframe
    *kwargs:
    display - whether to use .compare(x1, x2) method"""
    def add_PCA(self, display = False):
        #Import the given decomposition from the relevant libary
        from sklearn.decomposition import PCA

        #Instantiate the imported class
        self.projector = PCA(n_components=2)
        self.projector.fit(self.X_df)
        PCA_transformation = self.projector.transform(self.X_df)

        for i in range(PCA_transformation.shape[1]):
            self.X_df[f"PCA_{i}"] = PCA_transformation[:,i]
            self.data[f"PCA_{i}"] = PCA_transformation[:,i]
            self.datacols.append(f"PCA_{i}")

        if display == True:
            if self.problem == 'classification':
                self.compare("PCA_0", "PCA_1")
            elif self.problem == 'regression':
                self.compare_regression("PCA_0", "PCA_1")

    def add_TSNE(self, display = False):
        from sklearn.manifold import TSNE

        self.projector = TSNE(n_components=2)

        #self.projector.fit(self.X_df)
        PCA_transformation = self.projector.fit_transform(self.X_df)

        for i in range(PCA_transformation.shape[1]):
            self.X_df[f"TSNE_{i}"] = PCA_transformation[:,i]
            self.data[f"TSNE_{i}"] = PCA_transformation[:,i]
            self.datacols.append(f"TSNE_{i}")

        if display == True:
            if self.problem == 'classification':
                self.compare("TSNE_0", "TSNE_1")
            elif self.problem == 'regression':
                self.compare_regression("TSNE_0", "TSNE_1")

    def add_UMAP(self, display = False):
        from umap import UMAP

        self.projector = UMAP(n_components=2)

        PCA_transformation = self.projector.fit_transform(self.X_df)

        for i in range(PCA_transformation.shape[1]):
            self.X_df[f"UMAP_{i}"] = PCA_transformation[:,i]
            self.data[f"UMAP_{i}"] = PCA_transformation[:,i]
            self.datacols.append(f"UMAP_{i}")

        if display == True:
            if self.problem == 'classification':
                self.compare("UMAP_0", "UMAP_1")
            elif self.problem == 'regression':
                self.compare_regression("UMAP_0", "UMAP_1")

    """
    Calculates feature correlations
    If the problem is regression, this includes the target variable
    *kwargs:
    display - uses a seaborn heatmap to show feature correlations
    """
    def feature_correlations(self, display = False):

        if self.problem == 'classification':
            self.correlations = self.X_df.corr()
        elif self.problem == 'regression':
            self.correlations = self.data.corr()

        if display:
            import seaborn as sn
            sn.heatmap(self.correlations, annot=True)
            plt.show()





    """
    Fits a high C-value svm to the data, and reports the degree of separability
    *kwargs:
    kernel - the type of kernel to use
    order - if kernel == 'poly', this specifies the polynomial order of the kernel
    """
    def linear_separable(self, kernel='linear', order=1):
        from sklearn.svm import SVC, SVR

        #Instantiate an SVM specific to the problem
        #A high C value forces a hard boundary (ie linear sepearation without soft margins)
        if self.problem == 'classification':
            svm = SVC(C=2^32,kernel=kernel, degree=order)
        elif self.problem == 'regression':
            svm = SVR(C=2^32,kernel=kernel, degree=order)

        #Fit the svm and find how it performs on the data
        #If score == 1 the data is perfectly linearly separable
        svm.fit(self.X_df, self.Y)
        self.svm = svm
        score = svm.score(self.X_df, self.Y)

        #Feed-back to the user (this will probably change in a later version)
        if score >= 0.99:
            print(f'Data is perfectly linearly separable, to an accuracy of {score}')
        elif score >= 0.9:
            print(f'Data is linearly separable to an accuracy of {score}')
        else:
            print(f'Data is not linearly separable, scoring an accuracy of {score}')



    # def linear_separable(self):
    #     from scipy.optimize import linprog
    #
    #     df = self.data.copy()
    #
    #     for i in np.unique(self.Y):
    #         newTarget = np.where(self.Y == i, 1, -1)
    #
    #         from sklearn.preprocessing import StandardScaler
    #         sc = StandardScaler()
    #         tmp = self.X_df.to_numpy()
    #         tmp = sc.fit_transform(tmp)
    #
    #         xx = np.array(newTarget.reshape(-1, 1) * tmp)
    #         t = np.where(self.Y == i, 1, -1)
    #
    #         # 2-D array which, when matrix-multiplied by x, gives the values of
    #         # the upper-bound inequality constraints at x.
    #         A_ub = np.append(xx, t.reshape(-1, 1), 1)
    #
    #         # 1-D array of values representing the upper-bound of each
    #         # inequality constraint (row) in A_ub.
    #         b_ub = np.repeat(-1, A_ub.shape[0]).reshape(-1, 1)
    #
    #         # Coefficients of the linear objective function to be minimized.
    #         c_obj = np.repeat(1, A_ub.shape[1])
    #         res = linprog(c=c_obj, A_ub=A_ub, b_ub=b_ub,
    #                       options={"disp": False})
    #
    #         if res.success:
    #             print(f'There is linear separability between {i} and the rest')
    #         else:
    #             print(f'No linear separability between {i} and the rest')



    # def proportion_overlap(self):
    #
    #     gamma_array = self.mean_separations(compare_devs=True)
    #
    #     class_means, class_labels = self.dist(classes=True)
    #     class_devs = self.dist(classes=True, return_labels=False, calculation=np.std)
    #
    #     mean_devs = np.mean(class_devs, axis=1)
    #     mean_means = np.mean(class_means, axis=1)
    #
    #     prop_array = np.zeros_like(gamma_array)
    #
    #     n_classes = class_labels.shape[0]
    #
    #     for i in range(n_classes):
    #         for n in range(n_classes):
    #             gamma = gamma_array[i,n]
    #             prop_array[i,n] = normpdf(gamma*mean_devs[i], mean_means[i], mean_devs[i])
    #
    #     return prop_array

    # def svm_separable(self, kernel='linear', order=1):
    #     from sklearn.svm import SVC, SVR
    #
    #     if self.problem == 'classification':
    #         svm = SVC(C=2^32,kernel=kernel, degree=order)
    #     elif self.problem == 'regression':
    #         svm = SVR(kernel=kernel, degree=order)
    #
    #     svm.fit(self.X_df, self.Y)
    #
    #     self.svm = svm
    #
    #     print(svm.fit_status_)





