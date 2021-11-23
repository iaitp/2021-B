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

import ipywidgets as widgets
import warnings
warnings.filterwarnings('ignore')


def rotate(X, angle):
    """
    This function rotates an aribitrary set of point coordinates by a given angle (in degrees)
    around the mean centre of the given points.
    X, the point coordinates rotated, is an array of form (n_instances * 2)
    If applied to a distribution of points, has the effect of changing variance and covariance

    Arguments:
    X - pointwise coordinates, shape (n_instances,2)
    angle - the angle around the mean centre of the instances to rotate
    """

    # Standard form of a counter-clockwise rotational transform by "angle"
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])

    # Centre of the distribution
    x_centre = np.mean(X, axis=0)

    # Normalise coordinates to centre at zero
    # necessary for rotation around the distributions own centre
    x_norm = X - x_centre

    # Empty array to store rotated coordinates in
    rotated = np.zeros_like(X)
    for i in range(X[:, 0].shape[0]):
        # Iterate over points, multiplying each coordinate vector by our rotation matrix

        rotated[i, :] = np.matmul(rotation_matrix, x_norm[i, :])

    # Move distribution back to its original centre
    rotated += x_centre

    return rotated


def normal_cloud(dev_x, dev_y, centre=0, samples=100, xtrans=0, ytrans=0):
    """Function to generate a gaussian cloud with zero covariance

    Arguments:
    dev_x, dev_y - deviations along each axis direction
    centre - initial centre of cloud (redundant)
    samples - the total number of instances in cloud
    xtrans, ytrans - shift the cloud by this amount
    """
    # Initial distribution
    cloud = np.random.normal(centre, (dev_x, dev_y), size=(samples, 2))

    # Shift cloud by xtrans and ytrans
    cloud[:, 0] = cloud[:, 0] + xtrans
    cloud[:, 1] = cloud[:, 1] + ytrans

    return cloud


def separate_gaussians(dev_x, dev_y, pos_samples, neg_samples, displacement, rotation=False):
    """
    Function to generate two separate gaussian clouds with variance and optional covariance, each translated away from the origin

    Arguments:
    dev_x, dev_y - deviations along x and y axis (ie long and short side of final ellipse)
    pos_samples, neg_samples - number of samples in each class/cloud
    displacement - translation of each cloud away from the origin in opposite directions
    """

    # Make clouds, first for the positive class
    cloud1 = normal_cloud(dev_y, dev_x, samples=pos_samples, xtrans=displacement, ytrans=displacement)
    cloud2 = normal_cloud(dev_y, dev_x, samples=neg_samples, ytrans=-displacement, xtrans=-displacement)

    # If rotation is specified, rotate each distribution by a random amount
    if rotation == True:
        cloud1 = rotate(cloud1, np.random.randint(0, high=180))
        cloud2 = rotate(cloud2, np.random.randint(0, high=180))

    # Labels for data
    y_pos = np.zeros((pos_samples)) + 1
    y_neg = np.zeros((neg_samples)) - 1

    # Final data
    x = np.append(cloud1, cloud2, axis=0)
    y = np.append(y_pos, y_neg)

    return x, y


def embedded_circle(centre_size, cloud_deviation, gap, total_samples):
    """
    Redundant function, not currently used. Generates a gaussian cloud within a ring of a second class
    """
    x = normal_cloud(cloud_deviation, cloud_deviation, samples=total_samples)

    x1s = x[:, 0]
    x2s = x[:, 1]
    distances = np.sqrt(x1s ** 2 + x2s ** 2)

    y = np.zeros(x.shape[0])

    y[distances > centre_size] = -1
    y[distances < centre_size] = 1

    x[distances > centre_size, 0] += np.sign(x[distances > centre_size, 0]) * gap
    x[distances > centre_size, 1] += np.sign(x[distances > centre_size, 1]) * gap

    return x, y


# First import the sklearn datasets
from sklearn import datasets


def construct_example_dataset(n_dimensions=2, circular=False):
    # Number of samples for negative and positive class
    pos_samples = 500
    neg_samples = 500

    data = pd.DataFrame()
    for i in range(int(n_dimensions / 2)):
        if circular == False:
            X_generic, Y_generic = separate_gaussians(2, 1.25, pos_samples, neg_samples, 4, rotation=True)
            data[f"X{2 * i}"] = X_generic[:, 0]
            data[f"X{2 * i + 1}"] = X_generic[:, 1]
        else:
            X_generic, Y_generic = embedded_circle(3, 6, 0.5, pos_samples + neg_samples)
            data[f"X{2 * i}"] = X_generic[:, 0]
            data[f"X{2 * i + 1}"] = X_generic[:, 1]

    data["target"] = Y_generic

    return data


def construct_sklearn_dataset(dataset):
    # Load a given dataset

    X = dataset.data

    try:  # try and give string labels - if not use the numeric ones from sklearn
        if dataset.target_names.shape[0] < 10:
            Y = dataset.target_names[dataset.target]
        else:
            Y = dataset.target
    except:
        Y = dataset.target

    # Put the data into a dataframe
    data = pd.DataFrame()

    for i in range(X.shape[1]):
        data[dataset["feature_names"][i]] = X[:, i]
        # data[f"X{i}"] = X[:,i]
    data["target"] = Y

    return data

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

def to_categorical(data):
    from sklearn.preprocessing import LabelEncoder

    enc = LabelEncoder().fit_transform(data)

    return enc



class analyser:

    """Initialisation of the analyser takes a dataframe and a label column as input.
    The user can optionally force a regression problem using the *regression kwarg.
    The user can also force the analyser to use all of the data provided, instead of the default 1000 sub-sample"""
    def __init__(self, data, label_col,
                 regression = False, limit_size = True, normalise = True):
        self.normalise = normalise

        #This part keeps processing fast, as we limit larger sets to 1000 instances
        #This tool isn't intended to actually build ML models, only provide useful insights
        if limit_size:
            if data.shape[0] > 1000:
                data = data.sample(n=1000)

        #Define the core class attributes
        self.data = data
        self.label_col = label_col
        self.init_data()

        if regression:
            self.problem = 'regression'
        else:
            if np.unique(self.Y).shape[0] >= 10:  #If the number of classes is greater than 10 assume regression
                self.problem = 'regression'
            else:
                self.problem = 'classification'

        self.analyse()

    def init_data(self):
        self.column_names = self.data.columns.to_list()
        self.target = self.label_col
        datacols = self.column_names.copy()
        datacols.remove(self.label_col)  #This is now a list of all the non-target features
        if "KMeans clusters" in datacols:
            datacols.remove("KMeans clusters")


        self.datacols = datacols

        self.Y = self.data[self.label_col].to_numpy()  #Array of class labels

        # #print(self.data.head())
        if self.normalise:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            self.data[datacols] = self.scaler.fit_transform(self.data[datacols])

        self.X_df = self.data[datacols]   #Dataframe only of features

        self.n_categorical = 0
        self.n_numeric     = 0
        for col in datacols:
            if np.unique(self.data[col]).shape[0] < 10:
                self.X_df[col] = to_categorical(self.data[col])
                self.n_categorical += 1
            else:
                self.n_numeric += 1
        #print(self.data.head())


        self.X = self.X_df.to_numpy()

    def analyse(self, other_arg = None):
        x1_drop = widgets.Dropdown(
            value=self.datacols[0],
            options=self.datacols,
            description='x Feature'
        )

        x2_drop = widgets.Dropdown(
            value=self.datacols[1],
            options=self.datacols,
            description='y Feature'
        )

        color_drop = widgets.Dropdown(
            value=self.label_col,
            options=self.column_names,# + [self.label_col, "KMeans clusters"],
            description='Color'
        )

        calculations = [('Mean', np.mean), ('Deviation', np.std), ('Max', np.max), ('Min', np.min), ('Range', np.ptp),
                        ('Feature Mean Separations', self.mean_separations), ('Feature Correlations', self.feature_correlations)]
        if self.problem != 'classification':
            calculations = calculations[:-2]
            self.use_classes = False
        else:
            self.use_classes = True

        dist_drop = widgets.Dropdown(
            value=np.mean,
            options=calculations,
            description='Calculation'
        )



        kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        order = widgets.BoundedIntText(
            value=1,
            min=0,
            max=5,
            step=1,
            description='Poly order:'
        )
        kernel = widgets.Dropdown(
            value = 'poly',
            options = kernels,
            description = 'Kernel'
        )

        pca_btn = widgets.Button(
            description='Add PCA',
            disabled=False,
            button_style='success',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Add PCA',
            icon=''  # (FontAwesome names without the `fa-` prefix)
        )
        pca_btn.on_click(self.add_PCA)

        rm_pca_btn = widgets.Button(
            description='Remove PCA',
            disabled=False,
            button_style='warning',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Remove PCA',
            icon=''  # (FontAwesome names without the `fa-` prefix)
        )
        rm_pca_btn.on_click(self.remove_PCA)

        tsne_btn = widgets.Button(
            description='Add TSNE',
            disabled=False,
            button_style='success',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Add TSNE',
            icon=''  # (FontAwesome names without the `fa-` prefix)
        )
        tsne_btn.on_click(self.add_TSNE)

        rm_tsne_btn = widgets.Button(
            description='Remove TSNE',
            disabled=False,
            button_style='warning',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Remove TSNE',
            icon=''  # (FontAwesome names without the `fa-` prefix)
        )
        rm_tsne_btn.on_click(self.remove_TSNE)

        add_clusters = widgets.Button(
            description='Find clusters',
            disabled=False,
            button_style='success',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Find clusters',
            icon=''  # (FontAwesome names without the `fa-` prefix)
        )
        add_clusters.on_click(self.find_clusters)

        rm_clusters = widgets.Button(
            description='Remove clusters',
            disabled=False,
            button_style='warning',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Remove clusters',
            icon=''  # (FontAwesome names without the `fa-` prefix)
        )
        rm_clusters.on_click(self.remove_clusters)


        reset_btn = widgets.Button(
            description='Compare new',
            disabled=False,
            button_style='info',  # 'success', 'info', 'warning', 'danger' or ''
            #tooltip='Reset the analysis process - allows comparison of dimensionality reduction features',
            icon=''  # (FontAwesome names without the `fa-` prefix)
        )
        reset_btn.on_click(self.analyse)

        dimensionality_buttons = [pca_btn, rm_pca_btn,  tsne_btn, rm_tsne_btn, reset_btn]

        sep_check = widgets.interactive_output(self.linear_separable,{'order':order, 'kernel':kernel})
        cluster = widgets.interactive_output(self.find_clusters,{})

        functions = widgets.VBox(children = [ widgets.HBox(children = dimensionality_buttons),
                                                 widgets.HBox(children = [sep_check,widgets.VBox(children =  [order, kernel])]),
                                              widgets.HBox(children=[add_clusters, rm_clusters])])


        compare = widgets.VBox(children = [widgets.interactive_output(self.compare, {'x1':x1_drop, 'x2':x2_drop, 'color_attrib':color_drop}),
                                        widgets.HBox(children = [x1_drop, x2_drop, color_drop])])

        importances = widgets.VBox(children=[widgets.interactive_output(self.feature_importances,{'color_attrib':color_drop}), color_drop])
        calcs = widgets.VBox(children = [widgets.interactive_output(self.dist, {'calculation':dist_drop}), dist_drop])

        rec_btn = widgets.Button(
            description='Re-run recommendation',
            disabled=False,
            button_style='success',  # 'success', 'info', 'warning', 'danger' or ''
            #tooltip='Reset the analysis process - allows comparison of dimensionality reduction features',
            icon=''  # (FontAwesome names without the `fa-` prefix)
        )
        rec_btn.on_click(self.recommend)

        recommendations = widgets.HBox(children = [widgets.interactive_output(self.recommend, {}), rec_btn])

        kids = [compare, importances, calcs, functions, recommendations]

        tab = widgets.Tab(children=kids)
        tab.set_title(0, 'Compare Features')
        tab.set_title(1, 'Best Features')
        tab.set_title(2, 'Calculations')
        tab.set_title(3, 'Functions')
        tab.set_title(4, 'Recommendations')

        display(tab)


    def recommend(self, dummy = None):
        corr = self.feature_correlations(display=False)
        sum_corr = np.sum(np.abs(corr.to_numpy())) - corr.shape[0]
        mean_corr = sum_corr / (corr.shape[0]**2 - corr.shape[0])

        recommendation = 'Recommendation:'
        reasons = '\nReasons:\n'

        if self.n_categorical != 0:
            reasons += f'Your data has {self.n_categorical} categorical columns\n'
            if self.n_numeric == 0:
                reasons += 'Your data only has categorical columns'
                recommendation += '\nNaive Bayes'

            else:
                reasons += 'Your data has mixed numeric and categorical columns'
                recommendation += '\nRandom Forest'
        else:
            reasons += 'Your data has no categorical columns\n'

            linseps = np.zeros(5, dtype=np.float32)
            i_max = 222
            for i in range(5):
                linseps[i] = self.linear_separable(order=i, report=False)
                if linseps[i] >= 0.99:
                    i_max = i
                    break
            if i_max == 222:
                i_max = np.argmax(linseps)

            if linseps[i_max] >= 0.99:
                reasons += f'Your data is linearly separable (score {linseps[i_max]})'
                recommendation += f'\nLinear model, kernel poly order {i_max}'

            elif linseps[i_max] >= 0.9:
                reasons += f'Your data is nearly linearly separable (score {linseps[i_max]}),\n but would require soft margins'
                recommendation += f'\nSVM, kernel poly order {i_max}'
            else:
                reasons += 'Your data is not linearly separable and has high feature correlations,\n fitting distributions is less likely to work'



                if mean_corr <= 0.25:
                    reasons += f'\nYour data has a low mean feature correlation ({mean_corr})'
                    recommendation += '\nGaussian Mixture Model'
                else:
                    reasons += f'\nYour data has a high mean feature correlation ({mean_corr})\n'
                    recommendation += '\nK-Means'


        print(recommendation, '\n',reasons)








    def compare_update(self, x1, x2):
        fig = self.compare(x1, x2)

        return fig




    """Function calculates the given metric,
    ideally a numpy function, on an overall or by-class basis"""
    def dist(self, calculation = np.mean, return_labels = True, dataframe = True):

        classes = self.use_classes
        #If the user has specified that the metric is for the whole dataset
        if classes == False:

            class_labels = [self.label_col]

            values = [calculation(self.X, axis=0)]

            if calculation == self.mean_separations:
                return self.mean_separations()
            elif calculation == self.feature_correlations:
                return self.feature_correlations()
        else:
            if calculation == self.mean_separations:
                return self.mean_separations()
            elif calculation == self.feature_correlations:
                return self.feature_correlations()

            #Get unique class labels
            class_labels = np.unique(self.Y)

            #Get the given metric for each class
            values = []
            for label in class_labels:
                values.append(calculation(self.X[self.Y == label, :], axis=0))

        if dataframe:
            import seaborn as sn
            df = pd.DataFrame(values, class_labels, self.datacols)
            fig = plt.figure(figsize=(df.shape[1],df.shape[0]))
            sn.heatmap(df, annot=True)

            return fig

        elif return_labels == True:
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
        class_means, class_labels = self.dist(dataframe=False)

        #Number of classes

        if self.problem == 'classification':
            n_classes = class_labels.shape[0]

            #Gamma is just a name, picture the hulk doing calculations. Here we store the mean separations.
            gamma_array = np.zeros((n_classes, n_classes), dtype=np.float64)
            #O(n^2) calculations - iterate over each class
            for i, c1 in enumerate(class_labels):
                for n, c2 in enumerate(class_labels):
                    if i != n:
                        gamma_array[i, n] = distances(class_means[i, :], class_means[n, :])
        else:
            return 'Mean separations not returned as this is a regression problem'

        #Adds the results array as a class attribute if *dataframe is specified as False
        if dataframe == False:
            self.separations = gamma_array
            return

        #Initialise pandas dataframe
        dataframe = pd.DataFrame(gamma_array, class_labels, class_labels)

        #Simple seaborn heatmap - using Seaborn here as its heatmap function allows automatic annotation
        if display:
            import seaborn as sn
            fig = plt.figure(figsize=(dataframe.shape[1],dataframe.shape[0]))
            sn.heatmap(dataframe, annot=True)
            #plt.show()

        #Add dataframe of results as a class attribute
        self.separations = dataframe

        return fig




    """Can't lie, I really like this function
    Plots two features against each other from a classification perspective, with histograms for each feature
    along with confidence intervals for each class

    OR

    A similar thing for regression (but this needs some work)

    Args:
    Takes two feature names as argument"""
    def compare(self, x1, x2, color_attrib):

        if x1 == x2:
            import seaborn as sn
            fig = plt.figure(figsize=(8,8))

            #sn.boxplot(self.data[x1], y = self.Y)
            #sn.swarmplot(self.data[x1], y = self.Y)

            sn.violinplot(self.data[x1], y = self.Y.astype(str))

            return fig



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
            fig = plt.figure(figsize=(8, 8))


            #Initial formatting for the axes
            ax_scatter = plt.axes(rect_scatter)
            ax_scatter.tick_params(direction='in', top=True, right=True)
            ax_histx = plt.axes(rect_histx)
            ax_histx.tick_params(direction='in', labelbottom=False)
            ax_histy = plt.axes(rect_histy)
            ax_histy.tick_params(direction='in', labelleft=False)

            #Scatter each class on the axis
            if color_attrib == self.label_col:
                for l in np.unique(self.Y):
                    ax_scatter.scatter(X[self.Y == l, 0], X[self.Y == l, 1], alpha=0.5, edgecolor='0')
            else:
                ax_scatter.scatter(X[:,0], X[:,1], c=self.data[color_attrib],  edgecolor='0')

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

            return fig



            #plt.show()

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
            #plt.show()

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
    def feature_importances(self, color_attrib, display=True, return_features=False):
        #Import forests, and instantiate a forest according to the analyser problem
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        if self.problem == 'classification':
            forest = RandomForestClassifier(n_estimators=250)#,verbose=1)
        elif self.problem == 'regression':
            forest = RandomForestRegressor(n_estimators=250)#,verbose=1)

        #Fit a forest to the data
        forest.fit(self.X, self.Y.astype(str))
        importances = forest.feature_importances_

        #SKLearn feature importances are by index
        #So we find the indexwise sort order and apply it to our feature names
        sort_inds = np.argsort(importances)
        sorted_features = np.array(self.datacols)[sort_inds[::-1]]

        if display:
            fig = self.compare(sorted_features[0], sorted_features[1], color_attrib)

        if return_features:

            return sorted_features, fig
        elif display:
            return fig

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
        self.projector.fit(self.X)
        PCA_transformation = self.projector.transform(self.X)

        for i in range(PCA_transformation.shape[1]):
            self.X_df[f"PCA_{i}"] = PCA_transformation[:,i]
            self.data[f"PCA_{i}"] = PCA_transformation[:,i]
            self.datacols.append(f"PCA_{i}")
            self.init_data()

        if display == True:
            if self.problem == 'classification':
                self.compare("PCA_0", "PCA_1")
            elif self.problem == 'regression':
                self.compare_regression("PCA_0", "PCA_1")


    def remove_PCA(self, empty=None):
        try:
            del self.data["PCA_0"]
            del self.data["PCA_1"]
            self.init_data()
        except:
            pass


    def add_TSNE(self, display = False):
        from sklearn.manifold import TSNE

        self.projector = TSNE(n_components=2)

        #self.projector.fit(self.X_df)
        PCA_transformation = self.projector.fit_transform(self.X)

        for i in range(PCA_transformation.shape[1]):
            self.X_df[f"TSNE_{i}"] = PCA_transformation[:,i]
            self.data[f"TSNE_{i}"] = PCA_transformation[:,i]
            self.datacols.append(f"TSNE_{i}")
            self.init_data()

        if display == True:
            if self.problem == 'classification':
                self.compare("TSNE_0", "TSNE_1")
            elif self.problem == 'regression':
                self.compare_regression("TSNE_0", "TSNE_1")

    def remove_TSNE(self, empty=None):
        try:
            del self.data["TSNE_0"]
            del self.data["TSNE_1"]
            self.init_data()
        except:
            pass


    """
    Calculates feature correlations
    If the problem is regression, this includes the target variable
    *kwargs:
    display - uses a seaborn heatmap to show feature correlations
    """
    def feature_correlations(self, display = True):

        if self.problem == 'classification':
            self.correlations = self.X_df.corr()
        elif self.problem == 'regression':
            self.correlations = self.data.corr()

        if display:
            import seaborn as sn
            fig = plt.figure(figsize=(self.correlations.shape[0],self.correlations.shape[1]))
            sn.heatmap(self.correlations, annot=True)
            #plt.show()
            return fig
        else:
            return self.correlations






    """
    Fits a high C-value svm to the data, and reports the degree of separability
    *kwargs:
    kernel - the type of kernel to use
    order - if kernel == 'poly', this specifies the polynomial order of the kernel
    """
    def linear_separable(self, kernel='poly', order=1, report = True):
        from sklearn.svm import SVC, SVR

        #Instantiate an SVM specific to the problem
        #A high C value forces a hard boundary (ie linear sepearation without soft margins)
        if self.problem == 'classification':
            svm = SVC(C=2^32,kernel=kernel, degree=order)
        elif self.problem == 'regression':
            svm = SVR(C=2^32,kernel=kernel, degree=order)

        #Fit the svm and find how it performs on the data
        #If score == 1 the data is perfectly linearly separable
        svm.fit(self.X, self.Y.astype(str))
        self.svm = svm
        score = svm.score(self.X, self.Y.astype(str))

        p_score = "{:.2f}".format(score)
        #Feed-back to the user (this will probably change in a later version)
        if report:
            if score >= 0.99:
                print(f'Data is perfectly separable, to an accuracy of {p_score}')
            elif score >= 0.9:
                print(f'Data is separable to an accuracy of {p_score}')
            else:
                print(f'Data is not separable, scoring an accuracy of {p_score}')
        else:
            return score

    def find_clusters(self, dummy = None):
        #try:
        from sklearn.cluster import KMeans
        # except:
        #     print('Clustering failed, is HDBSCAN installed?')

        min_cluster = int(self.X.shape[0] / 20)

        clusters = KMeans(n_clusters=np.unique(self.Y).shape[0]).fit_predict(self.X_df, self.Y)


        n_clusters = np.unique(clusters).shape[0]

        n_outliers = clusters[clusters == -1].shape[0]
        p_outliers = f'{n_outliers/clusters.shape[0]*100}%'

        print('KMeans clusters added to dataframe - press \"compare new\" to view')
        #print(f'HDBSCAN found {n_clusters} clusters, and {n_outliers} outliers, \n{p_outliers} of data')

        self.data[f"KMeans clusters"] = clusters
        self.init_data()
    def remove_clusters(self, dummy=None):
        try:
            del self.data["KMeans clusters"]
            self.init_data()
        except:
            pass




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
