from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# Color maps for 2 labels
cmap_bg = ListedColormap(['#FFAAAA', '#AAAAFF'])
cmap_fg = ['#FF0000', '#0000FF']


class TrainAndTestData:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
    def print_errors(self, clf):
        # Get training error
        train_err = empirical_err(clf, self.X_train, self.y_train)
        print(f'Train error: {train_err*100:0.2f}%')
        # Get test error
        test_err = empirical_err(clf, self.X_test, self.y_test)
        print(f'Test error: {test_err*100:0.2f}%')


def add_label_noise(y, noise_level: float = 0.) -> np.ndarray:
    '''
    Adds noise to labels and returns a modified array of labels. Labels are {-1, +1} valued.
    Each labels is replaced with a random label with probability noise_level.
    noise_level=0 : no corruption, returns y itself
    noise_level=1 : returns uniformly random labels
    noise_level=0.5 : means approx. 1/2 the labels will be replaced with
    uniformly random labels, so only 1/4 would actually flip.
    
    Args:
        noise_level: probability of corruption
    '''
    
    assert 0 <= noise_level <= 1
    return y * (1 - 2 * (np.random.rand(len(y)) > 1-noise_level/2.0))


def generate_spiral_data(
    m: int,
    noise_level: float = 0.,
    theta_sigma: float = 0.,
    r_sigma: float = 0.
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Generates m spiral data points from a distribution specified with theta_sigma
    and r_sigma. Labels are in {-1, +1}. With probability noise_level,
    each label is replaced with a random label.
    '''
    y = 1 - 2*(np.random.rand(m) > 0.5)
    true_r = np.random.rand(m)
    theta = true_r*10 + 5*y + theta_sigma*np.random.randn(m)
    r = (1 + r_sigma*np.random.randn(m))*true_r
    X = np.column_stack((r*np.cos(theta), r*np.sin(theta)))
    y = add_label_noise(y, noise_level)
    return X, y


def create_split(X: np.ndarray, y: np.ndarray, split_ratio: float, seed = 310202024) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Randomly splits (X, y) into sets (X1, y1, X2, y2) such that
    (X1, y1) contains split_ratio fraction of the data. Rest goes in (X2, y2).

    Args:
        X: data features of shape (m, d)
        y: data labels of shape (m)
        split_ratio: fraction of data to keep in (X1, y1) (must be between 0 and 1)
        seed (defaults to the arbirtrary number 31020): a seed to use for the random number generator.
        Using a hard coded seed ensures the same split every time the function is called.

    Returns:
        (X1, y1, X2, y2): each is a numpy array
    '''
    assert 0. <= split_ratio <= 1.
    assert X.shape[0] == len(y)
    assert len(y.shape) == 1

    m = X.shape[0]
    # The following line creates an independent source of psuedo-random numbers, which doesn't
    # affect subequent number generation that won't use the created rng.  We use this so that we
    # can get consistent splits in this routine, by using a fixed seed, without making subsequent
    # random numbers also be the same on each execution.
    rng = np.random.default_rng(seed)
    idxs_shuffled = rng.permutation(m)
    X_shuffled, y_shuffled = X[idxs_shuffled], y[idxs_shuffled]

    m1 = int(split_ratio * m)
    X1, y1 = X_shuffled[:m1], y_shuffled[:m1]
    X2, y2 = X_shuffled[m1:], y_shuffled[m1:]

    return (X1, y1, X2, y2)


def empirical_err(predictor, X, y):
    """
    Returns the empirical error of the predictor on the given sample.

    Args:
        predictor-- an object with predictor.predict(x) method
        X: array of input instances
        y: array of true (correct) labels

    Returns:
        err: empirical error value
    """
    assert len(X) == len(y)

    pred_y = predictor.predict(X)
    err = np.mean(y != pred_y)

    return err


def plot_decision_boundary(clf, X_train, y_train, X_test, y_test, labels=[-1, 1]):
    '''
    Plots the decision boundary of the given classifier on training and testing points.
    Colors the training points with true labels, and shows the incorrectly and correctly predicted test points.
    '''
    X, y = np.vstack([X_train, X_test]), np.hstack([y_train.flatten(), y_test.flatten()])
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    
    # Create a mesh of points
    eps = 0.
    x1s = np.linspace(x_min[0]-eps, x_max[0]+eps, 100)
    x2s = np.linspace(x_min[1]-eps, x_max[1]+eps, 100)
    xx1, xx2 = np.meshgrid(x1s, x2s)
    Z = clf.predict(np.c_[xx1.ravel(), xx2.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx1.shape)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pcolormesh(xx1, xx2, Z, cmap=cmap_bg, shading='auto')

    # Plot training points
    for i, l in enumerate(labels):
        l_idxs = np.where(y_train == l)
        ax.scatter(X_train[l_idxs, 0], X_train[l_idxs, 1], label=f'train/{l}', c=cmap_fg[i], marker='.')
    
    # Plot test points
    y_test_predict = clf.predict(X_test)
    for i, l in enumerate(labels):
        # Mark the wrong ones
        wrong_idxs = np.where((y_test_predict == l) & (y_test_predict != y_test))
        ax.scatter(X_test[wrong_idxs, 0], X_test[wrong_idxs, 1], label=f'test/predicted {l} (wrong)', c=cmap_fg[1-i], marker='x', s=100)

        # Plot the correct ones
        corr_idxs = np.where((y_test_predict == l) & (y_test_predict == y_test))
        ax.scatter(X_test[corr_idxs, 0], X_test[corr_idxs, 1], label=f'test/predicted {l} (correct)', c=cmap_fg[i], marker='+', s=50)
        
    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title('Decision boundary\nShaded regions show what the label clf would predict for a point there')
    plt.legend(title='label', bbox_to_anchor=(1.04, 1), loc='upper left')


def scatter_plot(X, y, labels, **plot_kwargs):
    plt.figure(figsize=(8, 6))
    for i, l in enumerate(labels):
        l_idxs = np.where(y == l)
        plt.scatter(X[l_idxs, 0], X[l_idxs, 1], label=l, c=cmap_fg[i], **plot_kwargs)
    plt.xlabel('$x_1$') # matplotlib allows basic latex in rendered text!
    plt.ylabel('$x_2$')
    plt.legend(title='label')


def load_data(path_to_data, type, filter_neutrals=True):
    '''
    path_to_data: path to the file where the data is stored
    filter_neutrals:
    '''
    if type=="train":
        with open(path_to_data, 'r', encoding='utf-8') as f:
            sample_tuples = f.readlines()
        if filter_neutrals:
            samples = [s.split('\t') for s in sample_tuples if s.split('\t')[0]!='neutral']
        else:
            samples = [s.split('\t') for s in sample_tuples]
        samples = [s for s in samples if len(s) == 2] # filtering here by length to clean up some extra new lines that crept in
        ys = [1 if s[0]=='positive' else -1 for s in samples]
        Xs = [s[1].strip() for s in samples]
        return Xs, ys 
    else:
        with open(path_to_data, 'r', encoding='utf-8') as f:
            Xs = f.readlines()
            return Xs


def plot_decision_boundary_with_svm(clf, X_train, y_train, X_test, y_test, labels=[-1, 1], C=None):
    '''
    Plots the decision boundary of the SVM classifier along with support vectors,
    margins, violations, and a legend for the decision boundary and margins.
    '''
    # Combine training and test data for axis scaling
    # X, y = np.vstack([X_train, X_test]), np.hstack([y_train.flatten(), y_test.flatten()])
    X, y = X_train, y_train.flatten()
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)

    # Create a mesh of points
    eps = 0.3  # padding for mesh grid
    x1s = np.linspace(x_min[0] - eps, x_max[0] + eps, 200)
    x2s = np.linspace(x_min[1] - eps, x_max[1] + eps, 200)
    xx1, xx2 = np.meshgrid(x1s, x2s)

    # Predict decision boundary values
    Z = clf.decision_function(np.c_[xx1.ravel(), xx2.ravel()])
    Z = Z.reshape(xx1.shape)

    # Plot decision boundary and margins
    fig, ax = plt.subplots(figsize=(8, 6))
    background = ax.contourf(xx1, xx2, np.sign(Z), cmap=cmap_bg)  # Background based on decision boundary
    decision_boundary = ax.contour(xx1, xx2, Z, levels=[0], colors=['black'], linestyles=['-'], linewidths=2)
    positive_margin = ax.contour(xx1, xx2, Z, levels=[1], colors=['blue'], linestyles=['--'], linewidths=2)
    negative_margin = ax.contour(xx1, xx2, Z, levels=[-1], colors=['red'], linestyles=['--'], linewidths=2)

    # Plot training points with labels
    for i, l in enumerate(labels):
        l_idxs = np.where(y_train == l)
        ax.scatter(X_train[l_idxs, 0], X_train[l_idxs, 1], label=f'Train/{l}', c=cmap_fg[i], marker='.')

    # # Plot test points
    # y_test_predict = clf.predict(X_test)
    # for i, l in enumerate(labels):
    #     wrong_idxs = np.where((y_test_predict == l) & (y_test_predict != y_test))
    #     ax.scatter(X_test[wrong_idxs, 0], X_test[wrong_idxs, 1], label=f'Test/Predicted {l} (Wrong)', c=cmap_fg[1 - i], marker='x', s=100)

    #     corr_idxs = np.where((y_test_predict == l) & (y_test_predict == y_test))
    #     ax.scatter(X_test[corr_idxs, 0], X_test[corr_idxs, 1], label=f'Test/Predicted {l} (Correct)', c=cmap_fg[i], marker='+', s=50)

    # Highlight support vectors
    SVs = X_train[clf.support_]
    ax.scatter(SVs[:, 0], SVs[:, 1], s=50, facecolors='none', edgecolors='purple', linewidths=1.5, label='Support Vectors')
    

    # Plot margin violations if C is provided
    if C is not None:
        alpha = clf.dual_coef_.ravel()
        tolC = C * (1 - 1e-5)
        pos_violations = SVs[alpha >= tolC]
        neg_violations = SVs[alpha <= -tolC]
        if len(neg_violations) > 0:
            ax.scatter(neg_violations[:, 0], neg_violations[:, 1], s=50, facecolors='r', edgecolors='k', marker='s', label='Margin Violations (Negative)')
        if len(pos_violations) > 0:
            ax.scatter(pos_violations[:, 0], pos_violations[:, 1], s=50, facecolors='b', edgecolors='k', marker='s', label='Margin Violations (Positive)')
        

    # Add custom legend for the decision boundary and margins
    custom_lines = [
        plt.Line2D([0], [0], color='black', lw=2, linestyle='-', label='Decision Boundary'),
        plt.Line2D([0], [0], color='red', lw=2, linestyle='--', label='Negative Margin'),
        plt.Line2D([0], [0], color='blue', lw=2, linestyle='--', label='Positive Margin')
    ]

    ax.legend(handles=custom_lines + ax.get_legend_handles_labels()[0],
              loc='upper left', bbox_to_anchor=(1.04, 1), title="Legend")

    # Adjust plot settings
    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title('SVM Decision Boundary with Margins and Support Vectors')
    plt.show()

def demo_1():

    x=np.array([[0.29637097, 0.7202381 ],
       [0.56854839, 0.60930736],
       [0.59475806, 0.53625541],
       [0.5625    , 0.46861472],
       [0.81048387, 0.55790043],
       [0.79032258, 0.36580087],
       [0.21370968, 0.58766234],
       [0.12298387, 0.43885281],
       [0.28427419, 0.27651515],
       [0.70766129, 0.8284632 ],
       [0.55846774, 0.1737013 ],
       [0.39112903, 0.63095238],
       [0.7016129 , 0.25757576],
       [0.37298387, 0.14935065],
       [0.22177419, 0.16558442],
       [0.3266129 , 0.48755411],
       [0.46370968, 0.48214286],
       [0.46169355, 0.32792208],
       [0.53024194, 0.78787879],
       [0.8266129 , 0.73106061]])
    y=np.array([ 1., -1., -1., -1., -1., -1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
        1.,  1., -1., -1., -1., -1., -1.])
    return TrainAndTestData(x, y, None, None)

def demo_2():

    x=np.array([[0.23185484, 0.74729437],
       [0.29435484, 0.4469697 ],
       [0.76612903, 0.6228355 ],
       [0.63709677, 0.23322511],
       [0.50806452, 0.62824675],
       [0.11290323, 0.3495671 ],
       [0.24596774, 0.11147186],
       [0.68346774, 0.14935065],
       [0.64717742, 0.77435065],
       [0.32862903, 0.50378788],
       [0.44959677, 0.30898268],
       [0.18346774, 0.16558442],
       [0.11693548, 0.66883117],
       [0.51612903, 0.54437229],
       [0.7983871 , 0.43885281],
       [0.90927419, 0.2521645 ],
       [0.82459677, 0.76623377],
       [0.42137097, 0.77435065],
       [0.61491935, 0.35768398],
       [0.25604839, 0.54978355]])
    y=np.array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1.])
    return TrainAndTestData(x, y, None, None)

def demo_3():

    x=np.array([[0.80443548, 0.47132035],
       [0.43346774, 0.58495671],
       [0.24193548, 0.34686147],
       [0.4516129 , 0.13582251],
       [0.75604839, 0.53354978],
       [0.57056452, 0.29545455],
       [0.35483871, 0.33603896],
       [0.56854839, 0.45238095],
       [0.54233871, 0.24675325],
       [0.13306452, 0.88528139],
       [0.57258065, 0.80681818],
       [0.83870968, 0.81764069],
       [0.22177419, 0.60660173],
       [0.83266129, 0.17911255],
       [0.6391129 , 0.54707792],
       [0.18548387, 0.1737013 ],
       [0.09677419, 0.43344156],
       [0.64919355, 0.21158009],
       [0.69758065, 0.30898268],
       [0.31451613, 0.76623377]])
    y=np.array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1.,  1.])
    return TrainAndTestData(x, y, None, None)

def get_demo(i):
    demo_functions = {
        1: demo_1,
        2: demo_2,
        3: demo_3
    }
    if i in demo_functions:
        demo= demo_functions[i]()
        return demo
    else:
        raise ValueError(f"Invalid input {i}. Choose between 1, 2, or 3.")
  