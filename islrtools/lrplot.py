__author__ = 'ryu'

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pandas import Series
from statsmodels.stats.outliers_influence import variance_inflation_factor


def plot_R_graphs(result):
    infl = result.get_influence()
    fig = plt.figure()

    ax1 = fig.add_subplot(221)
    ax1.scatter(result.fittedvalues, result.resid, c='w')
    ax1.set_ylabel("Residuals")
    ax1.set_xlabel("Fitted values")
    ax1.set_title("Residuals vs Fitted")

    ax2 = fig.add_subplot(222)
    probplot = sm.ProbPlot(infl.resid_studentized_internal)
    probplot.qqplot(ax=ax2, xlabel="Theoretical Quantitles", ylabel="Standardized residuals", c='w')
    ax2.set_title("Normal Q-Q")

    ax3 = fig.add_subplot(223)
    ax3.scatter(result.fittedvalues, np.sqrt(np.abs(infl.resid_studentized_internal)), c='w')
    ax3.set_ylabel(r"$\sqrt{|Standardized \; residuals|}$")
    ax3.set_xlabel("Fitted values")
    ax3.set_title("Scale-Location")

    ax4 = fig.add_subplot(224)
    ax4.scatter(infl.hat_matrix_diag, infl.resid_studentized_external, c='w')
    ax4.set_ylabel("Studentized residuals")
    ax4.set_xlabel("Leverage")
    ax4.set_title("Residuals vs Leverage")

    plt.show()


def plot_fitted_student_residual(df, result):
    infl = result.get_influence()
    graph_y = infl.resid_studentized_external
    graph_x = result.fittedvalues
    original_index = df.index
    #print graph_x, graph_y
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(graph_x, graph_y, c='w')
    assert(len(graph_y) == len(graph_x))
    print len(graph_y)
    for i in xrange(len(graph_x)):
        if np.abs(graph_y[i]) > 2:
            ax.annotate(original_index[i], (graph_x[i], graph_y[i]), xytext=(-3, 3),
                        textcoords="offset points", size="x-small")
    plt.show()


def plot_fitted_residual(result):
    ''' How to fit this type of data? '''
    graph_x = result.fittedvalues  #result.predict(self.X)
    graph_y = result.resid         #self.df['mpg'] - graph_x
    plt.scatter(graph_x, graph_y, c='w')
    '''
    x_new = np.linspace(graph_x.min(), graph_x.max())
    power_smooth = spline(graph_x, graph_y, x_new)
    plt.plot(x_new, power_smooth)
    '''
    plt.show()


def plot_qq(result):
    infl = result.get_influence()
    probplot = sm.ProbPlot(infl.resid_studentized_internal)
    probplot.qqplot(c='w')
    plt.show()


def plot_scale_location(result):
    infl = result.get_influence()
    graph_y = np.sqrt(np.abs(infl.resid_studentized_internal))
    graph_x = result.fittedvalues
    plt.scatter(graph_x, graph_y, c='w')
    plt.show()


def get_vifs(df):
    X = sm.add_constant(df)
    col_num = X.shape[1]
    df = X.ix[:, 1:]
    vif_list = [variance_inflation_factor(np.array(X), i) for i in np.arange(1, col_num, 1)]
    result = Series(vif_list, df.columns)
    print "VIF of all columns are: \n", result


def get_leverages_resid(result):
    infl = result.get_influence()
    leverage = infl.hat_matrix_diag
    resid = infl.resid_studentized_external
    print "leverage is:\n", leverage
    print "studentize residual is:\n", resid
    plt.scatter(leverage, resid, c='w')
    plt.show()