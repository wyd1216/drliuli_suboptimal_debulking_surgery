# python3.7
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : YuDong Wang 
# Created Date: 25/07/2022
# version ='1.0'
# ---------------------------------------------------------------------------
""" 
Package for draw the evaluation figure for sci paper in deep learning.
"""  
import pathlib
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix    #导入计算混淆矩阵的包
from math import sqrt
from natsort import natsorted, os_sorted
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FuncFormatter, ScalarFormatter
from sklearn.calibration import calibration_curve
from itertools import cycle

from deepplot.statistics import DelongTest

plt.clf()
plt.style.use(pathlib.Path(__file__).parent/'mplstyle'/'wydplot.mplstyle')

# ====================================================================================================
# == SCI Paper Format
# ====================================================================================================
def Plt_ticks_set(ax, x_range=1.0, y_range=1.0):
    '''
    :Description: Set ticks
    :param ax: Paths of the batch csv files. All the files have the same columns.
    :param x_range: 'float', = x_max - x_min
    :param y_range: 'float', = y_max - y_min
    :return: ax
    '''
    #ax.tick_params(which='major',length=10)
    #ax.tick_params(which='minor',length=5)
    xinter = x_range/5
    yinter = y_range/5
    xmajorLocator   = MultipleLocator(xinter) #将x主刻度标签设置为20的倍数
    xmajorFormatter = FormatStrFormatter('%5.1f') #设置x轴标签文本的格式
    xminorLocator   = MultipleLocator(xinter/5) #将x轴次刻度标签设置为5的倍数
    ymajorLocator   = MultipleLocator(yinter) #将y轴主刻度标签设置为0.2的倍数
    ymajorFormatter = FormatStrFormatter('%5.1f') #设置y轴标签文本的格式
    yminorLocator   = MultipleLocator(yinter/5) #将此y轴次刻度标签设置为0.1的倍数
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.xaxis.set_major_formatter(xmajorFormatter)
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    # ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    #显示次刻度标签的位置,没有标签文本
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    ax.yaxis.offsetText.set_fontsize(30)
    return ax

def Plt_legend_set(ax, style=0):
    if style == 0:
        # Transparent format
        legend = ax.legend()
    return legend

def subplots_remove_residual(ax, subfig_num, rows, cols):
    for i in range(subfig_num, cols * rows):
        ax[i // cols, i % cols].set_xticks([])
        ax[i // cols, i % cols].set_yticks([])
        ax[i // cols, i % cols].spines['left'].set_linewidth(0)
        ax[i // cols, i % cols].spines['right'].set_linewidth(0)
        ax[i // cols, i % cols].spines['top'].set_linewidth(0)
        ax[i // cols, i % cols].spines['bottom'].set_linewidth(0)
# ====================================================================================================
# == Draw the confusion-matrix
# ====================================================================================================
def confusion_matrix_plot(y_true, y_pred, savefig='CM.png', fig_title=None):
    '''
    :Description: To draw the  confusion-matrix based on the truth and predicted label.
    :
    :savefig: path of output fig.
    '''
    C = confusion_matrix(y_true, y_pred)
    C = C.astype(int)
    fig, ax = plt.subplots(figsize=(8,6))
    df=pd.DataFrame(C)
    sns.heatmap(df, fmt='g', annot=True, cmap='Blues', ax=ax, annot_kws={"size":35})
    ax.set_xlabel('Predict',fontsize=40, color='k') #x轴label的文本和字体大小
    ax.set_ylabel('True',fontsize=40, color='k') #y轴label的文本和字体大小
    ax.tick_params(labelsize=35)
    #设置colorbar的刻度字体大小
    cax = plt.gcf().axes[-1] 
    cax.tick_params(labelsize=35)
    #设置colorbar的label文本和字体大小
    cbar = ax.collections[0].colorbar
    if fig_title:
        ax.set_title(fig_title, fontsize=40)
    plt.savefig(savefig)
    return ax
    
# ====================================================================================================
# == Calculate the metrics
# ====================================================================================================
def specificity_score(y_true, y_pred):
    C = confusion_matrix(y_true, y_pred)
    TP = C[1,1]
    FP = C[0,1]
    TN = C[0,0]
    FN = C[1,0]
    specificity = TN/(TN+FP)
    return specificity

def classification_evaluation(y_true, y_pred, y_score):
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    accuracy = metrics.accuracy_score(y_true, y_pred)
    auc = metrics.roc_auc_score(y_true, y_score)
    f1 = metrics.f1_score(y_true, y_pred)
    specificity = specificity_score(y_true, y_pred)
    auc_low, auc_up = roc_auc_ci(y_true, y_score)
    evaluation = {'auc':auc, 'auc_CI':[auc_low, auc_up], 'sensitivity(recall)':recall, 'specificity':specificity, 'accuracy':accuracy, 'precision':precision, 'f1':f1}
    return evaluation

def Cmatrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat

def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings

def quadratic_weighted_kappa(y_true, y_pred):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y_true
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = Cmatrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)

# ====================================================================================================
# == (ROC)
# ====================================================================================================
# Calculate the confidence interval of the
def roc_auc_ci(y_true, y_score, positive=1):
    AUC = roc_auc_score(y_true, y_score)
    N1 = sum(y_true == positive)
    N2 = sum(y_true != positive)
    Q1 = AUC / (2 - AUC)
    Q2 = 2*AUC**2 / (1 + AUC)
    SE_AUC = sqrt((AUC*(1 - AUC) + (N1 - 1)*(Q1 - AUC**2) + (N2 - 1)*(Q2 - AUC**2)) / (N1*N2))
    lower = AUC - 1.96*SE_AUC
    upper = AUC + 1.96*SE_AUC
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1
    return (lower, upper)

# Define the ROC Plot on the subplot(ax).
def roc_plot_inside(y_true, y_score, ax, label='', positive=1):
    fpr,tpr,threshold = roc_curve(y_true, y_score) #计算真正率和假                            
    roc_auc = auc(fpr,tpr) #计算auc的值
    roc_auc_low, roc_auc_up = roc_auc_ci(y_true, y_score, positive=positive)
    if roc_auc_up > 1.0:
        roc_auc_up = 1.0
    ax.plot(fpr, tpr, lw=2, label='{3} (AUC={0:0.3f}, 95%CI:({1:0.3f}-{2:0.3f}))'.format(roc_auc,roc_auc_low,roc_auc_up,label)) ###假正率为横坐标，真正率为纵坐标做曲线
    return threshold

def Roc_Auc_Plot(results, legends=None, saved='roc_auc.png'):
    '''
    @Description: 
    @results: list('str')('str'=*.csv file with column = ['y_true', 'y_pred', 'y_score']) or list(np.array=['y_true','y_score']) 
    '''
    if isinstance(results[0], str):
        results = [pd.read_csv(result_) for result_ in results]
        results = [[np.array(df['y_true']), np.array(df['y_score'])] for df in results]

    plt.clf()
    plt.figure().set_size_inches(8,6)
    #fig = plt.figure(figsize=(12,10))
    fig, ax = plt.subplots()

    
    # Set offset for margin
    #left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    #ax = fig.add_axes([left,bottom,width,height])
    #ax.set_xlim(-0.05, 1.05)
    #ax.set_ylim(-0.05, 1.05)
    
    for ind_, result in enumerate(results):
        y_true = result[0]
        y_score = result[1]
        if legends:
            legend = legends[ind_]
            roc_plot_inside(y_true, y_score, ax, legend, positive=1)
        else:
            roc_plot_inside(y_true, y_score, ax, '',positive=1)

    ax.plot([0, 1], [0, 1], lw=2, color='grey', linestyle='--')
    
    # Set x, y-axis label and fontsize.
    ax.set_ylabel('Sensitivity')
    ax.set_xlabel('1-Specificity')
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)
    # set ticks
    Plt_ticks_set(ax, x_range=1.0, y_range=1.0)
    
    legend = ax.legend()
    fig.savefig(saved)
    return ax


# ====================================================================================================
# == Decision Curves Analysis (DCA)
# ====================================================================================================
def calculate_net_benefit_model(thresh_group, y_pred_score, y_label):
    net_benefit_model = np.array([])
    for thresh in thresh_group:
        y_pred_label = y_pred_score > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        n = len(y_label)
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit_model = np.append(net_benefit_model, net_benefit)
    return net_benefit_model


def calculate_net_benefit_all(thresh_group, y_label):
    net_benefit_all = np.array([])
    tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
    total = tp + tn
    for thresh in thresh_group:
        net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
        net_benefit_all = np.append(net_benefit_all, net_benefit)
    return net_benefit_all


def DCA_Plot(y_true, y_score, savefig = 'DCA.png', thresh_group = np.arange(0,1,0.01), legend_titles='Model', legend_titles1=['Treat all', 'Treat none']):
    plt.figure().set_size_inches(8,6)
    fig, ax = plt.subplots()
    y_true = np.array(y_true)
    if len(y_true.shape) == 1:
        net_benefit_model = calculate_net_benefit_model(thresh_group, y_score, y_true)
        net_benefit_all = calculate_net_benefit_all(thresh_group, y_true)
        ax.plot(thresh_group, net_benefit_model, color = 'crimson', label = legend_titles)
    else:
        for i in range(len(y_true)):
            net_benefit_model = calculate_net_benefit_model(thresh_group, y_score[i], y_true[i])
            net_benefit_all = calculate_net_benefit_all(thresh_group, y_true[i])
            ax.plot(thresh_group, net_benefit_model, label = legend_titles[i])
    ax.plot(thresh_group, net_benefit_all, color = 'black',label = legend_titles1[0])
    ax.plot((0, 1), (0, 0), color = 'black', linestyle = ':', label = legend_titles1[1])

    # #Fill，显示出模型较于treat all和treat none好的部分
    # y2 = np.maximum(net_benefit_all, 0)
    # y1 = np.maximum(net_benefit_model, y2)
    # ax.fill_between(thresh_group, y1, y2, color = 'crimson', alpha = 0.2)

    #Figure Configuration， 美化一下细节
    ax.set_xlim(0,1)
    ax.set_ylim(net_benefit_model.min() - 0.15, net_benefit_model.max() + 0.15)#adjustify the y axis limitation
    ax.set_xlabel(
        xlabel = 'Threshold Probability',
        fontdict= {'family': 'Times New Roman', 'fontsize': 20}
        )
    ax.set_ylabel(
        ylabel = 'Net Benefit',
        fontdict= {'family': 'Times New Roman', 'fontsize': 20}
        )
    # Ticks setting
    Plt_ticks_set(ax)
    # ax.grid('major')
    ax.spines['right'].set_color((0.8, 0.8, 0.8))
    ax.spines['top'].set_color((0.8, 0.8, 0.8))
    # ax.legend(loc = 'upper right')
    # legend set
    Plt_legend_set(ax, style=0)
    fig.savefig(savefig)
    return ax

def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point

def ROC_threshold(label, y_prob):
    """
    Receiver_Operating_Characteristic, ROC
    :param label: (n, )
    :param y_prob: (n, )
    :return: fpr, tpr, roc_auc, optimal_th, optimal_point
    """
    fpr, tpr, thresholds = metrics.roc_curve(label, y_prob)
    roc_auc = metrics.auc(fpr, tpr)
    optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
    return optimal_th, optimal_point

# ====================================================================================================
# == Calibration Curve
# ====================================================================================================
def Calibration_curve_plot(results, savefig='Calibration.png', labels=None, n_bins=10):
    '''
    Param: y_trues, True labels
    Param: y_scores, Model predicted scores
    n_bins: Number of bins to discretize the [0, 1] interval.
    '''
    plt.clf()
    plt.figure().set_size_inches(8,6)
    fig, ax = plt.subplots()


    # Creating Calibration Curve
    for i in range(len(results)):
        y_true = results[i][0]
        y_score = results[i][2]
        if labels:
            calibration_plot_inside(y_true, y_score, ax, label=labels[i], n_bins=n_bins)
        else:
            calibration_plot_inside(y_true, y_score, ax, n_bins=n_bins)
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins = n_bins, normalize = True)
    # Plot perfectly calibrated
    ax.plot([0, 1], [0, 1], lw=2, linestyle = '--', label = 'Ideally Calibrated')

    # Set x, y-axis label and fontsize.
    ax.set_ylabel('Ratio of positives')
    ax.set_xlabel('Predicted probability')
    # set ticks
    Plt_ticks_set(ax, x_range=1.0, y_range=1.0)
    legend = ax.legend()
    fig.savefig(savefig)
    return ax

def paths_sorted(paths):
    # return natsorted(paths, key = lambda x: x.name)
    return os_sorted(paths, key = lambda x: x.name)

# Define the Calibration Plot on the subplot(ax).
def calibration_plot_inside(y_true, y_score, ax, label='', n_bins=10):
    # Creating Calibration Curve
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins = n_bins, normalize = False, strategy='uniform')# 'quantile'
    # Plot model's calibration curve
    ax.plot(prob_pred, prob_true, lw=2, marker = '.', label = label)
    return ax

# ====================================================================================================
# == Lasso visualization
# ====================================================================================================
#这个图显示随着lambda的变化，系数的变化走势

def lasso_variable_trajectory(lassoCV_model, feas_df, title=None, savedir=None):
    col = 1 # 设置子图列数
    row = 1
    nfig = 0
    fig, ax = plt.subplots(row, col, figsize=(20,18*row/col))  # 通过"sharex='col', sharey='row'"参数设置共享行列坐标轴
    # 将ax统一成二维数组
    lassoCV_x = feas_df.drop(columns=['label'], axis=1)
    lassoCV_y = feas_df['label']
    alpha_range = np.logspace(-3, -1, 100, base=10)
    # 使用path的到alphas_,并不会影响之前训练的最优alpha_和coef_的值
    alphas1, coefs, _ = lassoCV_model.path(lassoCV_x,lassoCV_y, alphas=alpha_range, max_iter=10000)
    coefs = coefs.T
    alphas1 = np.log10(alphas1)
    _ = ax.plot(alphas1,coefs,'-');
    ax.axvline(np.log10(lassoCV_model.alpha_) , color='red', ls='--')  # dual_gap_
    ax.set_xlabel('Log Lambda')
    ax.set_ylabel('Coefficients')
    if title:
        ax.set_title(title, fontsize=20)
    if savedir:
        plt.savefig(os.path.join(savedir, title+'lasso_coefs.png'))

def lasso_variable_trajectory_multi(lassoCV_models, feas_dfs, titles=None, savefig=None):
    subfig_num = len(feas_dfs)
    cols = 4
    rows = subfig_num//cols
    if subfig_num % cols > 0:
        rows += 1
    fig, ax = plt.subplots(rows, cols, sharex='col', sharey='row',figsize=(cols*4, rows*4))  # 通过"sharex='col', sharey='row'"参数设置共享行列坐标轴
    if rows == 1:
        ax = np.array([ax])
    if cols == 1:
        ax = np.array([[x] for x in ax])
    alpha_range = np.logspace(-3, -1, 100, base=10)
    for i in range(subfig_num):
        model = lassoCV_models[i]
        feas_df = feas_dfs[i]
        lassoCV_x = feas_df.drop(columns=['label'], axis=1)
        lassoCV_y = feas_df['label']
        # 使用path的到alphas_,并不会影响之前训练的最优alpha_和coef_的值
        alphas1, coefs, _ = model.path(lassoCV_x,lassoCV_y, alphas=alpha_range, max_iter=10000)
        coefs = coefs.T
        alphas1 = np.log10(alphas1)
        _ = ax[i//cols, i%cols].plot(alphas1,coefs,'-');
        _ = ax[i//cols, i%cols].axvline(np.log10(model.alpha_) , color='red', ls='--')  # dual_gap_
        ax[i//cols, i%cols].set_xlabel('Log Lambda')
        ax[i//cols, i%cols].set_ylabel('Coefficients')
        if titles:
            ax[i//cols, i%cols].set_title(titles[i])
    for i in range(subfig_num, cols*rows):
        ax[i//cols, i%cols].set_xticks([])
        ax[i//cols, i%cols].set_yticks([])
        ax[i//cols, i%cols].axis('off')
    if savefig:
        plt.savefig(savefig)

def lasso_coefficient_screening_multi(lassoCV_models, feas_dfs, titles=None, savefig=None):
    subfig_num = len(feas_dfs)
    cols = 4
    rows = subfig_num//cols
    if subfig_num % cols > 0:
        rows += 1
    fig, ax = plt.subplots(rows, cols, sharex='col', sharey='row',figsize=(cols*4, rows*4))  # 通过"sharex='col', sharey='row'"参数设置共享行列坐标轴
    if rows == 1:
        ax = np.array([ax])
    if cols == 1:
        ax = np.array([[x] for x in ax])
        
    alpha_range = np.logspace(-3, -1, 100, base=10)
    for i in range(subfig_num):
        model = lassoCV_models[i]
        feas_df = feas_dfs[i]
        MSEs = model.mse_path_
        mse = list()
        std = list()
        for m in MSEs:
            mse.append(np.mean(m))
            std.append(np.std(m))
        ax[i//cols, i%cols].errorbar(np.log10(model.alphas_), mse, std,fmt='o:', ecolor='lightblue', elinewidth=3,ms=5,mfc='wheat',mec='salmon',capsize=3)
        ax[i//cols, i%cols].axvline(np.log10(model.alpha_), color='red', ls='--')
        ax[i//cols, i%cols].set_xlabel('Log Lambda',fontsize=20)
        ax[i//cols, i%cols].set_ylabel('MSE',fontsize=20)
        if titles:
            ax[i//cols, i%cols].set_title(titles[i])
    for i in range(subfig_num, cols*rows):
        ax[i//cols, i%cols].set_xticks([])
        ax[i//cols, i%cols].set_yticks([])
        ax[i//cols, i%cols].axis('off')
    if savefig:
        plt.savefig(savefig)
        
def lasso_coefficient_screening(lassoCV_model, feas_df, title=None, savefig=None):
    #绘制误差棒图
    col = 1
    row = 1
    fig, ax = plt.subplots(row, col, figsize=(20,18*row/col))  
    # 将ax统一成二维数组
    MSEs = lassoCV_model.mse_path_
    mse = list()
    std = list()
    for m in MSEs:
        mse.append(np.mean(m))
        std.append(np.std(m))
    ax.errorbar(np.log10(lassoCV_model.alphas_), mse, std,fmt='o:', ecolor='lightblue', elinewidth=3,ms=5,mfc='wheat',mec='salmon',capsize=3)
    ax.axvline(np.log10(lassoCV_model.alpha_), color='red', ls='--')
    ax.set_xlabel('Log Lambda',fontsize=20)
    ax.set_ylabel('MSE',fontsize=20)
    if savefig:
        plt.savefig(savefig)
        
def lasso_coefficient_weight(lassoCV_model, feas_df, title=None, savefig=None):
    col = 1
    row = 1
    fig, ax = plt.subplots(row, col, figsize=(20, 25) ) # 通过"sharex='col', sharey='row'"参数设置共享行列坐标轴
    #画一个特征系数的柱状图
    coefs = lassoCV_model.coef_
    coefs = coefs[coefs != 0]
    feas_list = feas_df.columns.to_list()
    feas_list.remove('label')
    coefs = pd.Series(lassoCV_model.coef_, index=feas_list)
    index = coefs[coefs !=0].index
    coefs = pd.Series(coefs, index=index)
    weight = coefs[coefs != 0].to_dict()
    #根据值大小排列一下
    weight = dict(sorted(weight.items(),key=lambda x:x[1],reverse=False))
    ax.set_xlabel(f' Weighted value')#设置x轴，并设定字号大小
    ax.set_ylabel(u'Features')
    ax.barh(range(len(weight.values())), list(weight.values()),tick_label = list(weight.keys()),alpha=0.6, facecolor = 'blue', edgecolor = 'black', label='feature weight')
    ax.legend(loc=0)#图例展示位置，数字代表第几象限
    if savefig:
        plt.savefig(savefig)
        
def lasso_coefficient_weight_multi(lassoCV_models, feas_dfs, titles=None, savefig=None):
    plt.clf()
    subfig_num = len(feas_dfs)
    cols = 2
    rows = subfig_num//cols
    if subfig_num % cols > 0:
        rows += 1
    fig, ax = plt.subplots(rows, cols, figsize=(cols*15, rows*10))  # 通过"sharex='col', sharey='row'"参数设置共享行列坐标轴
    if rows == 1:
        ax = np.array([ax])
    if cols == 1:
        ax = np.array([[x] for x in ax])
    for i in range(subfig_num):
        model = lassoCV_models[i]
        feas_df = feas_dfs[i]
        coefs = model.coef_
        coefs = coefs[coefs != 0]
        feas_list = feas_df.columns.to_list()
        feas_list.remove('label')
        coefs = pd.Series(model.coef_, index=feas_list)
        index = coefs[coefs !=0].index
        coefs = pd.Series(coefs, index=index)
        weight = coefs[coefs != 0].to_dict()
        #根据值大小排列一下
        weight = dict(sorted(weight.items(),key=lambda x:x[1],reverse=False))
        ax[i//cols, i%cols].set_xlabel(f' Weighted value')#设置x轴，并设定字号大小
        ax[i//cols, i%cols].set_ylabel(u'Features')
        ax[i//cols, i%cols].barh(range(len(weight.values())), list(weight.values()),tick_label = list(weight.keys()),alpha=0.6, facecolor = 'blue', edgecolor = 'black', label='feature weight')
        ax[i//cols, i%cols].legend()#图例展示位置，数字代表第几象限
        if titles:
            ax[i//cols, i%cols].set_title(titles[i])
    for i in range(subfig_num, cols*rows):
        ax[i//cols, i%cols].set_xticks([])
        ax[i//cols, i%cols].set_yticks([])
        ax[i//cols, i%cols].axis('off')
    if savefig:
        plt.savefig(savefig)

def coefficient_correlation_heat_map_multi(feas_dfs, titles=None, savefig=None):
    # 绘制特征相关系数热力图
    subfig_num = len(feas_dfs)
    cols = 3
    rows = subfig_num//cols
    if subfig_num % cols > 0:
        rows += 1
    # fig, ax = plt.subplots(rows, cols, sharex='col', sharey='row', figsize=(cols*4, rows*5))  
    fig, ax = plt.subplots(rows, cols, figsize=(cols*15, rows*13))  
    if rows == 1:
        ax = np.array([ax])
    if cols == 1:
        ax = np.array([[x] for x in ax])
        
    for i in range(subfig_num):
        feas_df = feas_dfs[i]
        _ = sns.heatmap(feas_df.corr(),annot=True,fmt='.2f',cmap='coolwarm',annot_kws={'size':6,'weight':'bold', },ax=ax[i//cols, i%cols])#绘制混淆矩阵
        _ = ax[i//cols, i%cols].set_xticklabels(ax[i//cols, i%cols].get_xticklabels(), fontdict={'size':8}, rotation=45,va="top",ha="right")
        ax[i//cols, i%cols].set_yticklabels(ax[i//cols, i%cols].get_yticklabels(), fontdict={'size':8}, rotation=45, va='top', ha='right')
        ax[i//cols, i%cols].tick_params(bottom=False, top=False, left=False, right=False)
        if titles:
            ax[i//cols, i%cols].set_title(titles[i])
    # Delete the frame of empty figure
    for i in range(subfig_num, cols*rows):
        ax[i//cols, i%cols].set_xticks([])
        ax[i//cols, i%cols].set_yticks([])
        ax[i//cols, i%cols].axis('off')
    if savefig:
        plt.savefig(savefig)

def coefficient_correlation_heat_map(feas_df, title=None, savefig=None):
    # 绘制特征相关系数热力图
    # The fusion features
    fig, ax= plt.subplots(figsize = (15, 13),dpi=300)
    _ = sns.heatmap(feas_df.corr(),annot=True,fmt='.2f',cmap='coolwarm',annot_kws={'size':6,'weight':'bold', },ax=ax)#绘制混淆矩阵
    _ = ax.set_xticklabels(ax.get_xticklabels(), fontdict={'size':8}, rotation=45,va="top",ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), fontdict={'size':8}, rotation=45, va='top', ha='right')
    ax.tick_params(bottom=False, top=False, left=False, right=False)
    if title:
        ax.set_title(title)
    if savefig:
        plt.savefig(savefig)
# ====================================================================================================
# == visulization class for results of machine learning.
# ====================================================================================================
class ModelResultsEvaluation:
    def __init__(self, input_data, name=None):
        '''
        @params: input_data, table of columns['pid','y_true','y_pred','y_score0','y_score1',...,'dataset']
        '''
        if isinstance(input_data, (str, pathlib.PosixPath)):
            # Load data by input path.
            self._all_data = pd.read_csv(input_data)
            self._name = pathlib.Path(input_data).with_suffix('').name
        else:
            # Load data by pd.DataFrame
            self._all_data = input_data
            self._name = ''
        self._datasets = list(self._all_data['dataset'].value_counts().index)
        self._trainplus = [x for x in self._datasets if x in ['train', 'validation']]
        self._testplus = [x for x in self._datasets if x not in ['train', 'validation']]
        self._evaluation = self.__evaluation_init()
        
    def get_score(self, dataset='test', score='y_score1'):
        if dataset == 'train+':
            dataset = ['train', 'validation']
        if isinstance(dataset, str):
            data = self._all_data[self._all_data['dataset']==dataset]
        else:
            data = self._all_data[self._all_data['dataset'].isin(dataset)]
        return np.array(data[score])
    
    def get_dataset(self, dataset='test'):
        if isinstance(dataset, str):
            if dataset == 'train+':
                data = self._all_data[self._all_data['dataset'].isin(self._trainplus)]
            elif dataset == 'test+':
                data = self._all_data[self._all_data['dataset'].isin(self._testplus)]
            else:    
                data = self._all_data[self._all_data['dataset'] == dataset]
        else:
            data = self._all_data[self._all_data['dataset'].isin(dataset)]
        return data
        
    def get_distribution(self):
        tmp_data = self._all_data.rename(columns={'y_true':'label'})
        dataset = pd.crosstab(tmp_data['label'], tmp_data['dataset'], margins=True)
        columns = list(dataset.columns)
        full_list = ['train','validation','test','external','public','All'] 
        tmp_list = [x for x in full_list if x in columns]
        return dataset[tmp_list]
    
    def plot_confusion_matrix(self, dataset='test', savefig=None, fig_title=None, ax=None):
        '''
        :Description: To draw the  confusion-matrix based on the truth and predicted label.
        :
        :savefig: path of output fig.
        '''
        y_true = self.get_score(dataset, 'y_true')
        y_pred = self.get_score(dataset, 'y_pred')
        if fig_title is None:
            fig_title = self.name
        C = confusion_matrix(y_true, y_pred)
        C = C.astype(int)
        df=pd.DataFrame(C)
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(df, fmt='g', annot=True, cmap='Blues', ax=ax, annot_kws={"size":20})
        ax.set_xlabel('Predict') 
        ax.set_ylabel('True') 
        ax.set_xticks([])
        ax.set_yticks([])
        cax = plt.gcf().axes[-1] 
        cbar = ax.collections[0].colorbar
        if fig_title:
            ax.set_title(fig_title)
        if savefig:
            plt.savefig(savefig)
        return ax
    
    def plot_roc_auc(self, dataset=['train+', 'test', 'external'], savefig=None, legends=None, title=None):
        '''
        '''
        results = [[self.get_score(x, 'y_true') , self.get_score(x, 'y_score1')] for x in dataset]
        if legends is None:
            legends = ['training' if x == 'train+' else x for x in dataset]
        plt.clf()
        fig, ax = plt.subplots(figsize=(10,8))
        for ind_, result in enumerate(results):
            y_true = result[0]
            y_score = result[1]
            legend = legends[ind_]
            roc_plot_inside(y_true, y_score, ax, legend, positive=1)
    
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
        
        # Set x, y-axis label and fontsize.
        ax.set_ylabel('Sensitivity')
        ax.set_xlabel('1-Specificity')
        if title == 'name':
            title = self._name
        ax.set_title(title)
        # set ticks
        _ = Plt_ticks_set(ax, x_range=1.0, y_range=1.0)
    
        legend = ax.legend()
        if savefig:                       
            fig.savefig(savefig)
        return ax
    
    def plot_roc_auc1(self, dataset='test', savefig=None, legends=None, title=None):
        # 计算每一类的ROC
        df = self.get_dataset(dataset)
        y_test = df['y_true']
        n_classes = len(y_test.value_counts())
        y_score = np.array(df.iloc[:,3:-1])
        y_test = np.array(pd.get_dummies(y_test))
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        #    roc_auc_low, roc_auc_up = roc_auc_ci(y_test, y_score, positive=1)
        
        # Compute micro-average ROC curve and ROC area（方法二）
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Compute macro-average ROC curve and ROC area（方法一）
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        # Finally average it and compute AUC
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        if legends:
            legend_titles = legends
        else:
            legend_titles = [x for x in range(len(y_score))]
        # Plot all ROC curves
        plt.clf()
        fig = plt.figure(figsize=(10,8))
        
        # Set offset for margin
        left, bottom, width, height = 0.10, 0.10, 0.8, 0.8
        ax = fig.add_axes([left,bottom,width,height])
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        
        
        ax.plot(fpr["micro"], tpr["micro"],
                 label='ROC curve of micro-average (AUC = {0:0.3f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)
        
        ax.plot(fpr["macro"], tpr["macro"],
                 label='ROC curve of macro-average (AUC = {0:0.3f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)
        
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            roc_plot_inside(y_test[:,i], y_score[:,i], label=legend_titles[i], ax=ax)
            # plt.plot(fpr[i], tpr[i], color=color, lw=lw,
            #          label='ROC curve of class {0} (AUC = {1:0.3f})'
            #          ''.format(legend_titles[i], roc_auc[i]))
            
        # set ticks
        Plt_ticks_set(ax, x_range=1.0, y_range=1.0)
        
        
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('1-Specificity')
        ax.set_ylabel('Sensitivity')
        ax.legend(loc="lower right")
        if title == 'name':
            title = self._name
        ax.set_title(title)
        if savefig:                       
            fig.savefig(savefig)
    
    
    def get_metrics(self, metrics=['auc', 'accuracy', 'recall', 'specificity', 'f1'], dataset=['train+','test','external'], type='binary'):
        if type == 'multiclass':
            metrics = ['accuracy', 'kappa']
        for x in dataset:
            self.evaluation_update(metrics, x)
        cols = ['data'] + metrics
        eval_table = pd.DataFrame(columns=cols)
        for x in dataset:
            metric_dic = self.evaluation[x].copy()
            metric_dic['data'] = x
            eval_table = eval_table.append(metric_dic, ignore_index=True)
        return eval_table
    
    
    def evaluation_update(self, metric, dataset):
        if isinstance(metric, str) and isinstance(dataset, str):
            self.__evaluation_update1(metric, dataset)
        elif isinstance(metric, list):
            for x in metric:
                self.evaluation_update(x, dataset)
        elif isinstance(dataset, list):
            for x in dataset:
                self.evaluation_update(metric, x)
    
    def __evaluation_update1(self, metric, dataset):
        eva_part = self._evaluation[dataset]
        if metric in eva_part.keys():
            return
        eval_data = self.get_dataset(dataset)
        y_true = eval_data['y_true']
        y_pred = eval_data['y_pred']
        y_score = eval_data['y_score1']
        if metric == 'auc':
            eva_part[metric] = metrics.roc_auc_score(y_true, y_score)
        elif metric == 'precision':
            eva_part[metric] = metrics.precision_score(y_true, y_pred)
        elif metric == 'accuracy':
            eva_part[metric] = metrics.accuracy_score(y_true, y_pred)
        elif metric == 'recall':
            eva_part[metric] = metrics.recall_score(y_true, y_pred)
        elif metric == 'f1':
            eva_part[metric] = metrics.f1_score(y_true, y_pred)
        elif metric == 'specificity':
            eva_part[metric] = specificity_score(y_true, y_pred)
        elif metric == 'kappa':
            eva_part[metric] = quadratic_weighted_kappa(y_true, y_pred)
        
    def __evaluation_init(self):
        evaluation = {x:{} for x in self._datasets}
        evaluation['train+'] = {}
        evaluation['test+'] = {}
        return evaluation
    
    @property
    def name(self):
        return self._name
    
    @property
    def evaluation(self):
        return self._evaluation
        
    
class MultiModelResultsEvaluation:
    def __init__(self, input_data):
        '''
        param: input_data, list of ModelResultsEvaluation or str.
        '''
        if isinstance(input_data[0], str):
            self._results = [ModelResultsEvaluation(x) for x in input_data]
        elif isinstance(input_data[0], ModelResultsEvaluation):
            self._results = input_data
        self._names = [x.name for x in input_data]
        
    def plot_roc_auc(self, legends=None, dataset='test', savefig='./roc_auc.png', title=None):
        '''
        @Description: 
        @results: list('str')('str'=*.csv file with column = ['y_true', 'y_pred', 'y_score0', 'y_score1', ...]) or list(np.array=['y_true','y_score1']) 
        '''
        results = [[x.get_score(dataset, 'y_true') , x.get_score(dataset, 'y_score1')] for x in self._results]
        if legends is None:
            legends = self._names
        plt.clf()
        fig, ax = plt.subplots(figsize=(10,8))
        for ind_, result in enumerate(results):
            y_true = result[0]
            y_score = result[1]
            legend = legends[ind_]
            roc_plot_inside(y_true, y_score, ax, legend, positive=1)
    
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
        # Set x, y-axis label and fontsize.
        ax.set_ylabel('Sensitivity')
        ax.set_xlabel('1-Specificity')
        ax.set_title(title)
        # set ticks
        _ = Plt_ticks_set(ax, x_range=1.0, y_range=1.0)
        legend = ax.legend()
        fig.savefig(savefig)
        return ax
                               
    def plot_confusion_matrix_multi(self, dataset='test', savefig=None, fig_titles=None):
        subfig_num = self.length
        cols = 4
        rows = subfig_num//cols
        if fig_titles == 'name':
            fig_titles = [x.name for x in self._results]
        if subfig_num % cols > 0:
            rows += 1
        fig, ax = plt.subplots(rows, cols, sharex='col', sharey='row',figsize=(cols*3, rows*3))  # 通过"sharex='col', sharey='row'"参数设置共享行列坐标轴
        if rows == 1:
            ax = np.array([ax])
        if cols == 1:
            ax = np.array([[x] for x in ax])
        for i in range(subfig_num):
            C = confusion_matrix(self._results[i].get_score(dataset, score='y_true'), self._results[i].get_score(dataset, score='y_pred'))
            df = pd.DataFrame(C.astype(int))
            sns.heatmap(df, fmt='g', annot=True, cmap='Blues', ax=ax[i//cols, i%cols], annot_kws={"size":20})
            # ax[i//cols, i%cols].axis('off')     #不显示坐标尺寸
            ax[i//cols, i%cols].set_xticks([])
            ax[i//cols, i%cols].set_yticks([])
            ax[i//cols, i%cols].set_ylabel('True')
            ax[i//cols, i%cols].set_xlabel('Predict')
            if fig_titles:
                ax[i//cols, i%cols].set_title(fig_titles[i])
        fig.suptitle(dataset,fontsize=20)        
        if savefig:
            plt.savefig(savefig)
        return ax
    
    def plot_roc_auc_multi(self, dataset=['train+', 'test', 'external'], savefig='./roc_auc.png', legends=None, title=None):
        subfig_num = self.length
        cols = 3
        rows = subfig_num//cols
        if title == 'name':
            title = [x.name for x in self._results]
        if legends is None:
            legends = ['training' if x == 'train+' else x for x in dataset]
        if subfig_num % cols > 0:
            rows += 1
        fig, ax = plt.subplots(rows, cols, figsize=(cols*6, rows*6), constrained_layout=True)  # 通过"sharex='col', sharey='row'"参数设置共享行列坐标轴
        if rows == 1:
            ax = np.array([ax])
        if cols == 1:
            ax = np.array([[x] for x in ax])
        for i in range(subfig_num):
            results = [[self._results[i].get_score(x, 'y_true') , self._results[i].get_score(x, 'y_score1')] for x in dataset]
            for ind_, result in enumerate(results):
                y_true = result[0]
                y_score = result[1]
                legend = legends[ind_]
                roc_plot_inside(y_true, y_score, ax[i//cols, i%cols], legend, positive=1)
            ax[i//cols, i%cols].plot([0, 1], [0, 1], color='grey', linestyle='--')
            ax[i//cols, i%cols].set_ylabel('Sensitivity')
            ax[i//cols, i%cols].set_xlabel('1-Specificity')
            if title:
                ax[i//cols, i%cols].set_title(title[i])
            ax[i//cols, i%cols].legend(fontsize=12)
        # handles, labels = ax[0, 0].get_legend_handles_labels()
        # fig.legend(handles, labels, fontsize='x-large', bbox_to_anchor=(1.05, 1), loc='upper left')#, loc="upper left", bbox_to_anchor=(0.65,1.05))
        subplots_remove_residual(ax, subfig_num, rows, cols)
        fig.suptitle('ROC_AUC',fontsize=20)
        if savefig:
            plt.savefig(savefig)
        return ax
    
    def evaluation(self, metrics=['auc', 'accuracy', 'recall', 'specificity', 'f1'], dataset='test'):
        for x in self._results:
            x.evaluation_update(metrics, dataset)
        cols = ['data'] + metrics
        eval_table = pd.DataFrame(columns=cols)
        for x in self._results:
            metric_dic = x.evaluation[dataset].copy()
            metric_dic['data'] = x.name
            eval_table = eval_table.append(metric_dic, ignore_index=True)
        return eval_table
    
    def delong_test(self, dataset='test'):
        """ Delong test for the multi-models
        
        param: dataset, str, dataset
        """
        score_list = [x.get_score(dataset=dataset, score='y_score1') for x in self._results]
        labels = self._results[0].get_score(dataset=dataset, score='y_true') 
        name_list = [x.name for x in self._results]
        test_df = pd.DataFrame(columns=name_list)
        for i, s1 in enumerate(score_list):
            pdict={}
            for j, s2 in enumerate(score_list):
                p_value = DelongTest(s1, s2, labels)._compute_z_p()[1]
                p_value = round(p_value, 4)
                pdict[name_list[j]] = p_value
            test_df = test_df.append(pdict, ignore_index=True)
        test_df.set_index(pd.Series(name_list), inplace=True)
        return test_df
                
        
           
    @property
    def length(self):
        return len(self._results)
