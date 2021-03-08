import pylab as pl
from IPython import display
fig = pl.gcf()

def plot_roc_curve_aux(n_classes, pred, y_test):
  from itertools import cycle
  import numpy as np
  from sklearn.preprocessing import LabelBinarizer

  lb = LabelBinarizer()
  y_test2 = lb.fit_transform(y_test)
  y_test2 = np.stack((y_test2,)*1, axis=-1)

  pred2 = lb.fit_transform(pred)
  pred2 = np.stack((pred2,)*1, axis=-1)

  from sklearn.metrics import roc_curve, auc
  # Compute ROC curve and ROC area for each class
  fpr = dict()
  tpr = dict()
  roc_auc = dict()
  try:
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test2[:,i], pred2[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
      # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test2.ravel(), pred2.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

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
  except:
    return

  return fpr["macro"], tpr["macro"], roc_auc["macro"], fpr["micro"], tpr["micro"], roc_auc["micro"]

def plot_roc_curve(n_classes, pred, y_test):
  from scipy import interp
  import numpy as np
  from matplotlib import pyplot as plt
  from statistics import mean
  from sklearn.metrics import roc_curve, auc
  mean_fpr = np.linspace(0, 1, 100)
  tprs = []
  aucs = []
  erro = False
  plt.figure()
  fig, ax = plt.subplots()
  for i in range(pred.shape[0]):
      pred2 = np.frombuffer(pred[i], dtype=np.int)
      y_test2 = np.frombuffer(y_test[i], dtype=np.int)
      try:
        fpr_macro, tpr_macro, roc_auc_macro, fpr_micro, tpr_micro, roc_auc_micro = plot_roc_curve_aux(n_classes, pred2, y_test2)
        erro = False
        interp_tpr = np.interp(mean_fpr, fpr_macro, tpr_macro)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc_macro)
      except:
        erro = True
  if erro == False:
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color='b',
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')

        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
              title="Receiver operating characteristic example")
        ax.legend(loc="lower right")
        #plt.show()
        #display.display(pl.gcf())

  return
