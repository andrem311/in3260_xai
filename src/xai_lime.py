import joblib
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

DATAPATH_REAL = "data/real_data_windows.csv"
DATAPATH_TEST = "data/synthetic_network_system_hard.csv"
DATAPATH = "data/synthetic_network_system_short.csv"
LABEL_COL = "is_event"
REAL = 0
#works
def main():
    rf = joblib.load("models/rf.joblib")
    # we look at the logistic regression instead since this xai(might not be correct) does better with differentiable models
    # rather than non-differentiable ones. 
    lr = joblib.load("models/lr.joblib") 
    features = joblib.load("models/features.joblib")
    df = pd.read_csv(DATAPATH)
    X = df[features].values  

    df_real = pd.read_csv(DATAPATH)
    y = df_real[LABEL_COL].values.astype(int)
    if(REAL == 1):
        df_real = pd.read_csv(DATAPATH_REAL)
        y = []
    X_REAL = df_real[features].values

    explainer = LimeTabularExplainer(training_data=X,
                                     feature_names=features,class_names=["normal","anomaly"],
                                     mode="classification",discretize_continuous=True)
    

    # exp = explainer.explain_instance(data_row=X[idx], predict_fn=rf.predict_proba, num_features=7)
    List_exp = []
    start_time = time.time()
    for row in X_REAL:
        #to keep it simple, it is simply the matter of commenting in and out to check either random forest or lr
        #random forrest is a lot slower than what logistic regression is
        # exp = explainer.explain_instance(data_row=row, predict_fn=rf.predict_proba, num_features=7)
        exp = explainer.explain_instance(data_row=row, predict_fn=lr.predict_proba, num_features=7)
        List_exp.append(exp)
    end_time = time.time()

    print("time for lime: ", end_time-start_time)
    # exp_m = exp.as_map()
    # print(type(exp_m[1][0]))
    print("start the ordering")
    #we do this costly operation to reorder it corretly for the eval_driver, where we use the scores 
    list_all_instances = []
    for exp_x in List_exp:
        list_order = []
        # Turn the explanation object to a map so we can easily work on it.
        exp_m = exp_x.as_map()
        # Seven indexes that need to be sorted
        for index in range(0,7):
            #access the tuples inside
            for idex_2 in exp_m[1]:
                #look at the first element to see where it should go
                if idex_2[0] == index:
                    #get the item, so no np values, but rather normal floats 
                    list_order.append((idex_2[1].item()))
        #apppend one row/minute to the list            
        list_all_instances.append(list_order)
    
    np.save("outputs/lime_scores.npy",list_all_instances)

    print(f"\n[LIME] Global Explenation(MEAN):")
    mean_of_lime = np.mean(np.abs(list_all_instances), axis=0)
    order = reversed(np.argsort(mean_of_lime))
    for j in order:
        print(f"{features[j]}: {mean_of_lime[j]:.4f}")

    idx = 5
    #if the it is the testing set we look up.
    if len(y) != 0:
        idx = int(np.where(y==1)[0][0])

    fig = List_exp[idx].as_pyplot_figure()
    # fig
    #for running it in the shell of linux on venv, or else it will open and close for a short period of time
    #for saving it is best with some manual editing as the proportions are not correct when simply saving normally
    plt.show(block=True)
    #for directly saving image, (proportions not correct, most likely)
    # fig.savefig('lime_report.jpg')

    print(f"\n[LIME] Explanation for sample idx={idx}:")
    print(List_exp[idx].as_list())
    for feat, w in List_exp[idx].as_list():
        print(f"{feat}: weight={w:.4f}")
    

if __name__ == "__main__":
    main()



