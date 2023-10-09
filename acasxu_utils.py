import onnxruntime
import numpy as np 
import scipy.stats as stat
import torch


def onnx_to_model(net_file):
    sess = onnxruntime.InferenceSession(net_file)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    def out_model(X):
        if len(X.shape)<=1:
            logits= sess.run([label_name], {input_name: X.astype(np.float32)})[0]
        else:
            logits_ = []
            for x in X:
                logits= sess.run([label_name], {input_name: x.astype(np.float32)})[0]
                logits_.append(logits)
            logits = np.array(logits_)
        return logits
    return out_model






def read_acasxu_input_file(input_file, verbose=1):
    if verbose>= 1:
        print('Reading input constraints file')
        count =0
    input_constraint = open(input_file,'r')
    input_pre = []
    cur_line = input_constraint.readline().strip('\n')
    while len(cur_line.strip('[]'))>1:
        if verbose>=1:
            print(f"x{count} in {cur_line}")
            count+=1
        x = [np.float32(e) for e in cur_line.strip('[]').split(',')]
        input_pre.append(x)
        cur_line = input_constraint.readline().strip('\n')
    return np.array(input_pre)

ACAXUmeans = np.array([1.9791091e+04, 0.0, 0.0, 650.0, 600.0])
ACAXUstds = np.array([60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0])

def input_transformer_acasxu(X, input_con, from_gaussian=True, normalize=False):
    if from_gaussian:
        X=stat.norm.cdf(X)
    U=np.zeros(shape=X.shape)
    if len(X.shape)>1:
        U= input_con[None,:,0]+(input_con[None,:,1]-input_con[None,:,0])*X
        if normalize:
            U = (U-ACAXUmeans[None, :])/ACAXUstds[None, :]
    else:
        U = input_con[:,0]+(input_con[:,1]-input_con[:,0])*X 
        if normalize:
            U = (U-ACAXUmeans)/ACAXUstds

    
    return U



acasxu_prop1_score_1 = lambda y: y[0]-3.991125
acasxu_prop1_score_b = lambda y: y[:,0]-3.991125
acasxu_prop1_score = lambda y: acasxu_prop1_score_b(y) if len(y.shape)>1 else acasxu_prop1_score_1(y)

acasxu_prop2_score_1 = lambda y: y[0] -np.max(np.delete(y,[0]))
acasxu_prop2_score_b = lambda y: y[:,0]- np.max(np.delete(y,[0], axis=1), axis =1)
acasxu_prop2_score = lambda y: acasxu_prop2_score_b(y) if len(y.shape)>1 else acasxu_prop2_score_1(y)

acasxu_prop3_score_1 = lambda y: np.min(np.delete(y,[0]))-y[0]
acasxu_prop3_score_b = lambda y: np.min(np.delete(y,[0], axis=1), axis =1)-y[:,0]
acasxu_prop3_score = lambda y: acasxu_prop3_score_b(y) if len(y.shape)>1 else acasxu_prop3_score_1(y)

acasxu_prop4_score = acasxu_prop3_score

acasxu_prop5_score_1 = lambda y: y[4]-np.min(np.delete(y,[4]))
acasxu_prop5_score_b = lambda y: y[:,4]-np.min(np.delete(y,[4], axis=1), axis =1)
acasxu_prop5_score = lambda y: acasxu_prop5_score_b(y) if len(y.shape)>1 else acasxu_prop5_score_1(y)

acasxu_prop6_score_1 = lambda y: y[0]-np.min(np.delete(y,[0]))
acasxu_prop6_score_b = lambda y: y[:,0]-np.min(np.delete(y,[0], axis=1), axis =1)
acasxu_prop6_score = lambda y: acasxu_prop6_score_b(y) if len(y.shape)>1 else acasxu_prop6_score_1(y)


acasxu_temp_score = lambda y: np.min(np.delete(y,[3]))-y[3]
acasxu_temp_score_b = lambda y: np.min(np.delete(y,[3], 1),1)-y[:,3]

acasxu_prop7_score_1 = lambda y: np.max([-acasxu_prop5_score(y),acasxu_temp_score(y)])
acasxu_prop7_score_b = lambda y: np.max(np.concatenate([-acasxu_prop5_score(y)[:,None],acasxu_temp_score_b(y)[:,None]],1),1)
acasxu_prop7_score = lambda y: acasxu_prop7_score_b(y) if len(y.shape)>1 else acasxu_prop7_score_1(y)

acasxu_temp_score = lambda y: y[1]-np.min(np.delete(y,[1]))
acasxu_temp_score_b = lambda y: y[:,1]-np.min(np.delete(y,[1], axis=1), axis =1)

acasxu_prop8_score_1 = lambda y: np.min(acasxu_prop6_score_1(y),acasxu_temp_score(y))
acasxu_prop8_score_b = lambda y: np.min(np.concatenate([acasxu_prop6_score(y)[:,None],acasxu_temp_score_b(y)[:,None]],1),1)
acasxu_prop8_score = lambda y: acasxu_prop8_score_b(y) if len(y.shape)>1 else acasxu_prop8_score_1(y)


acasxu_prop9_score_1 = lambda y: y[3]-np.min(np.delete(y,[3]))
acasxu_prop9_score_b = lambda y: y[:,3]-np.min(np.delete(y,[3], axis=1), axis =1)
acasxu_prop9_score = lambda y: acasxu_prop9_score_b(y) if len(y.shape)>1 else acasxu_prop9_score_1(y)

acasxu_prop10_score = acasxu_prop6_score

acasxu_scores = [acasxu_prop1_score, acasxu_prop2_score, 
acasxu_prop3_score, acasxu_prop4_score,acasxu_prop5_score, acasxu_prop6_score, 
acasxu_prop7_score,acasxu_prop8_score, acasxu_prop9_score, 
acasxu_prop10_score]
'Clear-of-Conflict (COC), weak right, strong right, weakleft, or strong left'