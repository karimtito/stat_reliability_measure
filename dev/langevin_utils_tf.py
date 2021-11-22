import tensorflow as tf

def TimeStepTF(V,X,gradV,p=1):
    V_mean=tf.reduce_mean(V(X))
    V_grad_norm_mean = (tf.reduce_mean(tf.norm(gradV(X),axis = 1)**p))**(1/p)
    return V_mean/V_grad_norm_mean



def langevin_kernel_tf(X,gradV,delta_t,beta):
    G_noise = tf.random.normal(X.shape)
    X_new =X-delta_t*gradV(X)+tf.math.sqrt(2*delta_t/beta)*G_noise
    return X_new

def mult_expand_dims(x,k=2):
    for _ in range(k):
        x= tf.expand_dims(x,- 1)
    return x


def compute_V_grad_tf(model, input, target_class,mask_value=-1e-50):
    """ Returns potentials and potentials gradients for given input, model and target classes """
    with tf.GradientTape() as tape:
        tape.watch(input)
        logits = model(input) 
        val, ind= tf.math.top_k(logits,k=2)
        output=val[:,0]-val[:,1]

    a_priori_grad=tape.gradient(output,input)
    mask=tf.equal(ind[:,0],target_class)
    v=tf.where(condition=mask, x=output,y=tf.zeros(output.shape))
    mask=mult_expand_dims(mask,k=len(input.shape[1:]))
    grad=tf.where(condition=mask, x=a_priori_grad,y=tf.zeros(a_priori_grad.shape))
    return v,grad

def compute_V_tf(model, input, target_class):
    logits = model(input) 
    val, ind= tf.math.top_k(logits,k=2)
    output=val[:,0]-val[:,1]
    mask=tf.equal(ind[:,0],target_class)
    v=tf.where(condition=mask, x=output,y=tf.zeros(output.shape))
    return v 

def compute_h_tf(model, input, target_class):
    logits = model(input) 
    val, ind= tf.math.top_k(logits,k=2)

    output_pos=logits[:,target_class]-val[:,0]
    output_neg=val[:,1]-val[:,0]
    mask=tf.equal(ind[:,0],target_class)
    v=tf.where(condition=mask, x=output_neg,y=output_pos)
    return v 