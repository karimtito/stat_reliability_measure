import tensorflow as tf
import tensorflow_probability as tfp
def mult_expand_dims(x,k=2):
    for _ in range(k):
        x= tf.expand_dims(x,- 1)
    return x

dic_in_shape_tf={'mnist':(28,28,1),'cifar10':(32,32,3),
'cifar100':(32,32,3),'imagenet':(224,224,3)}

def compute_V_grad_tf(model, input_, target_class):
    """ Returns potentials and potentials gradients for given input_, model and target classes """
    with tf.GradientTape() as tape:
        tape.watch(input_)
        logits = model(input_) 
        val, ind= tf.math.top_k(logits,k=2)
        output=val[:,0]-val[:,1]

    a_priori_grad=tape.gradient(output,input_)
    mask=tf.equal(ind[:,0],target_class)
    v=tf.where(condition=mask, x=output,y=tf.zeros(output.shape))
    mask=mult_expand_dims(mask,k=len(input_.shape[1:]))
    grad=tf.where(condition=mask, x=a_priori_grad,y=tf.zeros(a_priori_grad.shape))
    return v,grad

def compute_V_tf(model, input_, target_class):
    logits = model(input_) 
    val, ind= tf.math.top_k(logits,k=2)
    output=val[:,0]-val[:,1]
    mask=tf.equal(ind[:,0],target_class)
    v=tf.where(condition=mask, x=output,y=tf.zeros(output.shape))
    return v 

def compute_h_tf(model, input_, target_class):
    logits = model(input_) 
    val, ind= tf.math.top_k(logits,k=2)

    output_pos=logits[:,target_class]-val[:,0]
    output_neg=val[:,1]-val[:,0]
    mask=tf.equal(ind[:,0],target_class)
    v=tf.where(condition=mask, x=output_neg,y=output_pos)
    return v 


norm_dist_tf = tfp.distributions.Normal