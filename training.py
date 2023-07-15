import argparse
import json
import time 

import tensorflow as tf 
import tensorflow_probability as tfp
import numpy as np 
tfd = tfp.distributions

from kalman_ruls.data.dataprep import DataPrep
from kalman_ruls.networks.DVAE import Kalman_DVAE
from kalman_ruls.networks.transition_models import *
from kalman_ruls.networks.measurement_models import * 
from kalman_ruls.networks.encoders import * 

def str2bool(v):
    """
    Used for boolean arguments in argparse; avoiding `store_true` and `store_false`.
    """
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

class RunningAverage():
    """
    Helper class for calculating the running average of training losses 
    """
    def __init__(self):
        self.running_losses = {}
        self.names = [] 

    def add_key(self, names):
        # names are a list of strings for the dictionary keys containing losses
        for name in names:
            self.running_losses[name] = []
            self.running_losses[name + "_avg"] = 0 
            self.running_losses[name + "_std"] = 0
            
            self.names.append(name)

    def add_loss(self, loss, name):
        self.running_losses[name].append(loss)
        
    def avg_loss(self):
        for name in self.names:
            avg = np.array(self.running_losses[name]).mean()
            self.running_losses[name + "_avg"] = avg 

    def std_loss(self):
        for name in self.names:
            std = np.array(self.running_losses[name]).std()
            self.running_losses[name + "_std"] = std

    def reset_all(self):
        # reset all losses and counters to 0 
        for name in self.running_losses:
            self.running_losses[name] = [] 

    def get_avg_loss(self, name):
        return self.running_losses[name + "_avg"]

    def get_std_loss(self, name):
        return self.running_losses[name + "_std"]

# Actual class for training models 
class Trainer():
    def __init__(self, lr, alpha):
        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.alpha = alpha 

    @tf.function
    def train_step(self, model: Kalman_DVAE, x, r, optimizer, replay, alpha, elbo):
        with tf.GradientTape() as tape:
            if elbo:
                loss = model.get_ELBO(x, r)
            else:
                loss = model.get_loss(x, r, replay, alpha)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss 

    @tf.function
    def valid_step(self, x, tgt, model):
        fzs, fzPs, r_mean, r_var = model(x)
        dists = tfd.MultivariateNormalTriL(r_mean, tf.linalg.cholesky(r_var))
        valid_nll = -tf.reduce_sum(dists.log_prob(tgt), axis=1)
        valid_nll = tf.reduce_mean(valid_nll)
        return valid_nll

    def train_model(self, epochs, train_loader, valid_loader, model, model_PATH, replay=True, elbo=False):
        best_loss = 1e10
        logger = RunningAverage()
        logger.add_key(["train_nll", "valid_nll"])
        for epoch in range(1, epochs+1):
            logger.reset_all() # resets running average losses to zero

            # --- Training --- 
            for x, tgt in train_loader: 
                loss = self.train_step(model, x, tgt, self.optimizer, replay, self.alpha, elbo)     # need a running average loss instead of using the last loss for printing to user 
                logger.add_loss(float(loss.numpy()), "train_nll")  
            
            # --- Validation --- 
            if (epoch % 10 == 0) or (epoch == 1):
                for x, tgt in valid_loader:
                    valid_nll = self.valid_step(x, tgt, model)
                    logger.add_loss(float(valid_nll.numpy()), "valid_nll")    # running loss 

                # take the average of the stored losses 
                logger.avg_loss()

                # get average losses 
                valid_loss = logger.get_avg_loss("valid_nll")
                train_loss = logger.get_avg_loss("train_nll")
                
                # --- Save Model and Report Loss --- 
                if float(valid_loss) < best_loss:
                    model.save_weights(model_PATH) 
                    best_loss = float(valid_loss)

                    message = "new best loss, saving model in " + model_PATH + " ..."

                else: 
                    message = ""

                print(("Epoch: {}/{}, nll loss: {:.4f}, valid nll: {:.4f}, " + message)
                .format(epoch, epochs, train_loss, valid_loss))

def prep_data(PATH, dataset, T, bs=150, max_rul=130.):
    prep_class = DataPrep(PATH, dataset)

    if dataset == "FD001" or dataset == "FD003":
        prep_class.op_normalize(K=1)    # K=1 normalization, K=6 operating condition norm 
    else: 
        prep_class.op_normalize(K=6) 

    x_train, y_train, t_train = prep_class.prep_data(prep_class.ntrain, T, max_rul)
    x_train, y_train, t_train, x_valid, y_valid, t_valid = prep_class.valid_set(x_train, y_train, t_train)
    x_test, y_test, t_test = prep_class.prep_test(prep_class.ntest, prep_class.RUL, max_rul)

    train_loader, valid_loader = prep_class.get_dataloaders(bs, x_train, t_train, y_train, x_valid, t_valid, y_valid)
    return train_loader, valid_loader, x_test, y_test, t_test

def model_selector(name:str, zdim, rdim=None, transition=True, **kwargs):
    """
    Given a model name this will select from a zoo of models avaliable.
    If transition = True it will select a transtion model otherwise it 
    will select a measurement model (to help deal with transition and 
    measurement models with the same name) 
    """
    if not transition:
        assert rdim != None, "measurement models must have rdim specified"

    if name == "constant" and transition:
        model = ConstantTransition(zdim)

    elif name == "constant" and not transition:
        model = ConstantMeasurement(zdim, rdim)

    elif name == "mixed" and transition:
        assert "K" in kwargs, "mixed model requires keyword argument \"K\""
        assert "hdim" in kwargs, "mixed model requires keyword argument \"hdim\""
        model = MixedTransition(zdim, **kwargs)

    elif name == "mixed" and not transition:
        assert "K" in kwargs, "mixed model requires keyword argument \"K\""
        assert "hdim" in kwargs, "mixed model requires keyword argument \"hdim\""
        model = MixedMeasurement(zdim, rdim, **kwargs)

    else:
        model = None 
    return model 

def encoder_selector(name: str, xdim, hdim, zdim, encode_d=True):
    if name == "gru":
        encoder = BidirectionalGRU(xdim, hdim, zdim, encode_d)
    elif name == "lstm":
        encoder = BidirectionalLSTM(xdim, hdim, zdim, encode_d)
    else:
        encoder = None 
    return encoder 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FD001")
    parser.add_argument("--save_path", type=str, default="saved_models/KRUL")
    parser.add_argument("--zdim", type=int, default=2)
    parser.add_argument("--hdim", type=int, default=50)
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--T", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--transition_model", type=str, default="mixed")
    parser.add_argument("--measurement_model", type=str, default="mixed")
    parser.add_argument("--encoder", type=str, default="gru")
    parser.add_argument("--replay", type=str2bool, default=True, const=True, nargs="?")
    parser.add_argument("--elbo", type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument("--bs", type=int, default=150)
    parser.add_argument("--max_rul", type=float, default=130)
    args = parser.parse_args()
    
    tf.keras.backend.set_floatx("float64")  # needed for parallel Kalman filter accuracy 

    # --- Data prep --- 
    PATH = "CMAPSS"
    train_loader, valid_loader, x_test, y_test, t_test = prep_data(PATH, args.dataset, T=args.T, bs=args.bs, max_rul=args.max_rul)

    # --- Create model and training instance --- 

    rdim = 1 
    xdim = x_test[0].shape[-1] + t_test[0].shape[-1]

    transition_model = model_selector(args.transition_model, args.zdim, transition=True, K=args.K, hdim=args.hdim)
    measurement_model = model_selector(args.measurement_model, args.zdim, rdim, transition=False, K=args.K, hdim=args.hdim)
    encoder = encoder_selector(args.encoder, xdim, args.hdim, args.zdim)

    if transition_model == None or measurement_model == None:
        print("ERROR: no model selected")
        exit()
    if encoder == None:
        print("ERROR: no encoder selected")
        exit()

    model = Kalman_DVAE(encoder, transition_model, measurement_model)
    if args.elbo:
        inf_encoder = encoder_selector(args.encoder, xdim+rdim, args.hdim, args.zdim, encode_d=False)
        inf_transition = model_selector(args.transition_model, args.zdim, transition=True, K=args.K, hdim=args.hdim)
        model.store_inference_models(inf_encoder, inf_transition)

    save_PATH = args.save_path + "_" + args.transition_model + "_"  + args.measurement_model + "_" + args.encoder\
         + "_" + args.dataset # file Path for saving the model with lowest validation loss 
    trainer = Trainer(args.lr, args.alpha)

    model.save_weights(save_PATH)   # just to test if it works 
    
    begin = time.time() # time how long training takes 
    # --- Training ---
    trainer.train_model(args.epochs, train_loader, valid_loader, model, save_PATH, args.replay, args.elbo)
    # ----------------
    end = time.time()
    runtime = end - begin 
    
    if args.replay:
        alpha = args.alpha
    else: 
        alpha = 1.
    model_params = {
        "xdim": xdim, 
        "hdim": args.hdim, 
        "zdim": args.zdim,
        "T": args.T,
        "K": args.K,
        "alpha": alpha,
        "epochs": args.epochs,
        "lr": args.lr,
        "max rul": args.max_rul,
        "batch size": args.bs, 
        "train time": runtime,
        "elbo": args.elbo
    }

    json_save = save_PATH + ".json"
    print("saving model construction hyperparameters in " + json_save)
    with open(json_save, "w") as outfile:
        json.dump(model_params, outfile)

    print("Training runtime: {:.4f} minutes".format(runtime / 60.))