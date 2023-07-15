import argparse
import json 
import time 
import os 

import tensorflow as tf 
import tensorflow_probability as tfp
import numpy as np 
import matplotlib.pyplot as plt 
tfd = tfp.distributions

from kalman_ruls.data.utils import sliding_window, win_to_seq
from kalman_ruls.data.dataprep import DataPrep
from kalman_ruls.networks.DVAE import Kalman_DVAE
from training import str2bool, model_selector, encoder_selector
from kalman_ruls.networks.utils import score_func, alpha_coverage, alpha_mean

# --- Testing Method --- 
def test_model(dvae: Kalman_DVAE, test_x, test_t, test_r, T):
    results = {
        "r_true": [],
        "r_RMSE": 0,
        "r_nll": [],
        "r_mean": [],
        "r_stds": [],
        "fr_mean": [],
        "fr_stds": [], 
        "z_mean": [],
        "z_covs": [],
        "times": [],
        "score": 0,
        "alpha_cover_95": 0,
        "alpha_cover_90": 0, 
        "alpha_cover_50": 0, 
        "alpha_mean_95": 0,
        "alpha_mean_90": 0, 
        "alpha_mean_50": 0
    }

    MSE = [] 
    scores = [] 
    cov_95 = [] 
    cov_90 = [] 
    cov_50 = []
    mu_95 = [] 
    mu_90 = []
    mu_50 = [] 
    for i, x in enumerate(test_x): 
        r = np.float64(test_r[i])
        x = np.float64(x)[0,:,:]
        t = np.float64(test_t[i])[0,:,:]

        x = np.concatenate([x,t], axis=-1)
        # --- get time windowed data --- 
        x = sliding_window(x, T)
        r = sliding_window(np.expand_dims(r,-1), T)

        # --- generate inputs and latent variables ---
        zs, zPs, r_mean, r_var = dvae(x)

        z0, P0 = dvae.get_init_states(x)
        fzs, fPs, us, ds = dvae.inference(x, r, z0, P0)

        _, Rs = dvae.model.get_noise_covar(us)
        Hs = dvae.model.create_Hs(us)

        # --- get nll ---
        p_r = dvae.model.get_marginal_dist(fzs, fPs, z0, P0, us, ds)
        nll = -tf.reduce_mean(p_r.log_prob(r), 0)   # take the mean along the batch dim.
        nll = tf.reduce_sum(nll)  # sum log probs across time (equivalent to taking products of all the probs)
        nll = nll.numpy()

        # --- get filtered RULs ---
        frs = tf.linalg.matvec(Hs, fzs) + ds 
        fSs = Hs @ tf.linalg.matmul(fPs, Hs, transpose_b=True) + Rs 

        # --- convert windowed data back to full sequence --- 
        r_mean = win_to_seq(r_mean.numpy())
        r_var = win_to_seq(r_var.numpy())
        fr_mean = win_to_seq(frs.numpy())
        fr_var = win_to_seq(fSs.numpy())
        z_mean = win_to_seq(zs.numpy())
        z_covs = win_to_seq(zPs.numpy())
        r = win_to_seq(r)
        
        # --- convert log variance to stdev ---
        r_std = np.sqrt(r_var)
        fr_std = np.sqrt(fr_var)

        # --- get metrics for this unit ---
        mse = (r[:,0] - r_mean[:,0]) ** 2
        score = score_func(r_mean[-1,:], r[-1,:])
        coverage_95 = alpha_coverage(r[:,0], r_mean[:,0], r_std[:,0,0], 1.96)
        coverage_90 = alpha_coverage(r[:,0], r_mean[:,0], r_std[:,0,0], 1.64)
        coverage_50 = alpha_coverage(r[:,0], r_mean[:,0], r_std[:,0,0], 0.675)
        mean_95 = alpha_mean(r_mean[:,0], r_std[:,0,0], 1.96)
        mean_90 = alpha_mean(r_mean[:,0], r_std[:,0,0], 1.64)
        mean_50 = alpha_mean(r_mean[:,0], r_std[:,0,0], 0.675)
        
        # --- store test results --- 
        MSE.append(mse)
        scores.append(score)
        cov_95.append(coverage_95)
        cov_90.append(coverage_90)
        cov_50.append(coverage_50)
        mu_95.append(mean_95)
        mu_90.append(mean_90)
        mu_50.append(mean_50)
        results["r_true"].append(r[:,0])
        results["r_nll"].append(nll)
        results["r_mean"].append(r_mean[:,0])
        results["r_stds"].append(r_std[:,0,0])
        results["fr_mean"].append(fr_mean[:,0])
        results["fr_stds"].append(fr_std[:,0,0])
        results["z_mean"].append(z_mean)
        results["z_covs"].append(z_covs)
        results["times"].append(t[:,0])

    MSE = np.concatenate(MSE, axis=0)
    RMSE = np.sqrt(MSE.mean())
    results["r_RMSE"] = RMSE 

    scores = np.concatenate(scores, axis=0).sum()
    results["score"] = scores 

    cov_95 = np.concatenate(cov_95, axis=0).mean()
    cov_90 = np.concatenate(cov_90, axis=0).mean()
    cov_50 = np.concatenate(cov_50, axis=0).mean()
    mu_95 = np.concatenate(mu_95, axis=0).mean()
    mu_90 = np.concatenate(mu_90, axis=0).mean()
    mu_50 = np.concatenate(mu_50, axis=0).mean()
    results["alpha_cover_95"] = cov_95
    results["alpha_cover_90"] = cov_90
    results["alpha_cover_50"] = cov_50
    results["alpha_mean_95"] = mu_95
    results["alpha_mean_90"] = mu_90
    results["alpha_mean_50"] = mu_50
    return results 

# --- Plotting Results --- 
def plot_filtered_ruls(unit, results):
    """
    Useful as a sanity check to see if providing the RUL and filtering 
    does really give us really track the true RUL value and give tighter uncertainty bounds 
    """
    t = results["times"][unit-1]
    lower_bound = results["fr_mean"][unit-1] - results["fr_stds"][unit-1]*2
    upper_bound = results["fr_mean"][unit-1] + results["fr_stds"][unit-1]*2

    plt.figure(figsize=(18,9))
    plt.rc('xtick', labelsize=28)
    plt.rc('ytick', labelsize=28)
    plt.plot(t, results["fr_mean"][unit-1], label="mean RUL estimate")
    plt.fill_between(t, upper_bound, lower_bound, alpha=0.3, label="95$\%$ confidence interval")
    plt.plot(t, results["r_true"][unit-1], lw=2, label="true RUL", color="tab:red")

    plt.xlabel("Time (cycles)", fontsize=32)
    plt.ylabel("RUL (cycles)", fontsize=32)
    plt.legend(prop={"size": 28})
    plt.show()

def plot_rul_vs_time(unit, results):
    """
    Plots the rul vs time for a single unit 
    """
    t = results["times"][unit-1]
    lower_bound = results["r_mean"][unit-1] - results["r_stds"][unit-1]*2
    upper_bound = results["r_mean"][unit-1] + results["r_stds"][unit-1]*2

    plt.figure(figsize=(18,9))
    plt.rc('xtick', labelsize=28)
    plt.rc('ytick', labelsize=28)
    plt.plot(t, results["r_mean"][unit-1], label="mean RUL estimate")
    plt.fill_between(t, upper_bound, lower_bound, alpha=0.3, label="95$\%$ confidence interval")
    plt.plot(t, results["r_true"][unit-1], lw=2, label="true RUL", color="tab:red")

    #plt.title("Unit %i: RUL vs Time"%unit, fontsize=20)
    plt.xlabel("Time (cycles)", fontsize=32)
    plt.ylabel("RUL (cycles)", fontsize=32)
    plt.legend(prop={"size": 28})
    plt.show()

def get_final_ruls(results):
    """
    Gets and stores the final test time RUL estimate (the latest estimate the model made)
    which is often used in evaluating prognostic models 

    Inputs:
        results (dict): results from the testing method 
    
    Outputs: 
        rmse (array): RMSE of the final RUL estimate vs the true RUL 
        r_final (array): final/latest true RUL for each unit in the testing dataset 
        r_fin_est (array): final/latest estimated mean RUL from the model for each unit 
        r_fin_std (array): final/latest estimated standard deviation of the RUL for each unit 
        max_time (int): the maximum time value considering all the testing units (used later in plotting)
    """
    r_final = [] 
    r_fin_est = [] 
    r_fin_std = [] 

    rmse = 0 
    units = len(results["r_true"])
    max_time = 0

    for i in range(units):
        r_max = results["r_true"][i][0]
        r_fin = results["r_true"][i][-1]
        r_est = results["r_mean"][i][-1]
        r_std = results["r_stds"][i][-1]

        rmse += float(np.sqrt(np.mean((r_fin - r_est) ** 2)))
        r_final.append(r_fin)
        r_fin_est.append(r_est)
        r_fin_std.append(r_std)

        if r_max > max_time:
            max_time = r_max

    rmse = rmse / units
    r_final = np.stack(r_final)
    r_fin_est = np.stack(r_fin_est)
    r_fin_std = np.stack(r_fin_std)
    return rmse, r_final, r_fin_est, r_fin_std, max_time

def plot_final_rul_vs_time(r_final, r_fin_est, max_time):
    """
    Plots the final/latest rul estimate for each unit in the testing dataset with 
    time on the x-axis and contrasts it with the true final RUL vs time (which would be 
    a linear equation with gradient of 1 and y-intercept at 0). 

    Hence, how well the final RUL estimates (plotted as a scatter plot) track this line
    representing the True RUL, is a good visual indicator for the accuracy of the model.
    Often we expect the lower the time value the better the estimate (the closer we are to failure 
    the better the RUL estimate, as the data better represents the machines imminent failure)
    """
    t = np.linspace(0, max_time+1, 1000)
    plt.figure(figsize=(18,9))
    plt.rc('xtick', labelsize=28)
    plt.rc('ytick', labelsize=28)
    plt.plot(t, t, color="k", lw=2., label="true")
    plt.scatter(r_fin_est, r_final, label="estimates")

    plt.xlabel("RUL Estimates (cycles)", fontsize=32)
    plt.ylabel("True RUL (cycles)", fontsize=32)
    plt.legend(prop={"size": 28})
    plt.show()


def plot_final_rul_vs_units(r_final, r_fin_est, r_fin_std):
    """
    Plots the final/latest RUL estimate with respect to the unit/machine (as the x-axis). 
    It also shows the bounds calculated with the standard deviation so we can see if the RUL estimate 
    is within these bounds and how tight they are. 

    This plot shows us an overall picture of how well the bounds capture the uncertainties in the model
    and also show us which units were easy with regards to RUL estimation and which were difficult. 
    """
    units = r_final.shape[0]
    unit_list = np.linspace(1, units, units)
    bound = 2*r_fin_std
    out_of_bounds = (r_final < (r_fin_est - bound)) + (r_final > (r_fin_est + bound))
    in_bounds = (1 - out_of_bounds).astype(np.bool_)

    plt.figure(figsize=(18,9))
    plt.rc('xtick', labelsize=28)
    plt.rc('ytick', labelsize=28)
    plt.scatter(unit_list[in_bounds], r_final[in_bounds], marker="o", color="tab:red", label="true")   
    plt.errorbar(unit_list, r_fin_est, bound, capsize=5., fmt="o", label="estimates")
    plt.scatter(unit_list[out_of_bounds], r_final[out_of_bounds], marker="x", color="tab:red", label="true (out of bounds)")

    plt.xlabel("Unit Number", fontsize=32)
    plt.ylabel("RUL (cycles)", fontsize=32)
    plt.legend(prop={"size": 28})
    plt.show()

def plot_latent_vs_time(unit, results):
    """
    Plots the latent trajectories vs time for a specific unit 
    """
    z_mean = results["z_mean"][unit-1]
    z_stds = np.diagonal(results["z_covs"][unit-1], axis1=-2, axis2=-1)
    t = results["times"][unit-1]

    plt.figure(figsize=(18,9))
    plt.plot(t, z_mean, label="latent")

    for i in range(z_mean.shape[-1]):
        plt.fill_between(t, 
                        z_mean[...,i] + 2*z_stds[...,i],
                        z_mean[...,i] - 2*z_stds[...,i],
                        alpha=0.4)

    plt.title("Latent vs Time", fontsize=20)
    plt.xlabel("cycles", fontsize=20)
    plt.ylabel("$\mathbf{z}$", fontsize=20) 
    plt.legend()
    plt.show()

def plot_latent_phase_space(unit, results, dim1, dim2):
    """
    Plots the 2D phase space of the choosen latent dimensions for a specific unit 
    (only works for model with 2 or more latent dimensions)
    """
    z_mean = results["z_mean"][unit-1]
    t = results["times"][unit-1]

    plt.figure(figsize=(18,9))
    plt.scatter(z_mean[:,dim1], z_mean[:,dim2], c=t)
    plt.xlabel("$z_1$", fontsize=32)
    plt.ylabel("$z_2$", fontsize=32)
    plt.rc('xtick', labelsize=28)
    plt.rc('ytick', labelsize=28)
    plt.colorbar()
    plt.show()


def plot_latent_vs_time_all(results, dim):
    """
    Plots the latent trajectory vs time for a choosen latent dimension 
    for all the units
    """
    units = len(results["z_mean"])

    plt.figure(figsize=(18,9))
    for i in range(units):
        z_mean = results["z_mean"][i][:,dim]
        t = results["times"][i]
        plt.plot(t, z_mean, label="latent")

    plt.title("Latent vs Time", fontsize=20)
    plt.xlabel("cycles", fontsize=20)
    plt.ylabel("$\mathbf{z}$", fontsize=20) 
    plt.show()  

def plot_latent_phase_space_all(results, dim1, dim2):
    ts = np.concatenate(results["times"])
    zs = np.concatenate(results["z_mean"])

    seq = 0 
    z_mean = 0
    for i, z in enumerate(results["z_mean"]):
        l = z.shape[0]
        if seq < l:
            seq = l
            z_mean = z 
            
    plt.figure(figsize=(18,9))
    plt.rc('xtick', labelsize=28)
    plt.rc('ytick', labelsize=28)
    plt.plot(z_mean[:,dim1], z_mean[:,dim2], color="tab:blue", lw=3)     
    plt.scatter(zs[:,dim1], zs[:,dim2], c=ts)

    plt.colorbar()
    plt.xlabel("$z_1$", fontsize=32)
    plt.ylabel("$z_2$", fontsize=32)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FD001")
    parser.add_argument("--save_path", type=str, default="saved_models/KRUL")
    parser.add_argument("--save_results", type=str2bool, default=True, const=True, nargs="?")
    parser.add_argument("--run_model", type=str2bool, default=True, const=True, nargs="?")
    parser.add_argument("--transition_model", type=str, default="mixed")
    parser.add_argument("--measurement_model", type=str, default="mixed")
    parser.add_argument("--encoder", type=str, default="gru")
    args = parser.parse_args()

    import seaborn as sb
    sb.set_theme()  # super important import 

    # --- Get Testing data ---
    PATH = "CMAPSS"
    prep_class = DataPrep(PATH, args.dataset)

    if args.dataset == "FD001" or args.dataset == "FD003":
        prep_class.op_normalize(K=1)    # K=1 normalization, K=6 operating condition norm 
    else: 
        prep_class.op_normalize(K=6) 

    save_PATH = args.save_path + "_" + args.transition_model + "_"  + args.measurement_model + "_" + args.encoder\
                 + "_" + args.dataset    # model save file 

    with open(save_PATH + ".json") as file:
        model_params = json.load(file)
    print("Loading hyperparameters from file: " + save_PATH + ".json")
    max_rul = model_params["max rul"]
    x_test, y_test, t_test = prep_class.prep_test(prep_class.ntest, prep_class.RUL, max_rul)
    
    # --- Run the Model --- 
    if args.run_model == True:
        # --- Define Model --- 
        tf.keras.backend.set_floatx("float64")

        xdim = model_params["xdim"]
        hdim = model_params["hdim"]
        zdim = model_params["zdim"]
        K = model_params["K"]
        T = model_params["T"]
        elbo = model_params["elbo"]

        # select specific transition and measurement models 
        rdim = 1 
        transition_model = model_selector(args.transition_model, zdim, transition=True, K=K, hdim=hdim)
        measurement_model = model_selector(args.measurement_model, zdim, rdim, transition=False, K=K, hdim=hdim)
        encoder = encoder_selector(args.encoder, xdim, hdim, zdim)
        if transition_model == None or measurement_model == None:
            print("ERROR: no model selected")
            exit()
        if encoder == None:
            print("ERROR: no encoder selected")
            exit()
        # construct the Kalman DVAE and load the weights 
        model = Kalman_DVAE(encoder, transition_model, measurement_model)

        if elbo:    # even though we don't use this we need it so all the model weights are loaded so an error isn't called 
            inf_encoder = encoder_selector(args.encoder, xdim+rdim, hdim, zdim, encode_d=False)
            inf_transition = model_selector(args.transition_model, zdim, transition=True, K=K, hdim=hdim)
            model.store_inference_models(inf_encoder, inf_transition)

        model.load_weights(save_PATH)

        begin = time.time()
        # --- Test --- 
        results = test_model(model, x_test, t_test, y_test, T)    
        # ------------
        end = time.time()
        runtime = end - begin
        print(f"Total runtime for testing: {runtime}s")

        # --- Save results --- 
        if args.save_results:
            npy_save = save_PATH + "_test_results" + ".npy"
            print("saving test results in " + npy_save)
            np.save(npy_save, results, allow_pickle=True)

    # --- If not running the model, load results of a previously run model ---
    else:
        npy_save = save_PATH + "_test_results" + ".npy" # location of the saved file to be loaded 
        assert os.path.exists(npy_save), f"{npy_save}, File does not exist"
        results = np.load(npy_save, allow_pickle=True).tolist() # actually returns a dict. when calling tolist

    # --- Print and Plot results --- 
    print("Total RMSE: ", results["r_RMSE"])
    print("Total score: ", results["score"])
    print("Total nll: ", sum(results["r_nll"]) / len(results["r_nll"]))
    print("95%% coverage ", results["alpha_cover_95"])
    print("90%% coverage ", results["alpha_cover_90"])
    print("50%% coverage ", results["alpha_cover_50"])
    print("95%% mean width ", results["alpha_mean_95"])
    print("90%% mean width ", results["alpha_mean_90"])
    print("50%% mean width ", results["alpha_mean_50"])

    unit = 100
    plot_filtered_ruls(unit, results)
    plot_rul_vs_time(unit, results)
    plot_latent_phase_space(unit, results, 0, 1)
    plot_latent_phase_space_all(results, 0, 1)
    rmse, r_final, r_fin_est, r_fin_std, max_time = get_final_ruls(results)
    plot_final_rul_vs_time(r_final, r_fin_est, max_time)
    plot_final_rul_vs_units(r_final, r_fin_est, r_fin_std)
