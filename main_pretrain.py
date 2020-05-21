from utils import *
from model import *
import numpy as np
import torch
import sys
import torch.nn.functional as F
import torch.optim as optim
# from transform_wrappers_multiprocessing import *
from transform_wrappers import *
from matplotlib import pyplot as plt
from tqdm import tqdm
from discretize import *
from visualization import *


np.set_printoptions(linewidth=np.inf)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(f"using device {device}")

    ####################     Generation parameters     #######################################################
    dataArgs = dict()

    maximum_number_of_nodes_n = "12" #@param [12, 24, 30, 48]
    dataArgs["max_n_node"] = int(maximum_number_of_nodes_n)

    range_of_linkage_probability_p = "0.0, 1.0" #@param [[0.0,1.0], [0.2,0.8], [0.5,0.5]]
    dataArgs["p_range"] = [float(range_of_linkage_probability_p.split(",")[0]), float(range_of_linkage_probability_p.split(",")[1])]

    node_attributes = "degree" #@param ["uniform", "degree", "random"]
    dataArgs["node_attr"] = node_attributes

    number_of_graph_instances = "15000" #@param [1, 100, 1000, 10000, 25000, 50000, 100000, 200000, 500000, 1000000]
    dataArgs["n_graph"] = int(number_of_graph_instances)

    dataArgs["upper_triangular"] = False
    A, Attr, Param, Topol = generate_data_v2(dataArgs)
    # g, a, attr = unpad_data(A[0], Attr[0])

    ####################     Model parameters     #######################################################
    modelArgs = {"gnn_filters": 2, "conv_filters": 16, "kernel_size": 3}

    number_of_latent_variables= "20" #@param [1, 2, 3, 4, 5]
    modelArgs["latent_dim"] = int(number_of_latent_variables)

    trainArgs = dict()

    weight_graph_reconstruction_loss = "30" #@param [0, 1, 2, 3, 5, 10, 20]
    weight_attribute_reconstruction_loss = "5" #@param [0, 1, 2, 3, 5, 10, 20]
    beta_value = "20" #@param [0, 1, 2, 3, 5, 10, 20]
    trainArgs["loss_weights"] = [int(weight_graph_reconstruction_loss), int(weight_attribute_reconstruction_loss), int(beta_value)]

    epochs = "35" #@param [10, 20, 50]
    trainArgs["epochs"] = int(epochs)
    batch_size = "512" #@param [2, 4, 8, 16, 32, 128, 512, 1024]
    trainArgs["batch_size"] = int(batch_size)
    early_stop = "2" #@param [1, 2, 3, 4, 10]
    trainArgs["early_stop"] = int(early_stop)
    train_test_split = "0.2" #@param [0.1, 0.2, 0.3, 0.5]
    train_validation_split = "0.1" #@param [0.1, 0.2, 0.3, 0.5]
    trainArgs["data_split"] = float(train_test_split)
    trainArgs["validation_split"] = float(train_validation_split)
    lr = "0.001"  #@param [0.1, 0.01, 0.001, 0.0001, 0.00001]
    trainArgs["lr"] = float(lr)



    ## Train and Test Split _______________________________________________

    A_train = torch.from_numpy(A[:int((1-trainArgs["data_split"]-trainArgs["validation_split"])*A.shape[0])])
    Attr_train = generate_batch(torch.from_numpy(Attr[:int((1-trainArgs["data_split"]-trainArgs["validation_split"])*Attr.shape[0])]), trainArgs["batch_size"])
    Param_train = generate_batch(torch.from_numpy(Param[:int((1-trainArgs["data_split"]-trainArgs["validation_split"])*Param.shape[0])]), trainArgs["batch_size"])
    Topol_train = generate_batch(torch.from_numpy(Topol[:int((1-trainArgs["data_split"]-trainArgs["validation_split"])*Topol.shape[0])]), trainArgs["batch_size"])

    A_validate = torch.from_numpy(A[int((1-trainArgs["data_split"]-trainArgs["validation_split"])*A.shape[0]):int((1-trainArgs["data_split"])*Attr.shape[0])])
    Attr_validate = generate_batch(torch.from_numpy(Attr[int((1-trainArgs["data_split"]-trainArgs["validation_split"])*Attr.shape[0]):int((1-trainArgs["data_split"])*Attr.shape[0])]), trainArgs["batch_size"])
    Param_validate = generate_batch(torch.from_numpy(Param[int((1-trainArgs["data_split"]-trainArgs["validation_split"])*Param.shape[0]):int((1-trainArgs["data_split"])*Attr.shape[0])]), trainArgs["batch_size"])
    Topol_validate = generate_batch(torch.from_numpy(Topol[int((1-trainArgs["data_split"]-trainArgs["validation_split"])*Topol.shape[0]):int((1-trainArgs["data_split"])*Attr.shape[0])]), trainArgs["batch_size"])

    A_test = torch.from_numpy(A[int((1-trainArgs["data_split"])*A.shape[0]):])
    Attr_test = generate_batch(torch.from_numpy(Attr[int((1-trainArgs["data_split"])*Attr.shape[0]):]), trainArgs["batch_size"])
    Param_test = generate_batch(torch.from_numpy(Param[int((1-trainArgs["data_split"])*Param.shape[0]):]), trainArgs["batch_size"])
    Topol_test = generate_batch(torch.from_numpy(Topol[int((1-trainArgs["data_split"])*Topol.shape[0]):]), trainArgs["batch_size"])

    # print(A_train.shape)
    # print(len(Attr_train), Attr_train[0].shape)

    ## build graph_conv_filters
    SYM_NORM = True
    A_train_mod = generate_batch(preprocess_adj_tensor_with_identity(torch.squeeze(A_train, -1), SYM_NORM), trainArgs["batch_size"])
    A_validate_mod = generate_batch(preprocess_adj_tensor_with_identity(torch.squeeze(A_validate, -1), SYM_NORM), trainArgs["batch_size"])
    A_test_mod = generate_batch(preprocess_adj_tensor_with_identity(torch.squeeze(A_test, -1), SYM_NORM), trainArgs["batch_size"])

    A_train = generate_batch(A_train, trainArgs["batch_size"])
    A_validate = generate_batch(A_validate, trainArgs["batch_size"])
    A_test = generate_batch(A_test, trainArgs["batch_size"])

    train_data = (Attr_train, A_train_mod, Param_train, Topol_train)
    validate_data = (Attr_validate, A_validate_mod, Param_validate, Topol_validate)
    test_data = (Attr_test, A_test_mod, Param_test, Topol_test)

    # attribute first -> (n, n), adjacency second -> (n, n, 1)
    modelArgs["input_shape"], modelArgs["output_shape"] = ((Attr_train[0].shape[1], Attr_train[0].shape[2]), (int(A_train_mod[0].shape[1] / modelArgs["gnn_filters"]), A_train_mod[0].shape[2], 1)),\
                                                          ((Attr_test[0].shape[1], Attr_test[0].shape[1]), (int(A_test_mod[0].shape[1] / modelArgs["gnn_filters"]), A_test_mod[0].shape[2], 1))
    # print(modelArgs["input_shape"], modelArgs["output_shape"])
    # print(A_train[0].shape)

    vae = torch.load("vae.model")

    train_losses = []
    validation_losses = []
    batched_z = []
    batched_A_hat = []
    batched_Attr_hat = []
    batched_A_hat_discretized = []
    batched_gcn_filters_from_A_hat = []
    batched_z_test = []
    batched_A_hat_test = []
    batched_Attr_hat_test = []
    batched_gcn_filters_from_A_hat_test = []
    batched_A_hat_raw_train = []
    batched_A_hat_raw_test = []
    batched_A_hat_max_train = []
    batched_A_hat_max_test = []
    batched_A_hat_min_train = []
    batched_A_hat_min_test = []
    print("\n\n =================Extracting useful information=====================")
    vae.eval()
    for e in range(1):
        loss_cum = 0
        for i in range(len(Attr_train)):
            attr = Attr_train[i].float().to(device)
            A = A_train[i].float().to(device)
            graph_conv_filters = A_train_mod[i].float().to(device)

            z, z_mean, z_log_var, A_hat, attr_hat, A_hat_raw, max_score_per_node, min_score_per_node = vae(attr, graph_conv_filters)

            if e + 1 == trainArgs["epochs"]:
                batched_z.append(z.detach())
                batched_Attr_hat.append(attr_hat.detach())
                batched_A_hat.append(A_hat.detach())
                temp = A_hat.detach().cpu()
                batched_gcn_filters_from_A_hat.append(preprocess_adj_tensor_with_identity(torch.squeeze(temp, -1), symmetric = False))
                A_discretize = A.cpu().squeeze().numpy()
                A_hat_discretize = A_hat.detach().cpu().squeeze().numpy()
                discretizer = Discretizer(A_discretize, A_hat_discretize)
                A_hat_discretize = discretizer.discretize('hard_threshold')
                A_hat_discretize = torch.unsqueeze(torch.from_numpy(A_hat_discretize), -1)
                batched_A_hat_discretized.append(A_hat_discretize)
                batched_A_hat_raw_train.append(A_hat_raw.detach())
                batched_A_hat_max_train.append(max_score_per_node.detach())
                batched_A_hat_min_train.append(min_score_per_node.detach())


            loss = loss_func((A, attr), (A_hat, attr_hat), z_mean, z_log_var, trainArgs, modelArgs)
            loss_cum += loss.item()


        print("Model loss {} ".format(loss_cum / len(Attr_train)))
        train_losses.append(loss_cum / len(Attr_train))

