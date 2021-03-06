from utils import *
from model import *
import numpy as np
import torch
import sys
import torch.nn.functional as F
import torch.optim as optim
from utils_for_experiments import *
# from transform_wrappers_multiprocessing import *
from transform_wrappers import *
from matplotlib import pyplot as plt
from tqdm import tqdm
from discretize import *
from visualization import *


np.set_printoptions(linewidth=np.inf)

if __name__ == "__main__":

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    print(f"using device {device}")

    ####################     Generation parameters     #######################################################
    dataArgs = dict()

    maximum_number_of_nodes_n = "20" #@param [12, 24, 30, 48]
    dataArgs["max_n_node"] = int(maximum_number_of_nodes_n)

    range_of_linkage_probability_p = "0.0, 1.0" #@param [[0.0,1.0], [0.2,0.8], [0.5,0.5]]
    dataArgs["p_range"] = [float(range_of_linkage_probability_p.split(",")[0]), float(range_of_linkage_probability_p.split(",")[1])]

    node_attributes = "degree" #@param ["uniform", "degree", "random"]
    dataArgs["node_attr"] = node_attributes

    number_of_graph_instances = "1000" #@param [1, 100, 1000, 10000, 25000, 50000, 100000, 200000, 500000, 1000000]
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



    ######################      VAE        ##########################################
    operation_name = "density"  ## ["transitivity", "density", "forest fire ..."]
    # param_path = operation_name + "_pretrained" + "_" + maximum_number_of_nodes_n
    param_path = "."
    vae = torch.load(param_path + "/vae.model")

    train_losses = []
    validation_losses = []
    batched_z = []
    batched_A_hat = []
    batched_Attr_hat = []
    batched_A_hat_discretized = []
    batched_A_hat_discretized_test = []
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

    def index_of(my_list, target):
        try: return my_list.index(target)
        except: return dataArgs["max_n_node"]

    for e in range(1):
        loss_cum = 0
        for i in range(len(Attr_train)):
            attr = Attr_train[i].float().to(device)
            A = A_train[i].float().to(device)
            graph_conv_filters = A_train_mod[i].float().to(device)

            z, z_mean, z_log_var, A_hat, attr_hat, A_hat_raw, max_score_per_node, min_score_per_node = vae(attr, graph_conv_filters)

            if e + 1 == 1:
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

                # count = 0
                for j in range(len(batched_A_hat_discretized[i])):
                    temp = list(torch.diag(batched_A_hat_discretized[i][j].detach().reshape(dataArgs["max_n_node"], -1)))[::-1]
                    pred_node_num = dataArgs["max_n_node"] - index_of(list(temp), 1)
                    Param_train[i][j][-1] = pred_node_num  # predicted node num have ~96% acc
                    true_node_num = int(Param_train[i][j][0])
                    # print(pred_node_num)
                    # print(true_node_num)

                    # count += pred_node_num == true_node_num
                # print(f"node prediction accuracy : {count / len(batched_A_hat_discretized[i])}")

            loss = loss_func((A, attr), (A_hat, attr_hat), z_mean, z_log_var, trainArgs, modelArgs)
            loss_cum += loss.item()

        print("Model loss {} ".format(loss_cum / len(Attr_train)))


        loss_cum = 0
        for i in range(len(Attr_validate)):
            attr = Attr_validate[i].float().to(device)
            A = A_validate[i].float().to(device)
            graph_conv_filters = A_validate_mod[i].float().to(device)

            z, z_mean, z_log_var, A_hat, attr_hat, A_hat_raw, max_score_per_node, min_score_per_node = vae(attr, graph_conv_filters)
            loss = loss_func((A, attr), (A_hat, attr_hat), z_mean, z_log_var, trainArgs, modelArgs)
            loss_cum += loss.item()

            if e + 1 == 1:
                batched_z_test.append(z.detach())
                batched_Attr_hat_test.append(attr_hat.detach())
                batched_A_hat_test.append(A_hat.detach())
                temp = A_hat.detach().cpu()
                batched_gcn_filters_from_A_hat_test.append(preprocess_adj_tensor_with_identity(torch.squeeze(temp, -1), symmetric = False))

                A_discretize = A.cpu().squeeze().numpy()
                A_hat_discretize = A_hat.detach().cpu().squeeze().numpy()
                discretizer = Discretizer(A_discretize, A_hat_discretize)
                A_hat_discretize = discretizer.discretize('hard_threshold')
                A_hat_discretize = torch.unsqueeze(torch.from_numpy(A_hat_discretize), -1)

                batched_A_hat_discretized_test.append(A_hat_discretize)
                batched_A_hat_raw_test.append(A_hat_raw.detach())
                batched_A_hat_max_test.append(max_score_per_node.detach())
                batched_A_hat_min_test.append(min_score_per_node.detach())

        # print("At Epoch {}, validation loss {} ".format(e + 1, loss_cum / len(Attr_validate)))

    a,p,r,f = compute_score_batched(A_train, batched_A_hat_discretized)
    print(f"VAE performance:  \n Accuracy: {a},  \n Precision: {p},  \n Recall: {r},  \n F1 Score: {f}\n")

    density_ori, diameter_ori, cluster_coef_ori, edges_ori, avg_degree_ori = topological_measure(A_train)
    density_hat, diameter_hat, cluster_coef_hat, edges_hat, avg_degree_hat = topological_measure(batched_A_hat_discretized)
    print(f"--- Truth topology (averaged) ---\n density: {density_ori} \n diameter: {diameter_ori} "
          f"\n clustering coefficient: {cluster_coef_ori} \n edges: {edges_ori} \n avgerage degree {avg_degree_ori}\n")
    print(f"--- Reconstructed topology (averaged) ---\n density: {density_hat} \n diameter: {diameter_hat} "
          f"\n clustering coefficient: {cluster_coef_hat} \n edges: {edges_hat} \n avgerage degree {avg_degree_hat}\n")


    ################ Load Discriminator #####################################
    discriminator = torch.load(param_path + "/discriminator.model")

    ############################# Steering GAN   ####################################

    ## training tip: same batch, same alpha!

    # w = torch.randn_like(batched_z[0][0], requires_grad=True).unsqueeze(0).to(device)
    w = torch.load(param_path + "/w_density.pt")
    a_w1 = torch.load(param_path + "/a_w1_density.pt")
    a_w2 = torch.load(param_path + "/a_w2_density.pt")
    a_b1 = torch.load(param_path + "/a_b1_density.pt")
    a_b2 = torch.load(param_path + "/a_b2_density.pt")




    ### Initialize generator
    generator = Decoder_v2(modelArgs, trainArgs, device).to(device)

    decoder_weight = dict(vae.decoder.named_parameters())
    generator_weight = dict(generator.named_parameters())
    for k in generator_weight.keys():
        assert k in decoder_weight
        generator_weight[k] = decoder_weight[k]
    generator.eval()


    discriminator.eval()
    ## operation = "transitivity", "density", "node_count"

    # transform = GraphTransform(dataArgs["max_n_node"], operation = operation_name, sigmoid = False)
    transform = GraphTransform(dataArgs["max_n_node"], operation = operation_name, sigmoid = False)
    w_epochs = 1  ### adjust epoch here!!!

    loss_train = []
    w_A_train = []
    w_A_hat_train = []
    w_edit_A_hat_train = []
    w_gen_A_hat_train = []
    gen_A_raw_train = []
    gen_A_max_train = []
    gen_A_min_train = []
    masked_norm_A_hats = []

    for e in range(w_epochs):
        for i in tqdm(range(len(batched_A_hat_discretized))):

            fil = batched_gcn_filters_from_A_hat[i].float().to(device)
            attr_hat = batched_Attr_hat[i].float().to(device)
            # A_hat = batched_A_hat[i].to(device)
            A_hat = batched_A_hat_discretized[i].to(device)
            A = A_train[i]
            z = batched_z[i].to(device)

            ## discretize
            # A = A_train[i].cpu().numpy().squeeze(-1)
            # A_hat = A_hat.cpu().numpy().squeeze(-1)
            # discretizer = Discretizer(A, A_hat)
            # A_hat = discretizer.discretize('hard_threshold')
            # A = torch.unsqueeze(torch.from_numpy(A), -1)
            # A_hat = torch.unsqueeze(torch.from_numpy(A_hat), -1)

            # _, alpha_edit = transform.get_train_alpha(A_hat)  # input continuous as default, need discretization!!!
            _, alpha_edit = -0.2, -0.3
            alpha_gen = a_w2 * F.relu(a_w1 * alpha_edit + a_b1) + a_b2
            # from_numpy
            ## first get edit and D(edit(G(z)))
            edit_attr = attr_hat
            edit_A = transform.get_target_graph(alpha_edit, A_hat, list(Param_train[i][:,-1].type(torch.LongTensor)))  # replace this with the edit(G(z)) attr & filter! Expect do all graphs in batch in one step!!
            # print(alpha_edit, alpha_gen)

            temp = edit_A.detach().cpu()
            edit_fil = preprocess_adj_tensor_with_identity(torch.squeeze(temp, -1), symmetric = False).to(device)
            feature_edit, _ = discriminator(edit_attr.float(), edit_fil.float())


            # Then get G(z + aw) and D(G(z + aw))
            gen_A, gen_attr, gen_A_raw, gen_A_max, gen_A_min = generator(z + alpha_gen * w)
            temp = gen_A.detach().cpu()
            gen_fil = preprocess_adj_tensor_with_identity(torch.squeeze(temp, -1), symmetric = False).to(device)
            feature_gen, preds = discriminator(gen_attr.float(), gen_fil.float())

            labels = torch.ones(edit_attr.shape[0]).to(device)



            if e + 1 == w_epochs:
                w_A_train.append(A)
                w_A_hat_train.append(A_hat)
                w_edit_A_hat_train.append(edit_A)
                w_gen_A_hat_train.append(gen_A.detach())
                gen_A_raw_train.append(gen_A_raw.detach())
                masked_norm_A = masked_normalization(gen_A_raw.detach(), Param_train[i])
                masked_norm_A_hats.append(masked_norm_A)
                gen_A_max_train.append(gen_A_max.detach())
                gen_A_min_train.append(gen_A_min.detach())

    print("====================== G(z + aw) v.s. edit(G(z))  results =============================")
    debugDiscretizer(w_A_hat_train, w_edit_A_hat_train, gen_A_raw_train, gen_A_max_train, gen_A_min_train, w_gen_A_hat_train, masked_norm_A_hats, discretize_method="hard_threshold", printMatrix=False, abortPickle=True)


    #debugDecoder(w_edit_A_hat_train, [], w_gen_A_hat_train, [], discretize_method="hard_threshold", printMatrix=True)
    # drawGraph(w_A_train, w_A_hat_train, w_edit_A_hat_train, w_gen_A_hat_train)
    # drawGraphSaveFigure(w_A_train, w_A_hat_train, w_edit_A_hat_train, w_gen_A_hat_train, clearImage=True)
