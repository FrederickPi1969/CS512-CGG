from utils import *
from model import *
from graph_operations import *
import numpy as np
import torch.optim as optim
import torch
from model_with_discriminator import *
from matplotlib import pyplot as plt

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    ####################     Generation parameters     #######################################################
    dataArgs = dict()

    maximum_number_of_nodes_n = "24" #@param [12, 24, 30, 48]
    dataArgs["max_n_node"] = int(maximum_number_of_nodes_n)

    range_of_linkage_probability_p = "0,1" #@param [[0.0,1.0], [0.2,0.8], [0.5,0.5]]
    dataArgs["p_range"] = [float(range_of_linkage_probability_p.split(",")[0]), float(range_of_linkage_probability_p.split(",")[1])]

    node_attributes = "uniform" #@param ["none", "uniform", "degree", "p_value", "random"]
    dataArgs["node_attr"] = node_attributes

    # number_of_graph_instances = "20000" #@param [1, 100, 1000, 10000, 25000, 50000, 100000, 200000, 500000, 1000000]
    number_of_graph_instances = "50" #@param [1, 100, 1000, 10000, 25000, 50000, 100000, 200000, 500000, 1000000]
    dataArgs["n_graph"] = int(number_of_graph_instances)

    A, Attr, Param, Topol = generate_data(dataArgs)
    g, a, attr = unpad_data(A[0], Attr[0])

    ####################     Model parameters     #######################################################
    modelArgs = {"gnn_filters": 2, "conv_filters": 16, "kernel_size": 3}

    number_of_latent_variables= "10" #@param [1, 2, 3, 4, 5]
    modelArgs["latent_dim"] = int(number_of_latent_variables)

    trainArgs = dict()

    weight_graph_reconstruction_loss = "5" #@param [0, 1, 2, 3, 5, 10, 20]
    weight_attribute_reconstruction_loss = "2" #@param [0, 1, 2, 3, 5, 10, 20]
    beta_value = "20" #@param [0, 1, 2, 3, 5, 10, 20]
    trainArgs["loss_weights"] = [int(weight_graph_reconstruction_loss), int(weight_attribute_reconstruction_loss), int(beta_value)]

    # epochs = "20" #@param [10, 20, 50]
    epochs = "3" #@param [10, 20, 50]
    trainArgs["epochs"] = int(epochs)
    batch_size = "1024" #@param [2, 4, 8, 16, 32, 128, 512, 1024]
    trainArgs["batch_size"] = int(batch_size)
    early_stop = "2" #@param [1, 2, 3, 4, 10]
    trainArgs["early_stop"] = int(early_stop)
    train_test_split = "0.2" #@param [0.1, 0.2, 0.3, 0.5]
    train_validation_split = "0.1" #@param [0.1, 0.2, 0.3, 0.5]
    trainArgs["data_split"] = float(train_test_split)
    trainArgs["validation_split"] = float(train_validation_split)
    lr = "0.001"  #@param [0.1, 0.01, 0.001, 0.0001, 0.00001]
    trainArgs["lr"] = float(lr)
    alpha = "0.1" #@param [0.1, 0.2, 0.3]
    trainArgs["alpha"] = float(alpha)
    gan_batch_size = "32" #@param [2, 4, 8, 16, 32, 128, 512, 1024]
    trainArgs["gan_batch_size"] = int(batch_size)



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

    generated_batch_size = int(A_train.shape[0])
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

    # attribute first -> (n, 1), adjacency second -> (n, n, 1)
    modelArgs["input_shape"], modelArgs["output_shape"] = ((Attr_train[0].shape[1], 1), (int(A_train_mod[0].shape[1] / modelArgs["gnn_filters"]), A_train_mod[0].shape[2], 1)),\
                                                          ((Attr_test[0].shape[1], 1), (int(A_test_mod[0].shape[1] / modelArgs["gnn_filters"]), A_test_mod[0].shape[2], 1))
    # print(modelArgs["input_shape"], modelArgs["output_shape"])
    # print(A_train[0].shape)

    ############################ Start Training #############################
    # encoder = Encoder(modelArgs, trainArgs, device).to(device)
    # z = encoder(Attr_train[0].float().to(device), A_train_mod[0].float().to(device))
    # decoder = Decoder(modelArgs, trainArgs, device).to(device)
    # A_hat, attr_hat = decoder(z)

    vae = VAE(modelArgs, trainArgs,device).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=trainArgs["lr"])
    # A_hat, attr_hat = vae(Attr_train[0].float().to(device), A_train_mod[0].float().to(device))

    train_losses = []
    validation_losses = []
    print("\n\n =================Start Training=====================")
    for e in range(trainArgs["epochs"]):
        print("Epoch {} / {}".format(e + 1, trainArgs["epochs"]))
        # for i in tqdm(range(len(Attr_train)), leave=True):
        loss = 0
        for i in range(len(Attr_train)):
            vae.train()
            optimizer.zero_grad()
            attr = Attr_train[i].float().to(device)
            A = A_train[i].float().to(device)
            graph_conv_filters = A_train_mod[i].float().to(device)

            z, z_mean, z_log_var, A_hat, attr_hat = vae(attr, graph_conv_filters)
            loss = loss_func((A, attr), (A_hat, attr_hat), z_mean, z_log_var, trainArgs, modelArgs)

            loss.backward()
            optimizer.step()

            vae.eval()
            ### validation dataset
        print("At Epoch {}, training loss {} ".format(e + 1, loss.item()))
        train_losses.append(loss.item())
        for i in range(len(Attr_validate)):
            attr = Attr_validate[i].float().to(device)
            A = A_validate[i].float().to(device)
            graph_conv_filters = A_validate_mod[i].float().to(device)

            z, z_mean, z_log_var, A_hat, attr_hat = vae(attr, graph_conv_filters)
            loss = loss_func((A, attr), (A_hat, attr_hat), z_mean, z_log_var, trainArgs, modelArgs)
            vae.eval()



        print("At Epoch {}, validation loss {} ".format(e + 1, loss.item()))
        validation_losses.append(loss.item())

    # plt.plot(np.arange(len(train_losses)), np.array(train_losses))
    # plt.plot(np.arange(len(validation_losses)), np.array(validation_losses))
    # plt.show()



    ########################## Discriminator Training #############################
    decoder = vae.decoder
    decoder.requires_grad = False
    discriminator = Discriminator(modelArgs, trainArgs, device).to(device)
    w = torch.tensor(np.random.normal(0.0, 0.1, [modelArgs['latent_dim']])).float().to(device)
    optimizer = optim.Adam(w, lr=trainArgs["lr"])

    def edit_graph():
        """
        edited_list = []
        for graph in A:
            graph = graph.numpy().reshape(dataArgs['max_n_node'], dataArgs['max_n_node'])
            graph = densify(graph).reshape(dataArgs['max_n_node'], dataArgs['max_n_node'], 1)
            edited_list.append(graph)
        edited_list = np.array(edited_list)
        edit_A = torch.from_numpy(edited_list).float()
        """

        # current dumb version
        pass

    
    for e in range(trainArgs["epochs"]):
        print("Epoch {} / {}".format(e + 1, trainArgs["epochs"]))
        loss = 0
        for i in range(len(A_train)):
            discriminator.train()
            optimizer.zero_grad()
            z = torch.tensor([[np.random.uniform(0, 1) for _ in range(modelArgs['latent_dim'])] for i in range(A_train[0].shape[0])]).float()
            A, attr = decoder(z)
            # edit_graph function placeholder
            edit_A = A
            edit_attr = attr
            graph_conv_filters = A_train_mod[i].float()

            decode_A, decode_Attr = decoder(z + trainArgs["alpha"] * w)
            # print(decode_A.shape)

            x, y = discriminator((decode_Attr, edit_attr), graph_conv_filters)
            loss = discriminator_loss_func((decode_A, edit_A), (x, y), trainArgs, modelArgs)

            loss.backward()
            optimizer.step()

            discriminator.eval()

        print("At Epoch {}, training loss {} ".format(e + 1, loss.item()))
        for i in range(len(A_validate)):
            z = torch.tensor([[np.random.uniform(0, 1) for _ in range(modelArgs['latent_dim'])] for i in range(A_validate[0].shape[0])]).float()
            A, attr = decoder(z)
            graph_conv_filters = A_validate_mod[i].float()
            # edit_graph function placeholder
            edit_A = A
            edit_attr = attr

            decode_A, decode_Attr = decoder(z + trainArgs["alpha"] * w)
            x, y = discriminator((edit_attr, decode_Attr), graph_conv_filters)
            loss = discriminator_loss_func((decode_A, edit_A), (x, y), trainArgs, modelArgs)

            vae.eval()

        print(w)