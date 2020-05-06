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
    print(device)

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

    epochs = "20" #@param [10, 20, 50]
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

    print(A_train.shape)
    print(len(Attr_train), Attr_train[0].shape)

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

    ############################ Start Training #############################
    # encoder = Encoder(modelArgs, trainArgs, device).to(device)
    # z = encoder(Attr_train[0].float().to(device), A_train_mod[0].float().to(device))
    # decoder = Decoder(modelArgs, trainArgs, device).to(device)
    # A_hat, attr_hat = decoder(z)

    vae = VAE_v2(modelArgs, trainArgs,device).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=trainArgs["lr"], weight_decay=1e-5)
    # A_hat, attr_hat = vae(Attr_train[0].float().to(device), A_train_mod[0].float().to(device))

    train_losses = []
    validation_losses = []
    batched_z = []
    batched_A_hat = []
    batched_Attr_hat = []
    batched_gcn_filters_from_A_hat = []
    batched_z_test = []
    batched_A_hat_test = []
    batched_Attr_hat_test = []
    batched_gcn_filters_from_A_hat_test = []

    print("\n\n =================Start Training=====================")
    for e in range(trainArgs["epochs"]):
        print("Epoch {} / {}".format(e + 1, trainArgs["epochs"]))
        # for i in tqdm(range(len(Attr_train)), leave=True):
        loss_cum = 0
        vae.train()
        for i in range(len(Attr_train)):
            optimizer.zero_grad()
            attr = Attr_train[i].float().to(device)
            A = A_train[i].float().to(device)
            graph_conv_filters = A_train_mod[i].float().to(device)

            z, z_mean, z_log_var, A_hat, attr_hat = vae(attr, graph_conv_filters)

            if e + 1 == trainArgs["epochs"]:
                batched_z.append(z.detach())
                batched_Attr_hat.append(attr_hat.detach())
                batched_A_hat.append(A_hat.detach())
                temp = A_hat.detach().cpu()
                batched_gcn_filters_from_A_hat.append(preprocess_adj_tensor_with_identity(torch.squeeze(temp, -1), symmetric = False))

            loss = loss_func((A, attr), (A_hat, attr_hat), z_mean, z_log_var, trainArgs, modelArgs)
            loss_cum += loss.item()
            loss.backward()
            optimizer.step()

        print("At Epoch {}, training loss {} ".format(e + 1, loss_cum / len(Attr_train)))
        train_losses.append(loss_cum / len(Attr_train))

        ### validation dataset
        vae.eval()
        with torch.no_grad():
            loss_cum = 0
            for i in range(len(Attr_validate)):
                attr = Attr_validate[i].float().to(device)
                A = A_validate[i].float().to(device)
                graph_conv_filters = A_validate_mod[i].float().to(device)

                z, z_mean, z_log_var, A_hat, attr_hat = vae(attr, graph_conv_filters)
                loss = loss_func((A, attr), (A_hat, attr_hat), z_mean, z_log_var, trainArgs, modelArgs)
                loss_cum += loss.item()

                if e + 1 == trainArgs["epochs"]:
                    batched_z_test.append(z.detach())
                    batched_Attr_hat_test.append(attr_hat.detach())
                    batched_A_hat_test.append(A_hat.detach())
                    temp = A_hat.detach().cpu()
                    batched_gcn_filters_from_A_hat_test.append(preprocess_adj_tensor_with_identity(torch.squeeze(temp, -1), symmetric = False))

                # print(torch.mean(list(vae.parameters())[0].grad))

        print("At Epoch {}, validation loss {} ".format(e + 1, loss_cum / len(Attr_validate)))
        validation_losses.append(loss_cum / len(Attr_validate))

    # showLoss("VAE", train_losses, validation_losses)
    # drawGraph(A_train, batched_A_hat)
    # debugDecoder(A_train, A_validate, batched_A_hat, batched_A_hat_test, "hard_threshold", True)


    ################ Training Discriminator
    print('=========== debug ======')
    discriminator = Discriminator(modelArgs, device).to(device)
    optimizer_D = optim.Adam(discriminator.parameters(), lr = 0.001)
    print('=========== debug ==== ')

    ## first train discriminatorm using old generated A_hat from last epoch and real A
    print("\n\n====================================================================================================")
    print("Training Discriminator...")
    epochs = 10 ######################## change epoch here
    loss_train, loss_test = [],[]
    for e in range(epochs):

        loss_cum = 0
        discriminator.train()

        for i in range(len(batched_z)):

            fil = batched_gcn_filters_from_A_hat[i].float().to(device)
            attr_hat = batched_Attr_hat[i].float().to(device)
            A_hat = batched_A_hat[i].to(device)
            A = A_train[i].to(device)
            attr = Attr_train[i].float().to(device)
            train_fil = A_train_mod[i].float().to(device)

            optimizer_D.zero_grad()

            _, preds = discriminator(attr_hat, fil)
            labels = torch.zeros(fil.shape[0]).to(device)
            loss_D_gen = binary_cross_entropy_loss_w(labels.flatten(), preds.flatten())

            _, preds = discriminator(attr, train_fil)
            labels = torch.ones(fil.shape[0]).to(device)
            loss_D_true = binary_cross_entropy_loss_w(labels.flatten(), preds.flatten())
            loss_D = loss_D_true + loss_D_gen
            loss_cum += loss_D.item()
            loss_D.backward()
            optimizer_D.step()

        loss_train.append(loss_cum / len(batched_z))
        print("At Epoch {}, training loss {} ".format(e + 1, loss_cum / len(batched_z)))

        with torch.no_grad():
            loss_cum = 0
            discriminator.eval()
            for i in range(len(batched_z_test)):
                fil = batched_gcn_filters_from_A_hat_test[i].float().to(device)
                attr_hat = batched_Attr_hat_test[i].float().to(device)
                A_hat = batched_A_hat_test[i].to(device)
                A = A_validate[i].to(device)
                attr = Attr_validate[i].float().to(device)
                test_fil = A_validate_mod[i].float().to(device)

                _, preds = discriminator(attr_hat, fil)
                labels = torch.zeros(fil.shape[0]).to(device)
                loss_D_gen = binary_cross_entropy_loss_w(labels.flatten(), preds.flatten())

                _, preds = discriminator(attr, test_fil)
                labels = torch.ones(fil.shape[0]).to(device)
                loss_D_true = binary_cross_entropy_loss_w(labels.flatten(), preds.flatten())
                loss_D = loss_D_true + loss_D_gen
                loss_cum += loss_D.item()

            print("At Epoch {}, validation loss {} ".format(e + 1, loss_cum / len(batched_z_test)))
            loss_test.append(loss_cum / len(batched_z_test))

    # showLoss("Discriminator", loss_train, loss_test)



    ############################# Steering GAN   ####################################

    ## training tip: same batch, same alpha!


    # w = torch.randn_like(batched_z[0][0], requires_grad=True).unsqueeze(0).to(device)
    w = torch.tensor(np.random.normal(0.0, 0.1, [1, modelArgs["latent_dim"]]),
                 device=device, dtype=torch.float32, requires_grad=True)
    a_w1, a_w2, a_b1, a_b2 = torch.FloatTensor(1).uniform_().to(device).requires_grad_(), \
                             torch.FloatTensor(1).uniform_().to(device).requires_grad_(),\
                             torch.FloatTensor(1).uniform_().to(device).requires_grad_(),\
                             torch.FloatTensor(1).uniform_().to(device).requires_grad_()
    # print(w.shape, attr.shape, A.shape, fil.shape)

    optimizer_w = optim.Adam([a_w1, a_w2, a_b1, a_b2, w], lr=0.001) ################################ adjust lr here!!!!!

    ### Initialize generator
    generator = Decoder_v2(modelArgs, trainArgs, device).to(device)

    decoder_weight = dict(vae.decoder.named_parameters())
    generator_weight = dict(generator.named_parameters())
    for k in generator_weight.keys():
        assert k in decoder_weight
        generator_weight[k] = decoder_weight[k]
    generator.eval()

    ## then train w, fix discriminator parameters
    print("\n\n=================================================================================")
    print("start w training...")

    transform = EdgeTransform()
    w_epochs = 35  ################################# adjust epoch here!!!
    discriminator.eval()
    loss_train = []
    w_A_train = []
    w_A_hat_train = []
    w_edit_A_hat_train = []
    w_gen_A_hat_train = []
    for e in range(w_epochs):
        loss_cum = 0
        for i in tqdm(range(len(batched_A_hat))):
            optimizer_w.zero_grad()

            fil = batched_gcn_filters_from_A_hat[i].float().to(device)
            attr_hat = batched_Attr_hat[i].float().to(device)
            A_hat = batched_A_hat[i].to(device)
            # A = A_train[i]
            z = batched_z[i].to(device)

            ## discretize
            A = A_train[i].cpu().numpy().squeeze(-1)
            A_hat = A_hat.cpu().numpy().squeeze(-1)
            discretizer = Discretizer(A, A_hat)
            A_hat = discretizer.discretize('hard_threshold')
            A = torch.unsqueeze(torch.from_numpy(A), -1)
            A_hat = torch.unsqueeze(torch.from_numpy(A_hat), -1)

            _, alpha_edit = transform.get_train_alpha(A_hat)  # input continuous as default, need discretization!!!

            alpha_gen = a_w2 * F.relu(a_w1 * alpha_edit + a_b1) + a_b2
            # from_numpy
            ## first get edit and D(edit(G(z)))
            edit_attr = attr_hat
            edit_A = transform.get_target_graph(alpha_edit, A_hat)  # replace this with the edit(G(z)) attr & filter! Expect do all graphs in batch in one step!!
            temp = edit_A.detach().cpu()
            edit_fil = preprocess_adj_tensor_with_identity(torch.squeeze(temp, -1), symmetric = False).to(device)
            feature_edit, _ = discriminator(edit_attr.float(), edit_fil.float())


            # Then get G(z + aw) and D(G(z + aw))
            gen_A, gen_attr = generator(z + alpha_gen * w)
            temp = gen_A.detach().cpu()
            gen_fil = preprocess_adj_tensor_with_identity(torch.squeeze(temp, -1), symmetric = False).to(device)
            feature_gen, preds = discriminator(gen_attr.float(), gen_fil.float())

            labels = torch.ones(edit_attr.shape[0]).to(device)
            loss_w = w_loss_func(labels, preds, feature_edit, feature_gen, alpha=10, beta=20)
            loss_w.backward()
            loss_cum += loss_w.item()
            optimizer_w.step()
            # print(w.grad)

            if e + 1 == w_epochs:
                w_A_train.append(A)
                w_A_hat_train.append(A_hat)
                w_edit_A_hat_train.append(edit_A)
                gen_A = gen_A.detach().cpu().numpy().squeeze(-1)
                discretizer = Discretizer(gen_A, gen_A)
                gen_A = discretizer.discretize('hard_threshold')
                gen_A = torch.unsqueeze(torch.from_numpy(gen_A), -1)
                w_gen_A_hat_train.append(gen_A)

        print("At Epoch {}, training loss {} ".format(e + 1, loss_cum / len(batched_A_hat)))
        loss_train.append(loss_cum / len(batched_A_hat))

    drawGraph(w_A_train, w_A_hat_train, w_edit_A_hat_train, w_gen_A_hat_train, sample_size=6)
    showLoss("w", loss_train)











