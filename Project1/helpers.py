import logging
import torch
from torch import optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from dlc_practical_prologue import generate_pair_sets
from torch.utils.data import DataLoader
import numpy as np

# set seed
torch.manual_seed(0)
np.random.seed(0)


def evaluate(model, data_loader, auxiliary_loss, criterion):
    """
    Evaluate given network model with given data set and parameters
    :param model: network model to be evaluated
    :param data_loader: data loader that contains image, target, and digit_target
    :param auxiliary_loss: boolean flag for applying auxiliary loss
    :param criterion: loss function
    :return: primary task accuracy, average digit recognition accuracy, loss
    """
    correct = 0
    correct_digit = 0
    total = 0
    loss = 0
    for (image, target, digit_target) in data_loader:
        total += len(target)
        if not auxiliary_loss: # case without auxiliary loss
            output = model(image)
            loss += criterion(output, target)
            _, pred = torch.max(output, 1)
            correct += (pred == target).sum().item()
        else: # case with auxiliary loss
            digit1, digit2, output = model(image)
            loss += criterion(output, target)
            _, pred = torch.max(output, 1)
            correct += (pred == target).sum().item()
            _, pred1 = torch.max(digit1, 1)
            correct_digit += (pred1 == digit_target[:, 0]).sum().item()
            _, pred2 = torch.max(digit2, 1)
            correct_digit += (pred2 == digit_target[:, 1]).sum().item()
    if not auxiliary_loss:
        return correct / total, loss
    else:
        return correct / total, correct_digit / 2 / total, loss


def train(train_data_loader, test_data_loader,
          model, optimizer, criterion, AL_weight=0.5,
          epochs=25, test_every=1, gamma = 0, weight_sharing=False, auxiliary_loss=False):
    """
    Train network model with given parameters
    :param train_data_loader: data loader for training set
    :param test_data_loader: data load for test(validation set)
    :param model: network model to be trained
    :param optimizer: optimizer for training
    :param criterion: loss function for optimizer
    :param AL_weight: Weight applied to auxiliary loss when combined with primary task loss
    :param epochs: number of training epochs
    :param test_every: number of steps between each model evaluation
    :param gamma: learning rate shceduler's multiplicative factor
    :param weight_sharing:  boolean flag for applying weight sharing
    :param auxiliary_loss:  boolean flag for applying auxiliary loss
    :return: if aux loss is applied, return primary task accuracies of training and test sets, digit recognition accuracies of traing and test sets, training loss and test(validation) loss
            if aux loss is not applied, return primary task accuracies of training and test sets, training loss and test(validation) loss
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {100}
        Training size:   {1000}
        Test size:       {1000}
        Device:          {"CPU"}
        Weight Sharing:  {weight_sharing}
        Auxialiary Loss:  {auxiliary_loss}
        Auxialiary loss weight: {AL_weight}
    ''')

    losses_tr = []
    losses_te = []

    accuracy_train = []
    accuracy_test = []
    accuracy_train_digit = []
    accuracy_test_digit = []

    if gamma != 0: # if gamma is not 0, set up learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=gamma)

    for epoch in range(epochs):
        step = 0
        with tqdm(total=1000, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for (image, target, digit_target) in train_data_loader:
                model.train()
                step += 1
                optimizer.zero_grad()
                if auxiliary_loss: # case with auxiliary loss
                    digit1, digit2, output = model(image)
                    loss = criterion(output, target) # primary task loss
                    loss += AL_weight * criterion(digit1, digit_target[:, 0]) # add weighted aux loss from 1st digit
                    loss += AL_weight * criterion(digit2, digit_target[:, 1]) # add weighted aux loss from 2nd digit
                else: # case without auxiliary loss
                    output = model(image)
                    loss = criterion(output, target) # primary task loss
                loss.backward()
                optimizer.step()
                if gamma != 0 and epoch > 5:
                    scheduler.step()
                pbar.set_postfix(**{"loss (batch)": loss.item()})
                pbar.update(100)
                if step % test_every == 0: # test step number to determine whether to evaluate network model or not
                    model.eval()
                    with torch.no_grad():
                        if auxiliary_loss: # case with aux loss
                            acc_train, acc_train_digit, loss_tr = evaluate(model, train_data_loader, auxiliary_loss, criterion) # evaluate model with training set
                            acc_test, acc_test_digit, loss_te = evaluate(model, test_data_loader, auxiliary_loss, criterion) # evaluate model with test set

                            accuracy_train_digit.append(acc_train_digit)
                            accuracy_test_digit.append(acc_test_digit)
                        else: # case without aux loss
                            acc_train, loss_tr= evaluate(model, train_data_loader, auxiliary_loss, criterion)# evaluate model with training set
                            acc_test, loss_te = evaluate(model, test_data_loader, auxiliary_loss, criterion) # evaluate model with test set

                        losses_tr.append(loss_tr)
                        losses_te.append(loss_te)

                        accuracy_train.append(acc_train)
                        accuracy_test.append(acc_test)

                    if accuracy_train_digit: # log results
                        pbar.set_postfix(**{"loss (batch)": loss.item(), "train acccuracy": accuracy_train[-1],
                                            "test accuracy:": accuracy_test[-1],
                                            "train digit accuracy ": accuracy_train_digit[-1],
                                            "test digit accuracy ": accuracy_test_digit[-1]})
                    else:
                        pbar.set_postfix(**{"loss (batch)": loss.item(), "train acccuracy": accuracy_train[-1],
                                            "test accuracy:": accuracy_test[-1]})

    if auxiliary_loss:
        return accuracy_train, accuracy_test, accuracy_train_digit, accuracy_test_digit, losses_tr, losses_te
    else:
        return accuracy_train, accuracy_test, losses_tr, losses_te


def plot_train_info(train_info, weight_sharing, auxiliary_loss):
    """
    Plot training information, include accuracies of primary task and auxiliary task if applicable, along with training and test losses

    :param train_info: all return values of function train, specifically:
            if aux loss is applied, include primary task accuracies of training and test sets, digit recognition accuracies of traing and test sets, training loss and test(validation) loss
            if aux loss is not applied, include primary task accuracies of training and test sets, training loss and test(validation) loss
    :param weight_sharing: boolean flag for applying weight sharing
    :param auxiliary_loss: boolean flag for applying auxiliary loss
    :return: None
    """
    if auxiliary_loss:
        accuracy_train, accuracy_test, acc_train_digit, acc_test_digit, losses_tr, losses_te= train_info
    else:
        accuracy_train, accuracy_test, losses_tr, losses_te = train_info
    fig, ax1 = plt.subplots()

    color_tr = 'tab:green'
    color_te = 'tab:blue'
    color_trd = 'tab:orange'
    color_ted = 'tab:purple'
    ax1.set_xlabel("Step")

    ax1.plot(range(len(accuracy_train)), accuracy_train, color=color_tr)
    ax1.plot(range(len(accuracy_test)), accuracy_test, color=color_te)
    if auxiliary_loss:
        ax1.plot(range(len(acc_train_digit)), acc_train_digit, color=color_trd)
        ax1.plot(range(len(acc_test_digit)), acc_test_digit, color=color_ted)
        ax1.set_ylabel("Mean Accuracy Across Trials")
        ax1.legend(['Train - Green', 'Validation - Blue', 'Train_Aux - Orange', 'Validation_Aux - Purple'], loc=1)

    else:
        ax1.set_ylabel("Mean Accuracy Across Trials")
        ax1.legend(['Train - Green', 'Test-Blue'], loc=1)
    ax1.tick_params(axis='y')
    ax1.set_title("Weight Sharing: "+ str(weight_sharing) + " Auxiliary Loss: "+ str(auxiliary_loss))
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color_loss_tr = 'tab:red'
    color_loss_te = 'tab:brown'

    ax2.set_ylabel("Loss", color=color_loss_tr)  # we already handled the x-label with ax1
    ax2.plot(range(len(losses_tr)), losses_tr, color=color_loss_tr)
    ax2.plot(range(len(losses_te)), losses_te, color=color_loss_te)
    ax2.legend(['Train Loss - Red', 'Validation Loss - Brown'], loc=2)

    ax2.tick_params(axis='y', labelcolor=color_loss_tr)

    fig.tight_layout()
    plt.grid()



def get_train_stats(model, lr, reg, criterion, AL_weight, epochs, trial = 10, batch_size = 100, test_every = 5, gamma = 0, weight_sharing = False, auxiliary_loss = False):
    """
    Perform training and testing for a given number of trials with the given model and parameters
    :param model: network model
    :param lr: learning rate of the model
    :param reg: weight decay parameter of the model
    :param criterion: loss function
    :param AL_weight: Weight applied to auxiliary loss when combined with primary task loss
    :param epochs: number of training epochs
    :param trial: number of trials
    :param batch_size: data batch size
    :param test_every: number of steps between each model evaluation
    :param gamma: learning rate shceduler's multiplicative factor
    :param weight_sharing: boolean flag for applying weight sharing
    :param auxiliary_loss: boolean flag for applying auxiliary loss
    :return: mean primary task training accuracy across trials, standard deviation of primary task training accuracy across trials,
    mean primary task test accuracy across trials, standard deviation of primary task test accuracy across trials
    """
    accuracy_trial_tr = []
    accuracy_trial_te = []
    mean_acc_tr = []
    mean_acc_te = []
    mean_acc_aux_tr = []
    mean_acc_aux_te = []
    mean_losses_tr = []
    mean_losses_te = []

    train_info_mean = []
    nb_channels = 2
    nb_class = 2
    for i in range(trial):
        net = model(nb_channels, nb_class, weight_sharing, auxiliary_loss)
        train_input, train_target, train_class, test_input, test_target, test_class = generate_pair_sets(1000)
        # Data loaders
        train_loader = DataLoader(list(zip(train_input, train_target, train_class)), batch_size, shuffle=True)
        test_loader = DataLoader(list(zip(test_input, test_target, test_class)), batch_size, shuffle=True)
        train_info = train(train_loader, test_loader,
                           model=net,
                           optimizer=optim.Adam(net.parameters(), lr=lr, weight_decay=reg),
                           criterion=criterion, AL_weight=AL_weight,
                           epochs=epochs, test_every=test_every, gamma = gamma,
                           weight_sharing=weight_sharing,
                           auxiliary_loss=auxiliary_loss)
        if auxiliary_loss:
            accuracy_train, accuracy_test, acc_train_digit, acc_test_digit, losses_tr, losses_te = train_info
            mean_acc_aux_tr.append(acc_train_digit)
            mean_acc_aux_te.append(acc_test_digit)
        else:
            accuracy_train, accuracy_test, losses_tr, losses_te = train_info
        mean_acc_tr.append(accuracy_train)
        mean_acc_te.append(accuracy_test)
        mean_losses_tr.append(losses_tr)
        mean_losses_te.append(losses_te)

        accuracy_trial_tr.append(accuracy_train[-1])
        accuracy_trial_te.append(accuracy_test[-1])
    if auxiliary_loss:
        train_info_mean.append(np.mean(mean_acc_tr,0))
        train_info_mean.append(np.mean(mean_acc_te,0))
        train_info_mean.append(np.mean(mean_acc_aux_tr,0))
        train_info_mean.append(np.mean(mean_acc_aux_te,0))
        train_info_mean.append(np.mean(mean_losses_tr,0))
        train_info_mean.append(np.mean(mean_losses_te,0))

    else:
        train_info_mean.append(np.mean(mean_acc_tr,0))
        train_info_mean.append(np.mean(mean_acc_te,0))
        train_info_mean.append(np.mean(mean_losses_tr,0))
        train_info_mean.append(np.mean(mean_losses_te,0))
    return np.mean(accuracy_trial_tr), np.std(accuracy_trial_tr), np.mean(accuracy_trial_te), np.std(accuracy_trial_te), train_info_mean

def cross_validation(k_fold, lr_set, reg_set, gamma_set, model, criterion, AL_weight, epochs,
                     batch_size = 100, test_every = 5, weight_sharing = False, auxiliary_loss = False):
    """
    Perform K-fold cross validation to optimize hyperprameters lr, reg, and/or gamma on the given model with given parameters
    :param k_fold: number of cross validation folds
    :param lr_set: set of lr
    :param reg_set: set of reg
    :param gamma_set: set of gamma
    :param model: network model
    :param criterion: loss function
    :param AL_weight: Weight applied to auxiliary loss when combined with primary task loss
    :param epochs: number of epochs
    :param batch_size: data batch size
    :param test_every: number of steps between each model evaluation
    :param weight_sharing: boolean flag for applying weight sharing
    :param auxiliary_loss: boolean flag for applying auxiliary loss
    :return: best lr, best reg, best gamma (all based on maximum validation accuracy),
    set of training loss, set of test loss, set of primary task training accuracy, set of primary task test accuracy (all across hyperparameters tested)
    """
    nb_channels = 2
    nb_class = 2

    train_input, train_target, train_class, test_input, test_target, test_class = generate_pair_sets(1000)

    interval = int(train_input.shape[0]/ k_fold)
    indices = torch.randperm(train_input.shape[0])

    accuracy_tr_set = []
    accuracy_te_set = []
    loss_tr_set = []
    loss_te_set = []
    train_info_mean_set = []

    max_acc_te = 0


    for lr in lr_set:
        for reg in reg_set:
            for gamma in gamma_set:
                accuracy_tr_k = 0
                accuracy_te_k = 0
                loss_tr_k = 0
                loss_te_k = 0

                mean_acc_tr = []
                mean_acc_te = []
                mean_acc_aux_tr = []
                mean_acc_aux_te = []
                mean_losses_tr = []
                mean_losses_te = []

                train_info_mean = []
                print('lr = ', lr, 'reg = ', reg, 'gamma = ', gamma)
                for k in range(k_fold):
                    net = model(nb_channels, nb_class, weight_sharing, auxiliary_loss)

                    train_indices = indices[k*interval:(k+1)*interval]
                    input_te = train_input[train_indices]
                    target_te = train_target[train_indices]
                    digit_target_te = train_class[train_indices]
                    residual = torch.cat((indices[0:k*interval],indices[(k+1)*interval:]),0)
                    input_tr = train_input[residual]
                    target_tr = train_target[residual]
                    digit_target_tr = train_class[residual]

                    # Data loaders
                    train_loader = DataLoader(list(zip(input_tr, target_tr, digit_target_tr)), batch_size, shuffle=True)
                    test_loader = DataLoader(list(zip(input_te, target_te, digit_target_te)), batch_size, shuffle=True)
                    train_info = train(train_loader, test_loader,
                                       model=net,
                                       optimizer=optim.Adam(net.parameters(), lr=lr, weight_decay=reg),
                                       criterion=criterion, AL_weight=AL_weight,
                                       epochs=epochs, test_every=test_every, gamma = gamma,
                                       weight_sharing=weight_sharing,
                                       auxiliary_loss=auxiliary_loss)
                    if auxiliary_loss:
                        accuracy_train, accuracy_test, acc_train_digit, acc_test_digit, losses_tr, losses_te = train_info
                    else:
                        accuracy_train, accuracy_test, losses_tr, losses_te = train_info

                    accuracy_tr_k += accuracy_train[-1]
                    accuracy_te_k += accuracy_test[-1]
                    loss_tr_k += losses_tr[-1]
                    loss_te_k += losses_te[-1]

                    mean_acc_tr.append(accuracy_train)
                    mean_acc_te.append(accuracy_test)
                    mean_losses_tr.append(losses_tr)
                    mean_losses_te.append(losses_te)

                accuracy_tr_set.append(accuracy_tr_k/k_fold)
                accuracy_te_set.append(accuracy_te_k/k_fold)
                loss_tr_set.append(loss_tr_k/k_fold)
                loss_te_set.append(loss_te_k/k_fold)
                if accuracy_te_set[-1] > max_acc_te:
                    max_acc_te = accuracy_te_set[-1]
                    best_lr = lr
                    best_reg = reg
                    best_gamma = gamma
                    print(max_acc_te)

                if auxiliary_loss:
                    train_info_mean.append(np.mean(mean_acc_tr,0))
                    train_info_mean.append(np.mean(mean_acc_te,0))
                    train_info_mean.append(np.mean(mean_acc_aux_tr,0))
                    train_info_mean.append(np.mean(mean_acc_aux_te,0))
                    train_info_mean.append(np.mean(mean_losses_tr,0))
                    train_info_mean.append(np.mean(mean_losses_te,0))

                else:
                    train_info_mean.append(np.mean(mean_acc_tr,0))
                    train_info_mean.append(np.mean(mean_acc_te,0))
                    train_info_mean.append(np.mean(mean_losses_tr,0))
                    train_info_mean.append(np.mean(mean_losses_te,0))
                train_info_mean_set.append(train_info_mean)
    return best_lr, best_reg, best_gamma, loss_tr_set, loss_te_set, accuracy_tr_set, accuracy_te_set, train_info_mean_set
