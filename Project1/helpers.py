import logging
import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from dlc_practical_prologue import generate_pair_sets
from torch.utils.data import DataLoader
import numpy as np


def evaluate(model, data_loader, auxiliary_loss, criterion):
    correct = 0
    correct_digit = 0
    total = 0
    loss = 0
    for (image, target, digit_target) in data_loader:
        total += len(target)
        if not auxiliary_loss:
            output = model(image)
            loss += criterion(output, target)
            _, pred = torch.max(output, 1)
            correct += (pred == target).sum().item()
        else:
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
          epochs=10, test_every=1, gamma = 0, weight_sharing=False, auxiliary_loss=False):
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

    if gamma != 0:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=gamma)

    for epoch in range(epochs):
        step = 0
        with tqdm(total=1000, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for (image, target, digit_target) in train_data_loader:
                model.train()
                step += 1
                optimizer.zero_grad()
                if auxiliary_loss:
                    digit1, digit2, output = model(image)
                    loss = criterion(output, target)
                    loss += AL_weight * criterion(digit1, digit_target[:, 0])
                    loss += AL_weight * criterion(digit2, digit_target[:, 1])

                else:

                    output = model(image)
                    # print(output.shape)
                    loss = criterion(output, target)

                loss.backward()
                optimizer.step()
                if gamma != 0 and epoch > 5:
                    scheduler.step()
                pbar.set_postfix(**{"loss (batch)": loss.item()})
                pbar.update(100)
                if step % test_every == 0:
                    model.eval()
                    with torch.no_grad():
                        if auxiliary_loss:
                            acc_train, acc_train_digit, loss_tr = evaluate(model, train_data_loader, auxiliary_loss, criterion)
                            acc_test, acc_test_digit, loss_te = evaluate(model, test_data_loader, auxiliary_loss, criterion)

                            accuracy_train_digit.append(acc_train_digit)
                            accuracy_test_digit.append(acc_test_digit)
                        else:
                            acc_train, loss_tr= evaluate(model, train_data_loader, auxiliary_loss, criterion)
                            acc_test, loss_te = evaluate(model, test_data_loader, auxiliary_loss, criterion)

                        losses_tr.append(loss_tr)
                        losses_te.append(loss_te)

                        accuracy_train.append(acc_train)
                        accuracy_test.append(acc_test)

                    if accuracy_train_digit:
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
        ax1.legend(['Train - Green', 'Validation - Blue', 'Train_Aux - Orange', 'Validation_Aux - Purple'])

    else:
        ax1.set_ylabel("Mean Accuracy Across Trials")
        ax1.legend(['Train - Green', 'Test-Blue'])
    ax1.tick_params(axis='y')
    ax1.set_title("Weight Sharing: "+ str(weight_sharing) + " Auxiliary Loss: "+ str(auxiliary_loss))
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.legend(['Train Loss - Red', 'Validation Loss - Brown'])
    color_loss_tr = 'tab:red'
    color_loss_te = 'tab:brown'

    ax2.set_ylabel("Loss", color=color_loss_tr)  # we already handled the x-label with ax1
    ax2.plot(range(len(losses_tr)), losses_tr, color=color_loss_tr)
    ax2.plot(range(len(losses_te)), losses_te, color=color_loss_te)

    ax2.tick_params(axis='y', labelcolor=color_loss_tr)

    fig.tight_layout()
    plt.grid()
    # plt.show()


def get_train_stats(model, lr, reg, criterion, AL_weight, epochs, batch_size = 100, test_every = 5, gamma = 0, weight_sharing = False, auxiliary_loss = False):
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
    for i in range(5): # change to 10+ for report
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
