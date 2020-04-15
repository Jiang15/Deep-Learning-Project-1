import torch
import matplotlib.pyplot as plt



def calc_accuracy(model, data_loader, auxiliary_loss):
    correct = 0
    correct_digit = 0
    total = 0

    for (image, target, digit_target) in data_loader:
        total += len(target)
        if not auxiliary_loss:
            output = model(image)
            _, pred = torch.max(output, 1)
            correct += (pred == target).sum().item()
            return correct / total
        else:
            digit1, digit2, output = model(image)
            _, pred = torch.max(output, 1)
            correct += (pred == target).sum().item()
            _, pred1 = torch.max(digit1, 1)
            correct_digit += (pred1 == digit_target[:,0]).sum().item()
            _, pred2 = torch.max(digit2, 1)
            correct_digit += (pred2 == digit_target[:,1]).sum().item()

        return correct / total, correct_digit/ 2 / total


def train(train_data_loader, test_data_loader,
          model, optimizer, criterion, AL_weight = 0.5,
          epochs=10, test_every=1, auxiliary_loss = False):

    losses = []
    accuracy_train = []
    accuracy_test = []
    accuracy_train_digit = []
    accuracy_test_digit = []

#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    for epoch in range(epochs):
        step = 0

        for (image, target, digit_target) in train_data_loader:
            step += 1
#             model.train()
            optimizer.zero_grad()
            if auxiliary_loss:
                digit1,digit2,output = model(image)
                loss = criterion(output, target)
                loss += AL_weight*criterion(digit1, digit_target[:,0])
                loss += AL_weight*criterion(digit2, digit_target[:,1])

            else:

                output = model(image)

                loss = criterion(output, target)

            loss.backward()
            optimizer.step()
#             scheduler.step()

            if step % test_every == 0:
                losses.append(loss)

                model.eval()
                with torch.no_grad():
                    accuracy_train_data = calc_accuracy(model, train_data_loader, auxiliary_loss)
                    accuracy_test_data = calc_accuracy(model, test_data_loader, auxiliary_loss)
                    if auxiliary_loss:
                        acc_train, acc_train_digit = accuracy_train_data
                        acc_test, acc_test_digit = accuracy_test_data

                        accuracy_train_digit.append(acc_train_digit)
                        accuracy_test_digit.append(acc_test_digit)
                    else:
                        acc_train = accuracy_train_data
                        acc_test = accuracy_test_data

                    accuracy_train.append(acc_train)
                    accuracy_test.append(acc_test)

                print(f"Epoch {epoch + 1}/{epochs}, Step {step}, loss {losses[-1]:0.4f},",
                      f"train accuracy {accuracy_train[-1]}, test accuracy {accuracy_test[-1]}",
                      f"train accuracy digit {accuracy_train_digit[-1]}, test accuracy digit {accuracy_test_digit[-1]}"
                      if accuracy_train_digit else "")
    if auxiliary_loss:
        return accuracy_train, accuracy_test, losses, accuracy_train_digit, accuracy_test_digit
    else:
        return accuracy_train, accuracy_test, losses

def plot_train_info(train_info, auxiliary_loss):
    if auxiliary_loss:
        accuracy_train, accuracy_test, losses, acc_train_digit, acc_test_digit = train_info
    else:
        accuracy_train, accuracy_test, losses = train_info
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
        ax1.set_ylabel("Accuracy")
        ax1.legend(['Train - Green','Test-Blue', 'Train_Aux - Orange', 'Test_Aux - Purple'])

    else:
        ax1.set_ylabel("Accuracy")
        ax1.legend(['Train - Green','Test-Blue'])
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.legend(['Loss'])
    color_loss = 'tab:red'
    ax2.set_ylabel("Loss", color=color_loss)  # we already handled the x-label with ax1
    ax2.plot(range(len(losses)), losses, color=color_loss)
    ax2.tick_params(axis='y', labelcolor=color_loss)

    fig.tight_layout()
    plt.grid()
    plt.show()
