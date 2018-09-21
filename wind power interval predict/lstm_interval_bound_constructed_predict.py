# -*-coding:utf-8-*-
import os, time, warnings, xlwt
import numpy as np
import funcs as fc
import tensorflow as tf
import dataset
from keras.utils import plot_model
from keras.layers.core import Dense, Activation, Dropout, Lambda, Reshape
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Input, concatenate,BatchNormalization
from keras.models import Model
from keras import losses, optimizers,initializers
from keras import backend as K
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide messy TensorFlow warnings
warnings.filterwarnings("ignore")  # Hide messy Numpy warnings
path = './Exp_Result/Experiment_Data_10_bound_论文/'

rnn_type = False  # T:LSTM F:GRU
epochs = 150
timesteps = 9
all_error_times = 0
width_upper = 0
width_lower = 0
alpha = 0
k_width_plus = 0.1
k_width_minus = 0.1
k_alpha_plus = 0.3
k_alpha_minus = 0.3
PINC = 0.92
exp_times = 5
field_start = 1
field_end = 8

for flag in range(field_start - 1, field_end):
    # define variables
    file = xlwt.Workbook()
    table = file.add_sheet('指标')
    row_0 = ['Times', 'PICP', 'PINRW', 'INAD', 'CWC', 'Time Cost(s)']
    for i in range(len(row_0)):
        table.write(0, i, row_0[i])
    table.write(1, 6, 'beta_plus:' + str(k_width_plus))
    table.write(2, 6, 'beta_minus:' + str(k_width_minus))
    table.write(3, 6, 'k_plus:' + str(k_alpha_plus))
    table.write(4, 6, 'k_minus:' + str(k_alpha_minus))
    table.write(5, 6, 'PINC:' + str(PINC))

    # clear current error times
    current_error_times = 0

    # select data set
    [file_name, sheet_name, col, row_start, seq_len, interval] = dataset.select_dataset_windpower8_boundvmd(flag)

    # preprocess data
    print('> Loading data... ')
    [X_train, Y_train, X_test, Y_test] = fc.load_data(file_name, type='excel',
                                                      sheet_name=sheet_name,
                                                      pre_seq_len=timesteps, row=row_start - 1, col=col - 1,
                                                      seq_len=seq_len, interval=interval)
    # normalization
    [x_train, x_maxmin] = fc.maponezero(X_train)
    [y_train, y_maxmin] = fc.maponezero(Y_train)
    x_test = fc.maponezero(X_test, "apply", x_maxmin)
    y_test = fc.maponezero(Y_test, "apply", y_maxmin)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    for times in range(0, exp_times):
        while True:
            width_upper_history = []
            width_lower_history = []
            alpha_history = []
            y_train_lower_bound = y_train - width_lower
            y_train_upper_bound = y_train + width_upper
            y_train_lower_bound_last = y_train_lower_bound
            y_train_upper_bound_last = y_train_upper_bound
            width_lower_tmp = width_lower
            width_upper_tmp = width_upper
            alpha_tmp = alpha

            # clear model data and state to avoid memory leak
            K.clear_session()
            tf.reset_default_graph()

            print('--------------------------------------------------------')
            print('> Compiling...')
            # record start time
            start_time = time.time()
            # build model and compile model
            ipt1 = Input(shape=(timesteps, 1), name='train_input')

            # ipt2 = Input(shape=(1,), name='true_input') #
            if (rnn_type):
                rnn = LSTM(output_dim=32, return_sequences=False)(ipt1)
            else:
                rnn = GRU(output_dim=32, return_sequences=False,kernel_initializer='glorot_normal')(ipt1)
            # lstm = LSTM(output_dim=32, return_sequences=False )(lstm)
            # ds = Dense(16, activation='relu',kernel_initializer='he_uniform')(rnn)
            # # ds=Dropout(0.1)(ds)
            # ds = Dense(8, activation='relu',kernel_initializer='he_uniform')(ds)
            # # ds=Dropout(0.1)(ds)
            # ds = Dense(4, activation='relu',kernel_initializer='he_uniform')(ds)
            ds = Dense(16, kernel_initializer='he_normal')(rnn)
            # ds = BatchNormalization()(ds)
            ds = Activation('relu')(ds)
            # ds=Dropout(0.5)(ds)
            ds = Dense(8, kernel_initializer='he_normal')(ds)
            # ds = BatchNormalization()(ds)
            ds = Activation('relu')(ds)
            # ds = LeakyReLU(alpha=0.05)(ds)
            # # ds=Dropout(0.5)(ds)
            ds = Dense(4, kernel_initializer='he_normal')(ds)
            # ds = BatchNormalization()(ds)
            ds = Activation('relu')(ds)
            # ds=Dropout(0.1)(ds)
            bound = Dense(2, activation="linear", name='upper_lower_bound')(ds)
            # merge = concatenate([ipt2, bound])    #
            # opt1 = Lambda(distance_evaluate_layer)(merge) #
            # opt2 = Lambda(bound_evaluate_layer)(merge)    #
            # opt1 = Reshape((1,), name='output_distance')(opt1)    #
            # opt2 = Reshape((2,), name='output_bound')(bound)   #
            train_model = Model(inputs=[ipt1], outputs=[bound])
            train_model.compile(loss='mse', optimizer=optimizers.Adam())

            # plot_model(train_model, to_file='model.png', show_shapes=False, show_layer_names=True)  # plot my model

            # train model
            print('> Training...')
            # memory leak exists in function fit
            for i in range(epochs):
                history = train_model.fit(x={'train_input': x_train},
                                          y={'upper_lower_bound': np.column_stack(
                                              (y_train_lower_bound, y_train_upper_bound))
                                          },
                                          epochs=1,
                                          batch_size=64,
                                          validation_split=0.05,
                                          verbose=0)
                # print("loss" + history.history['loss'][9])
                predict_bound_train_tmp = train_model.predict(x_train)
                predict_bound_train_tmp = np.sort(predict_bound_train_tmp, axis=1)
                lower_error = np.sum(np.abs(predict_bound_train_tmp[:, 0] - y_train_lower_bound)) / Y_train.shape[0]
                upper_error = np.sum(np.abs(predict_bound_train_tmp[:, 1] - y_train_upper_bound)) / Y_train.shape[0]

                predict_bound_train_tmp = fc.maponezero(predict_bound_train_tmp, 'reverse', y_maxmin)
                picp_train = fc.PICP(predict_bound_train_tmp, Y_train)
                inad_train = fc.INAD(predict_bound_train_tmp, Y_train)
                cwc_train = fc.CWC_proposed(predict_bound_train_tmp, Y_train)
                pinrw_train = fc.PINRW(predict_bound_train_tmp, Y_train)

                # print("------------------------------------------------------------")
                # print("times: " + str(i))
                # print(
                #     "width_lower_tmp:  %f    width_upper_tmp:  %f    alpha:  %f\nlower_error:  %f  upper_error:  %f  " % (
                #         width_lower_tmp, width_upper_tmp, alpha_tmp, lower_error, upper_error))
                # print("train_data:  PICP:  %f  PINRW:  %f  INAD:  %f  CWC:  %f" % (
                #     picp_train, pinrw_train, inad_train, cwc_train))

                if width_lower_tmp > lower_error * alpha_tmp:
                    width_lower_tmp += (lower_error * alpha_tmp - width_lower_tmp) * k_width_minus
                else:
                    width_lower_tmp += (lower_error * alpha_tmp - width_lower_tmp) * k_width_plus

                if width_upper_tmp > upper_error * alpha_tmp:
                    width_upper_tmp += (upper_error * alpha_tmp - width_upper_tmp) * k_width_minus
                else:
                    width_upper_tmp += (upper_error * alpha_tmp - width_upper_tmp) * k_width_plus

                if picp_train < PINC:
                    if np.abs(PINC - picp_train) < 0:
                        alpha_tmp += k_alpha_plus * 0.02
                    else:
                        alpha_tmp += k_alpha_plus * (PINC - picp_train)
                    if alpha_tmp > 5:  # avoid the explosion of alpha
                        alpha_tmp = 5
                else:
                    if np.abs(PINC - picp_train) < 0:
                        alpha_tmp -= k_alpha_plus * 0.02
                    else:
                        alpha_tmp += k_alpha_minus * (PINC - picp_train)
                    if alpha_tmp < 0:  # avoid the explosion of alpha
                        alpha_tmp = 0

                y_train_upper_bound = y_train + width_upper_tmp
                y_train_lower_bound = y_train - width_lower_tmp

                width_upper_history.append(width_upper_tmp)
                width_lower_history.append(width_lower_tmp)
                alpha_history.append(alpha_tmp)

            # predict interval of wind speed
            # memory leak exists in function predict
            predict_bound_train = train_model.predict(x_train)
            predict_bound_train = fc.maponezero(predict_bound_train, 'reverse', y_maxmin)
            predict_bound_train = np.sort(predict_bound_train, axis=1)

            # compute PICP PINRW INAD CWC of train set
            picp_train = fc.PICP(predict_bound_train, Y_train)
            inad_train = fc.INAD(predict_bound_train, Y_train)
            cwc_train = fc.CWC_proposed(predict_bound_train, Y_train)
            pinrw_train = fc.PINRW(predict_bound_train, Y_train)

            # judge effection of train set
            if picp_train >= 0.90 and pinrw_train <= 0.10:
                break

            # print unqualified index of train set
            current_error_times += 1
            all_error_times += 1
            print('Field: ' + str(flag + 1) + '    Times: ' + str(times + 1) + ' (Unqualified+' + str(
                current_error_times) + ')')
            print("train_data:  PICP:  %f  PINRW:  %f  INAD:  %f  CWC:  %f" % (
                picp_train, pinrw_train, inad_train, cwc_train))

        # predict test set
        predict_bound_test = train_model.predict(x_test)
        predict_bound_test = fc.maponezero(predict_bound_test, 'reverse', y_maxmin)
        predict_bound_test = np.sort(predict_bound_test, axis=1)

        # compute PICP PINRW INAD CWC of test set
        inad_test = fc.INAD(predict_bound_test, Y_test)
        picp_test = fc.PICP(predict_bound_test, Y_test)
        pinrw_test = fc.PINRW(predict_bound_test, Y_test)
        cwc_test = fc.CWC_proposed(predict_bound_test, Y_test)

        # save data and model
        fc.write_data(file, times, file_name[:-5], 'Month-' + sheet_name[5:], Y_test,
                      predict_bound_test[:, 0], predict_bound_test[:, 1],
                      Y_train, predict_bound_train[:, 0], predict_bound_train[:, 1],
                      history.history['loss'],
                      time.time() - start_time, picp_test, pinrw_test,
                      inad_test, cwc_test)
        # train_model.save(path + 'model-field' + str(flag + 1) + '-times' + str(times + 1) + '.h5')

        # print Index
        print('Training duration (s) : ', time.time() - start_time)
        print('Field: ' + str(flag + 1) + '    Times: ' + str(times + 1))
        print('unqualified times: ' + str(current_error_times) + '         all fields\' unqualified times: ' + str(
            all_error_times))
        print("train_data:  PICP:  %f  PINRW:  %f  INAD:  %f  CWC:  %f" % (
            picp_train, pinrw_train, inad_train, cwc_train))
        print("test_data:   PICP:  %f  PINRW:  %f  INAD:  %f  CWC:  %f" % (
            picp_test, pinrw_test, inad_test, cwc_test))
        #
        # plt.figure('width')
        # plt.plot(width_upper_history, 'b', label='upper width')
        # plt.plot(width_lower_history, 'g', label='lower width')
        # plt.legend()
        #
        # plt.figure('alpha')
        # plt.plot(alpha_history, 'r', label='alpha')
        # plt.legend()

        # save width and alpha
        # f = xlwt.Workbook()
        # t = f.add_sheet('sheet1')
        # t.write(0, 0, 'width_upper')
        # t.write(0, 1, 'width_lower')
        # t.write(0, 2, 'alpha')
        # for j in range(len(alpha_history)):
        #     t.write(1 + j, 0, str(width_upper_history[j]))
        #     t.write(1 + j, 1, str(width_lower_history[j]))
        #     t.write(1 + j, 2, str(alpha_history[j]))
        # f.save(path + "alpha_and_width.xls")

        # plt.figure('test data predict plot', facecolor='white')
        # plt.plot(predict_bound_test[:, 0], 'y', label='predict_lower_bound')
        # plt.plot(predict_bound_test[:, 1], 'y', label='predict_upper_bound')
        # plt.plot(predict_bound_test[:, 1] - predict_bound_test[:, 0], 'b', label='width')
        # plt.plot(Y_test, 'r', label='True')
        # plt.legend()
        #
        # plt.figure('train data predict plot', facecolor='white')
        # plt.plot(predict_bound_train[:, 0], 'y', label='train_lower_bound')
        # plt.plot(predict_bound_train[:, 1], 'y', label='train_upper_bound')
        # plt.plot(fc.maponezero(y_train_upper_bound, 'reverse', y_maxmin), 'g', label='structed_bound')
        # plt.plot(fc.maponezero(y_train_lower_bound, 'reverse', y_maxmin), 'g', label='structed_bound')
        # plt.plot(Y_train, 'r', label='True')
        # plt.legend()
        # plt.show()

    # save file
    if not os.path.exists(path):
        os.makedirs(path)
    file.save(path + 'lstm_experiment_data_field' + str(flag + 1) + '.xls')

    # plot
    # plt.figure('loss', facecolor='white')
    # # plt.plot(history.history['output_distance_loss'],'b',label='Distance Loss')
    # # plt.plot(history.history['output_bound_loss'],'y',label='Bound Loss')
    # plt.plot(history.history['loss'], 'r', label='Total Loss')
    # plt.legend()

    # plt.show()
