# import pickle
# import numpy as np
# import torch
# from torch_geometric.data import Data
# f = open('dataset/395/train_335.pkl', 'rb')
# train_data = pickle.load(f)
# train_protein_name = list(train_data.keys())
# # '>1f60B'
# print(train_data['>1f60B'].keys())
# # dict_keys(['seq', 'label', 'seq_emb', 'structure_emb', 'dssp', 'hmm', 'pssm', 's2', '_1024_dssp_hmm_pssm', 'graph'])
# label = []
# adj = [[], []]
# feature = []
# for name in train_protein_name:
#     _adj = np.array(train_data['{}'.format(name)]['graph'], dtype=int)
#     result_list = [[x + len(label) for x in y] for y in _adj]
#     adj[0] += result_list[0]
#     adj[1] += result_list[1]
#     _label = np.array(train_data['{}'.format(name)]['label'], dtype=int).tolist()
#     label += _label
#     _feature = train_data['{}'.format(name)]['seq_emb']
#     feature.extend(_feature)
# train_number = len(label)
# print(len(label), sum(label))
#
# f = open('dataset/395/Test_60.pkl', 'rb')
# test_data = pickle.load(f)
# test_protein_name = list(test_data.keys())
# for name in test_protein_name:
#     _adj = np.array(test_data['{}'.format(name)]['graph'], dtype=int)
#     result_list = [[x + len(label) for x in y] for y in _adj]
#     adj[0] += result_list[0]
#     adj[1] += result_list[1]
#     _label = np.array(test_data['{}'.format(name)]['label'], dtype=int).tolist()
#     label += _label
#     _feature = test_data['{}'.format(name)]['seq_emb']
#     feature.extend(_feature)
#
# train_mask = np.zeros(len(label))
# train_mask[:train_number] = [x + 1 for x in train_mask[:train_number]]
# test_mask = np.zeros(len(label))
# test_mask[train_number:] = [x + 1 for x in train_mask[train_number:]]
# print(len(label), sum(label))
#
# feature = torch.Tensor(feature)
# adj = torch.Tensor(adj)
# label = torch.Tensor(label)
# train_mask = torch.Tensor(train_mask)
# val_mask = torch.Tensor(test_mask)
# test_mask = torch.Tensor(test_mask)
# data = Data(x=feature, edge_index=adj, y=label, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
# print(data)


# import numpy as np
# import torch
# from torch_geometric.data import Data
# from keras.layers import Dense, Input
# from keras.models import Model
#
#
# def DeepAE1(x_train):
#     encoding_dim = 256
#     input_img = Input(shape=(1373,))
#
#     # encoder layers
#     encoded = Dense(512, activation='relu')(input_img)
#     encoded = Dense(128, activation='relu')(encoded)
#     encoder_output = Dense(encoding_dim)(encoded)
#
#     # decoder layers
#     decoded = Dense(128, activation='relu')(encoder_output)
#     decoded = Dense(512, activation='relu')(decoded)
#     decoded = Dense(1373, activation='tanh')(decoded)
#
#     # construct the autoencoder model
#     autoencoder = Model(inputs=input_img, outputs=decoded)
#     encoder = Model(inputs=input_img, outputs=encoder_output)
#
#     # compile autoencoder
#     autoencoder.compile(optimizer='adam', loss='mse')
#     autoencoder.fit(x_train, x_train, epochs=20, batch_size=64, shuffle=True)
#     encoded_imgs = encoder.predict(x_train)
#     return encoder_output, torch.Tensor(encoded_imgs)
#
#
# def DeepAE2(x_train):
#     encoding_dim = 256
#     input_img = Input(shape=(173,))
#
#     # encoder layers
#     encoded = Dense(128, activation='relu')(input_img)
#     encoded = Dense(64, activation='relu')(encoded)
#     encoder_output = Dense(encoding_dim)(encoded)
#
#     # decoder layers
#     decoded = Dense(64, activation='relu')(encoder_output)
#     decoded = Dense(128, activation='relu')(decoded)
#     decoded = Dense(173, activation='tanh')(decoded)
#
#     # construct the autoencoder model
#     autoencoder = Model(inputs=input_img, outputs=decoded)
#     encoder = Model(inputs=input_img, outputs=encoder_output)
#
#     # compile autoencoder
#     autoencoder.compile(optimizer='adam', loss='mse')
#     autoencoder.fit(x_train, x_train, epochs=20, batch_size=64, shuffle=True)
#     encoded_imgs = encoder.predict(x_train)
#     return encoder_output, torch.Tensor(encoded_imgs)
#
#
# drug = np.loadtxt("dataset/MDAD/drugfeatures.txt")  # 1373
# microbe = np.loadtxt("dataset/MDAD/microbefeatures.txt")    # 173
# adj = np.loadtxt("dataset/MDAD/adj.txt")    # 2470
# _, drug_emb = DeepAE1(drug)
# _, microbe_emb = DeepAE2(microbe)
# adj = adj[:, :-1]
# adj[:, 1] += 1373
#
# feature = torch.cat([drug_emb, microbe_emb])
# adj = torch.LongTensor(adj).T
# data = Data(x=feature, edge_index=adj)
#
# # train_mask = np.zeros(len(label))
# # train_mask[:train_number] = [x + 1 for x in train_mask[:train_number]]
# # test_mask = np.zeros(len(label))
# # test_mask[train_number:] = [x + 1 for x in train_mask[train_number:]]
#
# print(data)


# import numpy as np
# import torch
#
# d_ss = np.loadtxt("dataset/AMHMDA/d_ss.csv", delimiter=',')   # (591, 591)
# _, d_ss = DeepAE1(d_ss)
# m_ss = np.loadtxt("dataset/AMHMDA/m_ss.csv", delimiter=',')   # (853, 853)
# _, m_ss = DeepAE2(m_ss)
# feature = torch.cat([torch.Tensor(d_ss), torch.Tensor(m_ss)])
# m_d = np.loadtxt("dataset/AMHMDA/m_d.csv", delimiter=',')     # (853, 591)
# adj = []
# for m in range(len(m_ss)):
#     for d in range(len(d_ss)):
#         if m_d[m][d] == 1:
#             adj.append([d, m+len(d_ss)])
# adj = torch.LongTensor(adj).T
# data = Data(x=feature, edge_index=adj)

