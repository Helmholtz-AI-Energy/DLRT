from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras


class Linear(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32, name='linear', **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.input_dim = input_dim

    def build_model(self):
        self.w = self.add_weight(
            shape=(self.input_dim, self.units),
            initializer='random_normal',
            trainable=True,
        )
        self.b = self.add_weight(shape=(self.units,), initializer='random_normal', trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config

    def save(self, folder_name):
        w_np = self.w.numpy()
        np.save(folder_name + '/w_out.npy', w_np)
        b_np = self.b.numpy()
        np.save(folder_name + '/b_out.npy', b_np)
        return 0

    def load(self, folder_name):
        a_np = np.load(folder_name + '/w_out.npy')
        self.w = tf.Variable(
            initial_value=a_np,
            trainable=True,
            name='w_',
            dtype=tf.float32,
        )
        b_np = np.load(folder_name + '/b_out.npy')
        self.b = tf.Variable(
            initial_value=b_np,
            trainable=True,
            name='b_',
            dtype=tf.float32,
        )


class DenseLinear(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32, name='denselinear', **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.input_dim = input_dim

    def build_model(self):
        self.w = self.add_weight(
            shape=(self.input_dim, self.units),
            initializer='random_normal',
            trainable=True,
        )
        self.b = self.add_weight(shape=(self.units,), initializer='random_normal', trainable=True)

    def call(self, inputs):
        return tf.keras.activations.relu(tf.matmul(inputs, self.w) + self.b)

    def get_config(self):
        config = super(Linear, self).get_config()
        config.update({'units': self.units})
        return config

    def save(self, folder_name, layer_id):
        w_np = self.w.numpy()
        np.save(folder_name + '/w' + str(layer_id) + '.npy', w_np)
        b_np = self.b.numpy()
        np.save(folder_name + '/b' + str(layer_id) + '.npy', b_np)
        return 0

    def load(self, folder_name, layer_id):
        a_np = np.load(folder_name + '/w' + str(layer_id) + '.npy')
        self.w = tf.Variable(
            initial_value=a_np,
            trainable=True,
            name='w_',
            dtype=tf.float32,
        )
        b_np = np.load(folder_name + '/b' + str(layer_id) + '.npy')
        self.b = tf.Variable(
            initial_value=b_np,
            trainable=True,
            name='b_',
            dtype=tf.float32,
        )


class DLRALayerAdaptive(keras.layers.Layer):
    def __init__(
        self,
        input_dim: int,
        units=32,
        low_rank=10,
        epsAdapt=0.1,
        rmax_total=100,
        name='dlra_block',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.epsAdapt = epsAdapt  # for unconventional integrator
        self.units = units
        # tf.Variable(value=low_rank, dtype=tf.int32, trainable=False)
        self.low_rank = low_rank
        # tf.constant(value=rmax_total, dtype=tf.int32)
        self.rmax_total = rmax_total

        self.rmax_total = min(self.rmax_total, int(min(self.units, input_dim) / 2))
        print(
            'Max Rank has been set to:'
            + str(
                self.rmax_total,
            )
            + ' due to layer layout. Max allowed rank is min(in_dim,out_dim)/2',
        )
        if self.low_rank > self.rmax_total:
            self.low_rank = int(self.rmax_total)
        print('Start rank has been set to: ' + str(self.low_rank) + ' to match max rank')
        self.input_dim = input_dim

    def build_model(self):

        self.k = self.add_weight(
            shape=(self.input_dim, self.rmax_total),
            initializer='random_normal',
            trainable=True,
            name='k_',
        )
        self.l_t = self.add_weight(
            shape=(self.rmax_total, self.units),
            initializer='random_normal',
            trainable=True,
            name='lt_',
        )
        self.s = self.add_weight(
            shape=(2 * self.rmax_total, 2 * self.rmax_total),
            initializer='random_normal',
            trainable=True,
            name='s_',
        )
        self.b = self.add_weight(shape=(self.units,), initializer='random_normal', trainable=True, name='b_')
        # auxiliary variables
        self.aux_b = self.add_weight(
            shape=(self.units,),
            initializer='random_normal',
            trainable=False,
            name='aux_b',
        )
        self.aux_b.assign(self.b)  # non trainable bias or k and l step

        self.aux_U = self.add_weight(
            shape=(self.input_dim, self.rmax_total),
            initializer='random_normal',
            trainable=False,
            name='aux_U',
        )
        self.aux_Unp1 = self.add_weight(
            shape=(self.input_dim, 2 * self.rmax_total),
            initializer='random_normal',
            trainable=False,
            name='aux_Unp1',
        )
        self.aux_Vt = self.add_weight(
            shape=(self.rmax_total, self.units),
            initializer='random_normal',
            trainable=False,
            name='Vt',
        )
        self.aux_Vtnp1 = self.add_weight(
            shape=(2 * self.rmax_total, self.units),
            initializer='random_normal',
            trainable=False,
            name='vtnp1',
        )
        self.aux_N = self.add_weight(
            shape=(2 * self.rmax_total, self.rmax_total),
            initializer='random_normal',
            trainable=False,
            name='aux_N',
        )
        self.aux_M = self.add_weight(
            shape=(2 * self.rmax_total, self.rmax_total),
            initializer='random_normal',
            trainable=False,
            name='aux_M',
        )
        # Todo: initializer with low rank

    @tf.function
    def call(self, inputs, step: int = 0):
        """
        :param
        inputs: layer         input
        :param
        step: step         counter: k := 0, l := 1, s := 2
        :return:
        """
        if step == 0:  # k-step
            z = tf.matmul(tf.matmul(inputs, self.k[:, : self.low_rank]), self.aux_Vt[: self.low_rank, :])
            z = z + self.aux_b
        elif step == 1:  # l-step
            z = tf.matmul(tf.matmul(inputs, self.aux_U[:, : self.low_rank]), self.l_t[: self.low_rank, :])
            z = z + self.aux_b
        else:  # s-step
            z = tf.matmul(
                tf.matmul(
                    tf.matmul(inputs, self.aux_Unp1[:, : 2 * self.low_rank]),
                    self.s[: 2 * self.low_rank, : 2 * self.low_rank],
                ),
                self.aux_Vtnp1[: 2 * self.low_rank, :],
            )
            z = z + self.b

        return tf.keras.activations.relu(z)

    # @tf.function
    def k_step_preprocessing(self):
        k = tf.matmul(self.aux_U[:, : self.low_rank], self.s[: self.low_rank, : self.low_rank])
        self.k[:, : self.low_rank].assign(k)
        return 0

    # @tf.function
    def k_step_postprocessing_adapt(self):
        k_extended = tf.concat((self.k[:, : self.low_rank], self.aux_U[:, : self.low_rank]), axis=1)
        aux_Unp1, _ = tf.linalg.qr(k_extended)
        self.aux_Unp1[:, : 2 * self.low_rank].assign(aux_Unp1)
        aux_N = tf.matmul(tf.transpose(self.aux_Unp1[:, : 2 * self.low_rank]), self.aux_U[:, : self.low_rank])
        self.aux_N[: 2 * self.low_rank, : self.low_rank].assign(aux_N)
        return 0

    # @tf.function
    def l_step_preprocessing(self):
        l_t = tf.matmul(self.s[: self.low_rank, : self.low_rank], self.aux_Vt[: self.low_rank, :])
        # = tf.Variable(initial_value=l_t, trainable=True, name="lt_")
        self.l_t[: self.low_rank, :].assign(l_t)
        return 0

    # @tf.function
    def l_step_postprocessing_adapt(self):
        l_extended = tf.concat(
            (tf.transpose(self.l_t[: self.low_rank, :]), tf.transpose(self.aux_Vt[: self.low_rank, :])),
            axis=1,
        )
        aux_Vnp1, _ = tf.linalg.qr(l_extended)
        self.aux_Vtnp1[: 2 * self.low_rank, :].assign(tf.transpose(aux_Vnp1))
        aux_M = tf.matmul(
            self.aux_Vtnp1[: 2 * self.low_rank, :],
            tf.transpose(self.aux_Vt[: self.low_rank, :]),
        )
        self.aux_M[: 2 * self.low_rank, : self.low_rank].assign(aux_M)
        return 0

    # @tf.function
    def s_step_preprocessing(self):
        s = tf.matmul(
            tf.matmul(
                self.aux_N[: 2 * self.low_rank, : self.low_rank],
                self.s[: self.low_rank, : self.low_rank],
            ),
            tf.transpose(self.aux_M[: 2 * self.low_rank, : self.low_rank]),
        )
        self.s[: 2 * self.low_rank, : 2 * self.low_rank].assign(s)

        return 0

    # @tf.function
    def rank_adaption(self):
        # 1) compute SVD of S
        # d=singular values, u2 = left singuar vecs, v2= right singular vecs
        s_small = self.s[: 2 * self.low_rank, : 2 * self.low_rank]
        d, u2, v2 = tf.linalg.svd(s_small)

        tmp = 0.0
        # absolute value treshold (try also relative one)
        tol = self.epsAdapt * tf.linalg.norm(d)
        rmax = int(tf.floor(d.shape[0] / 2))
        for j in range(0, 2 * rmax - 1):
            tmp = tf.linalg.norm(d[j : 2 * rmax - 1])
            if tmp < tol:
                rmax = j
                break

        rmax = tf.minimum(rmax, self.rmax_total)
        rmax = tf.maximum(rmax, 2)

        # update s
        self.s[:rmax, :rmax].assign(tf.linalg.tensor_diag(d[:rmax]))

        # update u and v
        self.aux_U[:, :rmax].assign(tf.matmul(self.aux_Unp1[:, : 2 * self.low_rank], u2[:, :rmax]))
        self.aux_Vt[:rmax, :].assign(tf.matmul(v2[:rmax, :], self.aux_Vtnp1[: 2 * self.low_rank, :]))
        self.low_rank = int(rmax)

        # update bias
        self.aux_b.assign(self.b)
        return 0

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        config.update({'low_rank': self.low_rank})
        return config

    def save(self, folder_name, layer_id):
        # main_variables
        k_np = self.k[:, : self.low_rank].numpy()
        np.save(folder_name + '/k' + str(layer_id) + '.npy', k_np)
        l_t_np = self.l_t[: self.low_rank, : self.low_rank].numpy()
        np.save(folder_name + '/l_t' + str(layer_id) + '.npy', l_t_np)
        s_np = self.s[: 2 * self.low_rank, : 2 * self.low_rank].numpy()
        np.save(folder_name + '/s' + str(layer_id) + '.npy', s_np)
        b_np = self.b.numpy()
        np.save(folder_name + '/b' + str(layer_id) + '.npy', b_np)
        # aux_variables
        aux_U_np = self.aux_U[:, : self.low_rank].numpy()
        np.save(folder_name + '/aux_U' + str(layer_id) + '.npy', aux_U_np)
        aux_Unp1_np = self.aux_Unp1[:, : 2 * self.low_rank].numpy()
        np.save(folder_name + '/aux_Unp1' + str(layer_id) + '.npy', aux_Unp1_np)
        aux_Vt_np = self.aux_Vt[:, : self.low_rank].numpy()
        np.save(folder_name + '/aux_Vt' + str(layer_id) + '.npy', aux_Vt_np)
        aux_Vtnp1_np = self.aux_Vtnp1[:, : 2 * self.low_rank].numpy()
        np.save(folder_name + '/aux_Vtnp1' + str(layer_id) + '.npy', aux_Vtnp1_np)
        aux_N_np = self.aux_N[: 2 * self.low_rank, : self.low_rank].numpy()
        np.save(folder_name + '/aux_N' + str(layer_id) + '.npy', aux_N_np)
        aux_M_np = self.aux_M[: 2 * self.low_rank, : self.low_rank].numpy()
        np.save(folder_name + '/aux_M' + str(layer_id) + '.npy', aux_M_np)
        return 0

    def load(self, folder_name, layer_id):

        # main variables
        k_np = np.load(folder_name + '/k' + str(layer_id) + '.npy')
        self.low_rank = k_np.shape[1]
        self.k = tf.Variable(
            initial_value=k_np,
            trainable=True,
            name='k_',
            dtype=tf.float32,
        )
        l_t_np = np.load(folder_name + '/l_t' + str(layer_id) + '.npy')
        self.l_t = tf.Variable(
            initial_value=l_t_np,
            trainable=True,
            name='lt_',
            dtype=tf.float32,
        )
        s_np = np.load(folder_name + '/s' + str(layer_id) + '.npy')
        self.s = tf.Variable(
            initial_value=s_np,
            trainable=True,
            name='s_',
            dtype=tf.float32,
        )
        # aux variables
        aux_U_np = np.load(folder_name + '/aux_U' + str(layer_id) + '.npy')
        self.aux_U = tf.Variable(
            initial_value=aux_U_np,
            trainable=True,
            name='aux_U',
            dtype=tf.float32,
        )
        aux_Unp1_np = np.load(folder_name + '/aux_Unp1' + str(layer_id) + '.npy')
        self.aux_Unp1 = tf.Variable(
            initial_value=aux_Unp1_np,
            trainable=True,
            name='aux_Unp1',
            dtype=tf.float32,
        )
        Vt_np = np.load(folder_name + '/aux_Vt' + str(layer_id) + '.npy')
        self.aux_Vt = tf.Variable(
            initial_value=Vt_np,
            trainable=True,
            name='Vt',
            dtype=tf.float32,
        )
        vtnp1_np = np.load(folder_name + '/aux_Vtnp1' + str(layer_id) + '.npy')
        self.aux_Vtnp1 = tf.Variable(
            initial_value=vtnp1_np,
            trainable=True,
            name='vtnp1',
            dtype=tf.float32,
        )
        aux_N_np = np.load(folder_name + '/aux_N' + str(layer_id) + '.npy')
        self.aux_N = tf.Variable(
            initial_value=aux_N_np,
            trainable=True,
            name='aux_N',
            dtype=tf.float32,
        )
        aux_M_np = np.load(folder_name + '/aux_M' + str(layer_id) + '.npy')
        self.aux_M = tf.Variable(
            initial_value=aux_M_np,
            trainable=True,
            name='aux_M',
            dtype=tf.float32,
        )
        return 0

    def get_rank(self):
        return self.low_rank

    def get_weights_num(self):
        full_rank_weights = self.input_dim * self.units
        low_rank_weights = self.low_rank * (self.input_dim + self.units + self.low_rank)
        return low_rank_weights, full_rank_weights


class DLRALayerLinear(keras.layers.Layer):
    def __init__(
        self,
        input_dim: int,
        units=32,
        low_rank=10,
        name='dlra_block',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.low_rank = low_rank
        self.input_dim = input_dim

    def build_model(self):

        self.k = self.add_weight(
            shape=(self.input_dim, self.low_rank),
            initializer='random_normal',
            trainable=True,
            name='k_',
        )
        self.l_t = self.add_weight(
            shape=(self.low_rank, self.units),
            initializer='random_normal',
            trainable=True,
            name='lt_',
        )
        self.s = self.add_weight(
            shape=(self.low_rank, self.low_rank),
            initializer='random_normal',
            trainable=True,
            name='s_',
        )
        self.b = self.add_weight(shape=(self.units,), initializer='random_normal', trainable=True, name='b_')
        # auxiliary variables
        self.aux_b = self.add_weight(
            shape=(self.units,),
            initializer='random_normal',
            trainable=False,
            name='aux_b',
        )
        self.aux_b.assign(self.b)  # non trainable bias or k and l step

        self.aux_U = self.add_weight(
            shape=(self.input_dim, self.low_rank),
            initializer='random_normal',
            trainable=False,
            name='aux_U',
        )
        self.aux_Unp1 = self.add_weight(
            shape=(self.input_dim, self.low_rank),
            initializer='random_normal',
            trainable=False,
            name='aux_Unp1',
        )
        self.aux_Vt = self.add_weight(
            shape=(self.low_rank, self.units),
            initializer='random_normal',
            trainable=False,
            name='aux_Vt',
        )
        self.aux_Vtnp1 = self.add_weight(
            shape=(self.low_rank, self.units),
            initializer='random_normal',
            trainable=False,
            name='aux_Vtnp1',
        )
        self.aux_N = self.add_weight(
            shape=(self.low_rank, self.low_rank),
            initializer='random_normal',
            trainable=False,
            name='aux_N',
        )
        self.aux_M = self.add_weight(
            shape=(self.low_rank, self.low_rank),
            initializer='random_normal',
            trainable=False,
            name='aux_M',
        )
        # Todo: initializer with low rank
        return 0

    @tf.function
    def call(self, inputs, step: int = 0):
        """
        :param inputs: layer input
        :param step: step conter: k:= 0, l:=1, s:=2
        :return:
        """

        if step == 0:  # k-step
            z = tf.matmul(tf.matmul(inputs, self.k), self.aux_Vt) + self.aux_b
        elif step == 1:  # l-step
            z = tf.matmul(tf.matmul(inputs, self.aux_U), self.l_t) + self.aux_b
        else:  # s-step
            z = tf.matmul(tf.matmul(tf.matmul(inputs, self.aux_Unp1), self.s), self.aux_Vtnp1) + self.b
        return z

    @tf.function
    def k_step_preprocessing(self):
        # update bias
        self.aux_b.assign(self.b)

        k = tf.matmul(self.aux_U, self.s)
        # = tf.Variable(initial_value=k, trainable=True, name="k_")
        self.k.assign(k)
        return 0

    @tf.function
    def k_step_postprocessing(self):
        aux_Unp1, _ = tf.linalg.qr(self.k)
        # = tf.Variable(initial_value=aux_Unp1, trainable=False, name="aux_Unp1")
        self.aux_Unp1.assign(aux_Unp1)
        N = tf.matmul(tf.transpose(self.aux_Unp1), self.aux_U)
        self.aux_N.assign(N)
        return 0

    @tf.function
    def l_step_preprocessing(self):
        l_t = tf.matmul(self.s, self.aux_Vt)
        # = tf.Variable(initial_value=l_t, trainable=True, name="lt_")
        self.l_t.assign(l_t)
        return 0

    @tf.function
    def l_step_postprocessing(self):
        aux_Vtnp1, _ = tf.linalg.qr(tf.transpose(self.l_t))
        self.aux_Vtnp1.assign(tf.transpose(aux_Vtnp1))
        M = tf.matmul(self.aux_Vtnp1, tf.transpose(self.aux_Vt))
        self.aux_M.assign(M)
        return 0

    @tf.function
    def s_step_preprocessing(self):
        self.aux_U.assign(self.aux_Unp1)
        self.aux_Vt.assign(self.aux_Vtnp1)
        s = tf.matmul(tf.matmul(self.aux_N, self.s), tf.transpose(self.aux_M))
        # = tf.Variable(initial_value=s, trainable=True, name="s_")
        self.s.assign(s)
        return 0

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        config.update({'low_rank': self.low_rank})
        return config

    def save(self, folder_name, layer_id):
        # main_variables
        k_np = self.k.numpy()
        np.save(folder_name + '/k' + str(layer_id) + '.npy', k_np)
        l_t_np = self.l_t.numpy()
        np.save(folder_name + '/l_t' + str(layer_id) + '.npy', l_t_np)
        s_np = self.s.numpy()
        np.save(folder_name + '/s' + str(layer_id) + '.npy', s_np)
        b_np = self.b.numpy()
        np.save(folder_name + '/b' + str(layer_id) + '.npy', b_np)
        # aux_variables
        aux_U_np = self.aux_U.numpy()
        np.save(folder_name + '/aux_U' + str(layer_id) + '.npy', aux_U_np)
        aux_Unp1_np = self.aux_Unp1.numpy()
        np.save(folder_name + '/aux_Unp1' + str(layer_id) + '.npy', aux_Unp1_np)
        aux_Vt_np = self.aux_Vt.numpy()
        np.save(folder_name + '/aux_Vt' + str(layer_id) + '.npy', aux_Vt_np)
        aux_Vtnp1_np = self.aux_Vtnp1.numpy()
        np.save(folder_name + '/aux_Vtnp1' + str(layer_id) + '.npy', aux_Vtnp1_np)
        aux_N_np = self.aux_N.numpy()
        np.save(folder_name + '/aux_N' + str(layer_id) + '.npy', aux_N_np)
        aux_M_np = self.aux_M.numpy()
        np.save(folder_name + '/aux_M' + str(layer_id) + '.npy', aux_M_np)
        return 0

    def load(self, folder_name, layer_id):

        # main variables
        k_np = np.load(folder_name + '/k' + str(layer_id) + '.npy')
        self.low_rank = k_np.shape[1]
        self.k = tf.Variable(
            initial_value=k_np,
            trainable=True,
            name='k_',
            dtype=tf.float32,
        )
        l_t_np = np.load(folder_name + '/l_t' + str(layer_id) + '.npy')
        self.l_t = tf.Variable(
            initial_value=l_t_np,
            trainable=True,
            name='lt_',
            dtype=tf.float32,
        )
        s_np = np.load(folder_name + '/s' + str(layer_id) + '.npy')
        self.s = tf.Variable(
            initial_value=s_np,
            trainable=True,
            name='s_',
            dtype=tf.float32,
        )
        bias = np.load(folder_name + '/b' + str(layer_id) + '.npy')
        self.b = tf.Variable(
            initial_value=bias,
            trainable=True,
            name='b_',
            dtype=tf.float32,
        )

        # aux variables
        aux_U_np = np.load(folder_name + '/aux_U' + str(layer_id) + '.npy')
        self.aux_U = tf.Variable(
            initial_value=aux_U_np,
            trainable=False,
            name='aux_U',
            dtype=tf.float32,
        )
        aux_Unp1_np = np.load(folder_name + '/aux_Unp1' + str(layer_id) + '.npy')
        self.aux_Unp1 = tf.Variable(
            initial_value=aux_Unp1_np,
            trainable=False,
            name='aux_Unp1',
            dtype=tf.float32,
        )
        Vt_np = np.load(folder_name + '/aux_Vt' + str(layer_id) + '.npy')
        self.aux_Vt = tf.Variable(
            initial_value=Vt_np,
            trainable=False,
            name='aux_Vt',
            dtype=tf.float32,
        )
        vtnp1_np = np.load(folder_name + '/aux_Vtnp1' + str(layer_id) + '.npy')
        self.aux_Vtnp1 = tf.Variable(
            initial_value=vtnp1_np,
            trainable=False,
            name='aux_Vtnp1',
            dtype=tf.float32,
        )
        aux_N_np = np.load(folder_name + '/aux_N' + str(layer_id) + '.npy')
        self.aux_N = tf.Variable(
            initial_value=aux_N_np,
            trainable=False,
            name='aux_N',
            dtype=tf.float32,
        )
        aux_M_np = np.load(folder_name + '/aux_M' + str(layer_id) + '.npy')
        self.aux_M = tf.Variable(
            initial_value=aux_M_np,
            trainable=False,
            name='aux_M',
            dtype=tf.float32,
        )

        # build model, but only
        # auxiliary variables, since in adaptive rank training, these have different shapes

        self.aux_Unp1 = self.add_weight(
            shape=(self.input_dim, self.low_rank),
            initializer='random_normal',
            trainable=False,
            name='aux_Unp1',
        )

        self.aux_Vtnp1 = self.add_weight(
            shape=(self.low_rank, self.units),
            initializer='random_normal',
            trainable=False,
            name='aux_Vtnp1',
        )
        self.aux_N = self.add_weight(
            shape=(self.low_rank, self.low_rank),
            initializer='random_normal',
            trainable=False,
            name='aux_N',
        )
        self.aux_M = self.add_weight(
            shape=(self.low_rank, self.low_rank),
            initializer='random_normal',
            trainable=False,
            name='aux_M',
        )

        return 0

    def load_from_fullW(self, folder_name, layer_id, rank):

        W_mat = np.load(folder_name + '/w_' + str(layer_id) + '.npy')
        # d=singular values, u2 = left singuar vecs, v2= right singular vecss
        d, u, v = tf.linalg.svd(W_mat)

        s_init = tf.linalg.tensor_diag(d[:rank])
        u_init = u[:, :rank]
        v_init = u[:rank, :]
        self.s = tf.Variable(initial_value=s_init, trainable=True, name='s_', dtype=tf.float32)
        self.aux_Vt = tf.Variable(initial_value=v_init, trainable=True, name='Vt', dtype=tf.float32)
        self.aux_U = tf.Variable(initial_value=u_init, trainable=True, name='aux_U', dtype=tf.float32)

        self.k_step_preprocessing()
        self.l_step_preprocessing()

        return 0

    def get_rank(self):
        return self.low_rank

    def get_weights_num(self):
        full_rank_weights = self.input_dim * self.units
        low_rank_weights = self.low_rank * (self.input_dim + self.units + self.low_rank)
        return low_rank_weights, full_rank_weights


class DLRALayerAdaptiveLinear(keras.layers.Layer):
    def __init__(
        self,
        input_dim: int,
        units=32,
        low_rank=10,
        epsAdapt=0.1,
        rmax_total=100,
        name='dlra_block',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.epsAdapt = epsAdapt  # for unconventional integrator
        self.units = units
        # tf.Variable(value=low_rank, dtype=tf.int32, trainable=False)
        self.low_rank = low_rank
        # tf.constant(value=rmax_total, dtype=tf.int32)
        self.rmax_total = rmax_total

        self.rmax_total = min(self.rmax_total, int(min(self.units, input_dim) / 2))
        print(
            'Max Rank has been set to:'
            + str(
                self.rmax_total,
            )
            + ' due to layer layout. Max allowed rank is min(in_dim,out_dim)/2',
        )
        if self.low_rank > self.rmax_total:
            self.low_rank = int(self.rmax_total)
        print('Start rank has been set to: ' + str(self.low_rank) + ' to match max rank')
        self.input_dim = input_dim

    def build_model(self):

        self.k = self.add_weight(
            shape=(self.input_dim, self.rmax_total),
            initializer='random_normal',
            trainable=True,
            name='k_',
        )
        self.l_t = self.add_weight(
            shape=(self.rmax_total, self.units),
            initializer='random_normal',
            trainable=True,
            name='lt_',
        )
        self.s = self.add_weight(
            shape=(2 * self.rmax_total, 2 * self.rmax_total),
            initializer='random_normal',
            trainable=True,
            name='s_',
        )
        self.b = self.add_weight(shape=(self.units,), initializer='random_normal', trainable=True, name='b_')
        # auxiliary variables
        self.aux_b = self.add_weight(
            shape=(self.units,),
            initializer='random_normal',
            trainable=False,
            name='aux_b',
        )
        self.aux_b.assign(self.b)  # non trainable bias or k and l step

        self.aux_U = self.add_weight(
            shape=(self.input_dim, self.rmax_total),
            initializer='random_normal',
            trainable=False,
            name='aux_U',
        )
        self.aux_Unp1 = self.add_weight(
            shape=(self.input_dim, 2 * self.rmax_total),
            initializer='random_normal',
            trainable=False,
            name='aux_Unp1',
        )
        self.aux_Vt = self.add_weight(
            shape=(self.rmax_total, self.units),
            initializer='random_normal',
            trainable=False,
            name='Vt',
        )
        self.aux_Vtnp1 = self.add_weight(
            shape=(2 * self.rmax_total, self.units),
            initializer='random_normal',
            trainable=False,
            name='vtnp1',
        )
        self.aux_N = self.add_weight(
            shape=(2 * self.rmax_total, self.rmax_total),
            initializer='random_normal',
            trainable=False,
            name='aux_N',
        )
        self.aux_M = self.add_weight(
            shape=(2 * self.rmax_total, self.rmax_total),
            initializer='random_normal',
            trainable=False,
            name='aux_M',
        )
        # Todo: initializer with low rank

    # @tf.function
    def call(self, inputs, step: int = 0):
        """
        :param
        inputs: layer         input
        :param
        step: step         counter: k := 0, l := 1, s := 2
        :return:
        """
        if step == 0:  # k-step
            z = tf.matmul(tf.matmul(inputs, self.k[:, : self.low_rank]), self.aux_Vt[: self.low_rank, :])
            z = z + self.aux_b
        elif step == 1:  # l-step
            z = tf.matmul(tf.matmul(inputs, self.aux_U[:, : self.low_rank]), self.l_t[: self.low_rank, :])
            z = z + self.aux_b
        else:  # s-step
            z = tf.matmul(
                tf.matmul(
                    tf.matmul(inputs, self.aux_Unp1[:, : 2 * self.low_rank]),
                    self.s[: 2 * self.low_rank, : 2 * self.low_rank],
                ),
                self.aux_Vtnp1[: 2 * self.low_rank, :],
            )
            z = z + self.b

        return z

    # @tf.function
    def k_step_preprocessing(self):
        k = tf.matmul(self.aux_U[:, : self.low_rank], self.s[: self.low_rank, : self.low_rank])
        self.k[:, : self.low_rank].assign(k)
        return 0

    # @tf.function
    def k_step_postprocessing_adapt(self):
        k_extended = tf.concat((self.k[:, : self.low_rank], self.aux_U[:, : self.low_rank]), axis=1)
        aux_Unp1, _ = tf.linalg.qr(k_extended)
        self.aux_Unp1[:, : 2 * self.low_rank].assign(aux_Unp1)
        aux_N = tf.matmul(tf.transpose(self.aux_Unp1[:, : 2 * self.low_rank]), self.aux_U[:, : self.low_rank])
        self.aux_N[: 2 * self.low_rank, : self.low_rank].assign(aux_N)
        return 0

    # @tf.function
    def l_step_preprocessing(self):
        l_t = tf.matmul(self.s[: self.low_rank, : self.low_rank], self.aux_Vt[: self.low_rank, :])
        # = tf.Variable(initial_value=l_t, trainable=True, name="lt_")
        self.l_t[: self.low_rank, :].assign(l_t)
        return 0

    # @tf.function
    def l_step_postprocessing_adapt(self):
        l_extended = tf.concat(
            (tf.transpose(self.l_t[: self.low_rank, :]), tf.transpose(self.aux_Vt[: self.low_rank, :])),
            axis=1,
        )
        aux_Vnp1, _ = tf.linalg.qr(l_extended)
        self.aux_Vtnp1[: 2 * self.low_rank, :].assign(tf.transpose(aux_Vnp1))
        aux_M = tf.matmul(
            self.aux_Vtnp1[: 2 * self.low_rank, :],
            tf.transpose(self.aux_Vt[: self.low_rank, :]),
        )
        self.aux_M[: 2 * self.low_rank, : self.low_rank].assign(aux_M)
        return 0

    # @tf.function
    def s_step_preprocessing(self):
        s = tf.matmul(
            tf.matmul(
                self.aux_N[: 2 * self.low_rank, : self.low_rank],
                self.s[: self.low_rank, : self.low_rank],
            ),
            tf.transpose(self.aux_M[: 2 * self.low_rank, : self.low_rank]),
        )
        self.s[: 2 * self.low_rank, : 2 * self.low_rank].assign(s)

        return 0

    # @tf.function
    def rank_adaption(self):
        # 1) compute SVD of S
        # d=singular values, u2 = left singuar vecs, v2= right singular vecs
        s_small = self.s[: 2 * self.low_rank, : 2 * self.low_rank]
        d, u2, v2 = tf.linalg.svd(s_small)

        # absolute value treshold (try also relative one)
        tol = self.epsAdapt * tf.linalg.norm(d)
        rmax = int(tf.floor(d.shape[0] / 2))
        for j in range(0, 2 * rmax - 1):
            tmp = tf.linalg.norm(d[j : 2 * rmax - 1])
            if tmp < tol:
                rmax = j
                break

        rmax = tf.minimum(rmax, self.rmax_total)
        rmax = tf.maximum(rmax, 2)

        # update s
        self.s[:rmax, :rmax].assign(tf.linalg.tensor_diag(d[:rmax]))

        # update u and v
        self.aux_U[:, :rmax].assign(tf.matmul(self.aux_Unp1[:, : 2 * self.low_rank], u2[:, :rmax]))
        self.aux_Vt[:rmax, :].assign(tf.matmul(v2[:rmax, :], self.aux_Vtnp1[: 2 * self.low_rank, :]))
        self.low_rank = int(rmax)

        # update bias
        self.aux_b.assign(self.b)
        return 0

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        config.update({'low_rank': self.low_rank})
        return config

    def save(self, folder_name, layer_id):
        # main_variables
        k_np = self.k[:, : self.low_rank].numpy()
        np.save(folder_name + '/k' + str(layer_id) + '.npy', k_np)
        l_t_np = self.l_t[: self.low_rank, : self.low_rank].numpy()
        np.save(folder_name + '/l_t' + str(layer_id) + '.npy', l_t_np)
        s_np = self.s[: 2 * self.low_rank, : 2 * self.low_rank].numpy()
        np.save(folder_name + '/s' + str(layer_id) + '.npy', s_np)
        b_np = self.b.numpy()
        np.save(folder_name + '/b' + str(layer_id) + '.npy', b_np)
        # aux_variables
        aux_U_np = self.aux_U[:, : self.low_rank].numpy()
        np.save(folder_name + '/aux_U' + str(layer_id) + '.npy', aux_U_np)
        aux_Unp1_np = self.aux_Unp1[:, : 2 * self.low_rank].numpy()
        np.save(folder_name + '/aux_Unp1' + str(layer_id) + '.npy', aux_Unp1_np)
        aux_Vt_np = self.aux_Vt[:, : self.low_rank].numpy()
        np.save(folder_name + '/aux_Vt' + str(layer_id) + '.npy', aux_Vt_np)
        aux_Vtnp1_np = self.aux_Vtnp1[:, : 2 * self.low_rank].numpy()
        np.save(folder_name + '/aux_Vtnp1' + str(layer_id) + '.npy', aux_Vtnp1_np)
        aux_N_np = self.aux_N[: 2 * self.low_rank, : self.low_rank].numpy()
        np.save(folder_name + '/aux_N' + str(layer_id) + '.npy', aux_N_np)
        aux_M_np = self.aux_M[: 2 * self.low_rank, : self.low_rank].numpy()
        np.save(folder_name + '/aux_M' + str(layer_id) + '.npy', aux_M_np)
        return 0

    def load(self, folder_name, layer_id):

        # main variables
        k_np = np.load(folder_name + '/k' + str(layer_id) + '.npy')
        self.low_rank = k_np.shape[1]
        self.k = tf.Variable(
            initial_value=k_np,
            trainable=True,
            name='k_',
            dtype=tf.float32,
        )
        l_t_np = np.load(folder_name + '/l_t' + str(layer_id) + '.npy')
        self.l_t = tf.Variable(
            initial_value=l_t_np,
            trainable=True,
            name='lt_',
            dtype=tf.float32,
        )
        s_np = np.load(folder_name + '/s' + str(layer_id) + '.npy')
        self.s = tf.Variable(
            initial_value=s_np,
            trainable=True,
            name='s_',
            dtype=tf.float32,
        )
        # aux variables
        aux_U_np = np.load(folder_name + '/aux_U' + str(layer_id) + '.npy')
        self.aux_U = tf.Variable(
            initial_value=aux_U_np,
            trainable=True,
            name='aux_U',
            dtype=tf.float32,
        )
        aux_Unp1_np = np.load(folder_name + '/aux_Unp1' + str(layer_id) + '.npy')
        self.aux_Unp1 = tf.Variable(
            initial_value=aux_Unp1_np,
            trainable=True,
            name='aux_Unp1',
            dtype=tf.float32,
        )
        Vt_np = np.load(folder_name + '/aux_Vt' + str(layer_id) + '.npy')
        self.aux_Vt = tf.Variable(
            initial_value=Vt_np,
            trainable=True,
            name='Vt',
            dtype=tf.float32,
        )
        vtnp1_np = np.load(folder_name + '/aux_Vtnp1' + str(layer_id) + '.npy')
        self.aux_Vtnp1 = tf.Variable(
            initial_value=vtnp1_np,
            trainable=True,
            name='vtnp1',
            dtype=tf.float32,
        )
        aux_N_np = np.load(folder_name + '/aux_N' + str(layer_id) + '.npy')
        self.aux_N = tf.Variable(
            initial_value=aux_N_np,
            trainable=True,
            name='aux_N',
            dtype=tf.float32,
        )
        aux_M_np = np.load(folder_name + '/aux_M' + str(layer_id) + '.npy')
        self.aux_M = tf.Variable(
            initial_value=aux_M_np,
            trainable=True,
            name='aux_M',
            dtype=tf.float32,
        )
        return 0

    def get_rank(self):
        return self.low_rank

    def get_weights_num(self):
        full_rank_weights = self.input_dim * self.units
        low_rank_weights = self.low_rank * (self.input_dim + self.units + self.low_rank)
        return low_rank_weights, full_rank_weights
