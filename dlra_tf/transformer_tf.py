from __future__ import annotations

import numpy as np
import tensorflow as tf
from networks.dense_layers import DLRALayerAdaptive
from networks.dense_layers import DLRALayerAdaptiveLinear

# global constants !!!!! DANGEROUS!!!
MAX_TOKENS = 128


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, tolerance):
        # Torch translations:
        #      d_model -> embed_dim
        #      num_heads -> num_heads
        #      tolerance -> eps_edapt for the layers
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads
        self.epsilon = tolerance

        self.wq = DLRALayerAdaptiveLinear(
            input_dim=d_model,
            units=d_model,
            low_rank=d_model // 2,
            epsAdapt=self.epsilon,
        )
        self.wk = DLRALayerAdaptiveLinear(
            input_dim=d_model,
            units=d_model,
            low_rank=d_model // 2,
            epsAdapt=self.epsilon,
        )
        self.wv = DLRALayerAdaptiveLinear(
            input_dim=d_model,
            units=d_model,
            low_rank=d_model // 2,
            epsAdapt=self.epsilon,
        )

        self.dense = DLRALayerAdaptiveLinear(
            input_dim=d_model,
            units=d_model,
            low_rank=d_model // 2,
            epsAdapt=self.epsilon,
        )

        # Build low-rank
        self.wq.build_model()
        self.wk.build_model()
        self.wv.build_model()
        self.dense.build_model()

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask, step: int):
        batch_size = tf.shape(q)[0]

        q = self.wq(q, step)  # (batch_size, seq_len, d_model)
        k = self.wk(k, step)  # (batch_size, seq_len, d_model)
        v = self.wv(v, step)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q,
            k,
            v,
            mask,
        )

        scaled_attention = tf.transpose(
            scaled_attention,
            perm=[0, 2, 1, 3],
        )  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(
            scaled_attention,
            (batch_size, -1, self.d_model),
        )  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention, step)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

    def k_step_preprocessing(self):
        self.wq.k_step_preprocessing()
        self.wk.k_step_preprocessing()
        self.wv.k_step_preprocessing()
        self.dense.k_step_preprocessing()

    def k_step_postprocessing_adapt(self):
        self.wq.k_step_postprocessing_adapt()
        self.wk.k_step_postprocessing_adapt()
        self.wv.k_step_postprocessing_adapt()
        self.dense.k_step_postprocessing_adapt()

    def l_step_preprocessing(self):
        self.wq.l_step_preprocessing()
        self.wk.l_step_preprocessing()
        self.wv.l_step_preprocessing()
        self.dense.l_step_preprocessing()

    def l_step_postprocessing_adapt(self):
        self.wq.l_step_postprocessing_adapt()
        self.wk.l_step_postprocessing_adapt()
        self.wv.l_step_postprocessing_adapt()
        self.dense.l_step_postprocessing_adapt()

    def s_step_preprocessing(self):
        self.wq.s_step_preprocessing()
        self.wk.s_step_preprocessing()
        self.wv.s_step_preprocessing()
        self.dense.s_step_preprocessing()

    def rank_adaption(self):
        self.wq.rank_adaption()
        self.wk.rank_adaption()
        self.wv.rank_adaption()
        self.dense.rank_adaption()

    def get_rank(self):
        return [self.wq.get_rank(), self.wk.get_rank(), self.wv.get_rank()]

    def get_weights_num(self):
        low_wq, full_wq = self.wq.get_weights_num()
        low_wk, full_wk = self.wk.get_weights_num()
        low_wv, full_wv = self.wv.get_weights_num()
        low_dense, full_dense = self.dense.get_weights_num()
        return low_wq + low_wk + low_wv + low_dense, full_wq + full_wk + full_wv + full_dense


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, rate=0.1, tolerance=0.1):
        super().__init__()

        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, tolerance=tolerance)
        self.ffn1 = DLRALayerAdaptive(input_dim=d_model, units=dff, low_rank=d_model // 2, epsAdapt=tolerance)
        self.ffn2 = DLRALayerAdaptiveLinear(
            input_dim=dff,
            units=d_model,
            low_rank=d_model // 2,
            epsAdapt=tolerance,
        )

        # Build low-rank layers
        self.ffn1.build_model()
        self.ffn2.build_model()

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask, step):
        # x = embedded portuguese , training = english, mask= remove padding tokens

        attn_output, _ = self.mha(x, x, x, mask, step)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn1(out1, step)  # (batch_size, input_seq_len, dff)
        ffn_output = self.ffn2(ffn_output, step)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2

    def k_step_preprocessing(self):
        self.mha.k_step_preprocessing()
        self.ffn1.k_step_preprocessing()
        self.ffn2.k_step_preprocessing()

    def k_step_postprocessing_adapt(self):
        self.mha.k_step_postprocessing_adapt()
        self.ffn1.k_step_postprocessing_adapt()
        self.ffn2.k_step_postprocessing_adapt()

    def l_step_preprocessing(self):
        self.mha.l_step_preprocessing()
        self.ffn1.l_step_preprocessing()
        self.ffn2.l_step_preprocessing()

    def l_step_postprocessing_adapt(self):
        self.mha.l_step_postprocessing_adapt()
        self.ffn1.l_step_postprocessing_adapt()
        self.ffn2.l_step_postprocessing_adapt()

    def s_step_preprocessing(self):
        self.mha.s_step_preprocessing()
        self.ffn1.s_step_preprocessing()
        self.ffn2.s_step_preprocessing()

    def rank_adaption(self):
        self.mha.rank_adaption()
        self.ffn1.rank_adaption()
        self.ffn2.rank_adaption()

    def get_rank(self):
        return [self.mha.get_rank(), self.ffn1.get_rank(), self.ffn2.get_rank()]

    def get_weights_num(self):
        low_mha, full_mha = self.mha.get_weights_num()
        low_ffn1, full_ffn1 = self.ffn1.get_weights_num()
        low_ffn2, full_ffn2 = self.ffn2.get_weights_num()
        return low_mha + low_ffn1 + low_ffn2, full_mha + full_ffn1 + full_ffn2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, rate=0.1, tolerance=0.1):
        super().__init__()

        self.mha1 = MultiHeadAttention(d_model=d_model, num_heads=num_heads, tolerance=tolerance)
        self.mha2 = MultiHeadAttention(d_model=d_model, num_heads=num_heads, tolerance=tolerance)

        self.ffn1 = DLRALayerAdaptive(input_dim=d_model, units=dff, low_rank=d_model // 2, epsAdapt=tolerance)
        self.ffn2 = DLRALayerAdaptiveLinear(
            input_dim=dff,
            units=d_model,
            low_rank=d_model // 2,
            epsAdapt=tolerance,
        )
        # Build low-rank layers
        self.ffn1.build_model()
        self.ffn2.build_model()

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask, step):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(
            x,
            x,
            x,
            look_ahead_mask,
            step,
        )  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output,
            enc_output,
            out1,
            padding_mask,
            step,
        )  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn1(out1, step)  # (batch_size, input_seq_len, dff)
        ffn_output = self.ffn2(ffn_output, step)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2

    def k_step_preprocessing(self):
        self.mha1.k_step_preprocessing()
        self.mha2.k_step_preprocessing()
        self.ffn1.k_step_preprocessing()
        self.ffn2.k_step_preprocessing()

    def k_step_postprocessing_adapt(self):
        self.mha1.k_step_postprocessing_adapt()
        self.mha2.k_step_postprocessing_adapt()
        self.ffn1.k_step_postprocessing_adapt()
        self.ffn2.k_step_postprocessing_adapt()

    def l_step_preprocessing(self):
        self.mha1.l_step_preprocessing()
        self.mha2.l_step_preprocessing()
        self.ffn1.l_step_preprocessing()
        self.ffn2.l_step_preprocessing()

    def l_step_postprocessing_adapt(self):
        self.mha1.l_step_postprocessing_adapt()
        self.mha2.l_step_postprocessing_adapt()
        self.ffn1.l_step_postprocessing_adapt()
        self.ffn2.l_step_postprocessing_adapt()

    def s_step_preprocessing(self):
        self.mha1.s_step_preprocessing()
        self.mha2.s_step_preprocessing()
        self.ffn1.s_step_preprocessing()
        self.ffn2.s_step_preprocessing()

    def rank_adaption(self):
        self.mha1.rank_adaption()
        self.mha2.rank_adaption()
        self.ffn1.rank_adaption()
        self.ffn2.rank_adaption()

    def get_rank(self):
        return [self.mha1.get_rank(), self.mha2.get_rank(), self.ffn1.get_rank(), self.ffn2.get_rank()]

    def get_weights_num(self):
        low_mha1, full_mha1 = self.mha1.get_weights_num()
        low_mha2, full_mha2 = self.mha2.get_weights_num()
        low_ffn1, full_ffn1 = self.ffn1.get_weights_num()
        low_ffn2, full_ffn2 = self.ffn2.get_weights_num()
        return low_mha1 + low_mha2 + low_ffn1 + low_ffn2, full_mha1 + full_mha2 + full_ffn1 + full_ffn2


class Encoder(tf.keras.layers.Layer):
    def __init__(
        self,
        *,
        num_layers,
        d_model,
        num_heads,
        dff,
        input_vocab_size,
        rate=0.1,
        tolerance,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(MAX_TOKENS, self.d_model)

        self.enc_layers = [
            EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, rate=rate, tolerance=tolerance)
            for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask, step):
        # x = portuguese , training = english, mask= remove padding tokens
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask, step)

        return x  # (batch_size, input_seq_len, d_model)

    def k_step_preprocessing(self):
        for i in range(self.num_layers):
            self.enc_layers[i].k_step_preprocessing()

    def k_step_postprocessing_adapt(self):
        for i in range(self.num_layers):
            self.enc_layers[i].k_step_postprocessing_adapt()

    def l_step_preprocessing(self):
        for i in range(self.num_layers):
            self.enc_layers[i].l_step_preprocessing()

    def l_step_postprocessing_adapt(self):
        for i in range(self.num_layers):
            self.enc_layers[i].l_step_postprocessing_adapt()

    def s_step_preprocessing(self):
        for i in range(self.num_layers):
            self.enc_layers[i].s_step_preprocessing()

    def rank_adaption(self):
        for i in range(self.num_layers):
            self.enc_layers[i].rank_adaption()

    def get_rank(self):
        ranks = []
        for i in range(self.num_layers):
            ranks.append(self.enc_layers[i].get_rank())
        return ranks

    def get_weights_num(self):
        low = 0
        full = 0
        for i in range(self.num_layers):
            low_i, full_i = self.enc_layers[i].get_weights_num()
            low += low_i
            full += full_i
        return low, full


class Decoder(tf.keras.layers.Layer):
    def __init__(
        self,
        *,
        num_layers,
        d_model,
        num_heads,
        dff,
        target_vocab_size,
        rate=0.1,
        tolerance=0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(MAX_TOKENS, d_model)

        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, rate=rate, tolerance=tolerance)
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask, step):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](
                x,
                enc_output,
                training,
                look_ahead_mask,
                padding_mask,
                step,
            )

            attention_weights[f"decoder_layer{i + 1}_block1"] = block1
            attention_weights[f"decoder_layer{i + 1}_block2"] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights

    def k_step_preprocessing(self):
        for i in range(self.num_layers):
            self.dec_layers[i].k_step_preprocessing()

    def k_step_postprocessing_adapt(self):
        for i in range(self.num_layers):
            self.dec_layers[i].k_step_postprocessing_adapt()

    def l_step_preprocessing(self):
        for i in range(self.num_layers):
            self.dec_layers[i].l_step_preprocessing()

    def l_step_postprocessing_adapt(self):
        for i in range(self.num_layers):
            self.dec_layers[i].l_step_postprocessing_adapt()

    def s_step_preprocessing(self):
        for i in range(self.num_layers):
            self.dec_layers[i].s_step_preprocessing()

    def rank_adaption(self):
        for i in range(self.num_layers):
            self.dec_layers[i].rank_adaption()

    def get_rank(self):
        ranks = []
        for i in range(self.num_layers):
            ranks.append(self.dec_layers[i].get_rank())
        return ranks

    def get_weights_num(self):
        low = 0
        full = 0
        for i in range(self.num_layers):
            low_i, full_i = self.dec_layers[i].get_weights_num()
            low += low_i
            full += full_i
        return low, full


class TransformerDLRA(tf.keras.Model):
    def __init__(
        self,
        *,
        num_layers,
        d_model,
        num_heads,
        dff,
        input_vocab_size,
        target_vocab_size,
        rate=0.1,
        tolerance=0.1,
    ):
        super().__init__()
        self.encoder = Encoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            input_vocab_size=input_vocab_size,
            rate=rate,
            tolerance=tolerance,
        )

        self.decoder = Decoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            target_vocab_size=target_vocab_size,
            rate=rate,
            tolerance=tolerance,
        )

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)  # stays full rank
        self.target_vocab_size = target_vocab_size
        self.d_model = d_model

    def call(self, inputs, training, step):
        # Keras models prefer if you pass all your inputs in the first argument
        inp, tar = inputs

        padding_mask, look_ahead_mask = self.create_masks(inp, tar)

        enc_output = self.encoder(inp, training, padding_mask, step)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar,
            enc_output,
            training,
            look_ahead_mask,
            padding_mask,
            step,
        )

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights

    def create_masks(self, inp, tar):
        # Encoder padding mask (Used in the 2nd attention block in the decoder too.)
        padding_mask = create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return padding_mask, look_ahead_mask

    def k_step_preprocessing(self):
        self.encoder.k_step_preprocessing()
        self.decoder.k_step_preprocessing()

    def k_step_postprocessing_adapt(self):
        self.encoder.k_step_postprocessing_adapt()
        self.decoder.k_step_postprocessing_adapt()

    def l_step_preprocessing(self):
        self.encoder.l_step_preprocessing()
        self.decoder.l_step_preprocessing()

    def l_step_postprocessing_adapt(self):
        self.encoder.l_step_postprocessing_adapt()
        self.decoder.l_step_postprocessing_adapt()

    def s_step_preprocessing(self):
        self.encoder.s_step_preprocessing()
        self.decoder.s_step_preprocessing()

    def rank_adaption(self):
        self.encoder.rank_adaption()
        self.decoder.rank_adaption()

    def get_rank(self):
        return [self.encoder.get_rank(), self.decoder.get_rank()]

    def get_weights_num(self):
        low_encoder, full_encoder = self.encoder.get_weights_num()
        low_decoder, full_decoder = self.decoder.get_weights_num()
        return (
            low_encoder + low_decoder + self.target_vocab_size * self.d_model,
            full_encoder + full_decoder + self.target_vocab_size * self.d_model,
        )

    def get_compression_rate(self):
        low, full = self.get_weights_num()
        return low / full

    @staticmethod
    def set_none_grads_to_zero(grads, weights):
        """
        :param grads: gradients of current tape
        :param weights: weights of current model
        :return: sets the nonexistent gradients to zero (i K step, the grads of S and L step are None, which throws annoying warnings)
        """
        for i in range(len(grads)):
            if grads[i] is None:
                grads[i] = tf.zeros(shape=weights[i].shape, dtype=tf.float32)
        return 0

    # @staticmethod
    # def set_dlra_bias_grads_to_zero(grads):
    #    """
    #    :param grads: gradients of current tape
    #    :return: sets the nonexistent gradients to zero (i K step, the grads of S and L step are None, which throws annoying warnings)
    #    """
    #    for i in range(len(grads)):
    #        if len(grads[i].shape) == 1:
    #            grads[i] = tf.math.scalar_mul(0.0, grads[i])
    #    return 0


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


class Translator(tf.Module):
    def __init__(self, tokenizers, transformer):
        self.tokenizers = tokenizers
        self.transformer = transformer

    def __call__(self, sentence, max_length=MAX_TOKENS):
        # input sentence is portuguese, hence adding the start and end token
        assert isinstance(sentence, tf.Tensor)
        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]

        sentence = self.tokenizers.pt.tokenize(sentence).to_tensor()

        encoder_input = sentence

        # As the output language is english, initialize the output with the
        # english start token.
        start_end = self.tokenizers.en.tokenize([""])[0]
        start = start_end[0][tf.newaxis]
        end = start_end[1][tf.newaxis]

        # `tf.TensorArray` is required here (instead of a python list) so that the
        # dynamic-loop can be traced by `tf.function`.
        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)

        for i in tf.range(max_length):
            output = tf.transpose(output_array.stack())
            predictions, _ = self.transformer([encoder_input, output], training=False)

            # select the last token from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.argmax(predictions, axis=-1)

            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output_array = output_array.write(i + 1, predicted_id[0])

            if predicted_id == end:
                break

        output = tf.transpose(output_array.stack())
        # output.shape (1, tokens)
        text = self.tokenizers.en.detokenize(output)[0]  # shape: ()

        tokens = self.tokenizers.en.lookup(output)[0]

        # `tf.function` prevents us from using the attention_weights that were
        # calculated on the last iteration of the loop. So recalculate them outside
        # the loop.
        _, attention_weights = self.transformer([encoder_input, output[:, :-1]], training=False)

        return text, tokens, attention_weights


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis],
        np.arange(d_model)[np.newaxis, :],
        d_model,
    )

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += mask * -1e9

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights
