import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Embedding, Conv1D
import numpy as np
import pandas as pd
import plotly.express as px
import yfinance as yf

from enum import Enum

 
class HyperParams(Enum):
    OPTIMIZER = tf.keras.optimizers.Adam()
    TRAINING_BATCH = 128
    NUM_LAYERS = 5
    GAMMA = 10

class BaseModel(Model):
    def __init__(self, input_dim: tf.int32, output_dim: tf.int32, embedder: tf.bool, output_activation: tf.string, num_layers: tf.int32):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        if embedder:
            self.embedder = Embedding(input_dim=input_dim, output_dim=output_dim)
        self.conv_layers = [Conv1D(32, 8, padding="causal", activation="elu") for _ in range(num_layers)]
        self.final = Dense(output_dim, activation=output_activation)
    
    def call(self, inputs):
        x = self.embedder(inputs) if self.embedder != None else None
        for conv in self.conv_layers:
            x = conv(inputs) if x is None else conv(x)
        return self.final(x)

def random_vectors(batch_size: tf.int32, z_dim: tf.int32):
    return tf.random.uniform(size=(z_dim, batch_size, 1))


class TimeGAN(Model):
    '''
    Inspired by the TimeGAN paper "insert link here"
    Keras model with custom training and eval functions for easy use like any other Keras model
    '''
    def __init__(self, len_seq, latent_dim):
        super().__init__()
        self.len_seq = len_seq
        self.latent_dim = latent_dim
        self.mse = tf.keras.losses.MeanSquaredError()
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy()
        self.embedder = BaseModel(len_seq, latent_dim, True, "sigmoid", HyperParams.NUM_LAYERS.value)
        self.recovery = BaseModel(latent_dim, len_seq, False, "linear", HyperParams.NUM_LAYERS.value)
        self.generator = BaseModel(len_seq, latent_dim, True, "linear", HyperParams.NUM_LAYERS.value)
        self.discriminator = BaseModel(latent_dim, 1, False, "sigmoid", HyperParams.NUM_LAYERS.value)
        self.supervisor = BaseModel(latent_dim, latent_dim, False, "linear", HyperParams.NUM_LAYERS.value - 1)
    
    def call(self, inputs):
        x = tf.random.uniform(shape=inputs.shape)
        x = self.generator(x)
        return self.recovery(x)
    
    def generator_loss(self, data, supervised_loss):
        with tf.GradientTape() as tape:
            sampled_data = random_vectors(data.shape[0], data.shape[1])
            fake_data = self.generator(sampled_data)
            supervised_fake = self.discriminator(self.supervisor(fake_data))
            gen_loss_unsupervised_fake = self.cross_entropy(y_true=tf.zeros_like(supervised_fake), y_pred=supervised_fake)

            supervised_direct = self.discriminator(fake_data)
            gen_loss_unsupervised = self.cross_entropy(y_true=tf.zeros_like(supervised_direct), y_pred=supervised_direct)

            supervised_real = self.discriminator(data)
            discrim_loss_real = self.cross_entropy(y_true=tf.ones_like(data), y_pred=supervised_real)
            discrim_loss = gen_loss_unsupervised + gen_loss_unsupervised_fake + discrim_loss_real

            fake_moment = tf.nn.moments(fake_data, axes=[0])
            real_moment = tf.nn.moments(data, axes=[0])
            moment_loss_mean = tf.reduce_mean(tf.abs(fake_moment[0] - real_moment[1]))
            moment_loss_variance = tf.reduce_mean(fake_moment[1] - real_moment[1])
            gen_loss = gen_loss_unsupervised + gen_loss_unsupervised_fake + (100 * supervised_loss) + (moment_loss_mean + moment_loss_variance)
        
        trainables = self.generator.trainable_variables + self.supervisor.trainable_variables
        grads = tape.gradient(gen_loss, trainables)
        HyperParams.OPTIMIZER.value.apply_gradients(zip(grads, trainables))
        discrim_grads = tape.gradient(discrim_loss, self.discriminator.trainable_variables)
        HyperParams.OPTIMIZER.value.apply_gradients(zip(discrim_grads, self.discriminator.trainable_variables))
        return (gen_loss, discrim_loss)
            
    def supervised_loss(self, data):
        with tf.GradientTape() as tape:
            X_latent = self.embedder(data)
            supervised_gen = self.supervisor(X_latent)
            supervised_loss = self.mse(X_latent[:, 1:, :], supervised_gen[:, :-1, :])
        trainables = self.supervisor.trainable_variables
        grads = tape.gradient(supervised_loss, trainables)
        HyperParams.OPTIMIZER.value.apply_gradients(zip(grads, trainables))
        return supervised_loss


    def reconstruction_loss(self, data):
        with tf.GradientTape() as tape:
            X_recovered = self.embedder(data)
            X_recovered = self.recovery(X_recovered)
            embed_loss = self.mse(data, X_recovered)
            embed_loss_0 = 10 * tf.sqrt(embed_loss)
        trainables = self.embedder.trainable_variables + self.recovery.trainable_variables
        grads = tape.gradient(embed_loss_0, trainables)
        HyperParams.OPTIMIZER.value.apply_gradients(zip(grads, trainables))
        return tf.sqrt(embed_loss_0)

    def train_step(self, data):
        rec_loss = self.reconstruction_loss(data)
        sup_loss = self.supervised_loss(data)
        (gen_loss, discrim_loss) = self.generator_loss(data, sup_loss)
        return {
            "D_loss": discrim_loss,
            "G_loss": gen_loss,
            "S_loss": sup_loss,
            "R_loss": rec_loss
        }