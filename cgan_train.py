import tensorflow as tf
import os
import numpy as np

BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 250
IMG_HEIGHT = 250
IMG_DEPTH = 1

tf.compat.v1.enable_eager_execution()

def load_image(image_file, is_train):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image,channels=1,dct_method='INTEGER_ACCURATE')
    
    
    w = tf.shape(image)[1]
    
    w = w // 2
    real_image = image[:, :w, :]
    input_image = image[:, w:, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)
  
#     # normalizing the images to [-1, 1]
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image


def load_dataset():

    train_dataset = tf.data.Dataset.list_files('./cgan_data/train/*.jpg')
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.map(lambda x: load_image(x, True))
    train_dataset = train_dataset.batch(BATCH_SIZE)

    return train_dataset

#%%
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

import time
import pickle


BATCH_SIZE = 1
IMG_WIDTH = 250
IMG_HEIGHT = 250
LAMBDA = 100
OUTPUT_CHANNELS = 1
checkpoint_dir = './cgan-model'

class Downsample(tf.keras.Model):
    
    def __init__(self, filters, size, apply_batchnorm=True):
        super(Downsample, self).__init__()
        self.apply_batchnorm = apply_batchnorm
        initializer = tf.random_normal_initializer(0., 0.02)

        self.conv1 = tf.keras.layers.Conv2D(filters,
                                            (size, size),
                                            strides=2,
                                            padding='valid',
                                            kernel_initializer=initializer,
                                            use_bias=False)
        if self.apply_batchnorm:
            self.batchnorm = tf.keras.layers.BatchNormalization()
  
    def call(self, x, training):
        x = self.conv1(x)
        if self.apply_batchnorm:
            x = self.batchnorm(x, training=training)
        x = tf.nn.leaky_relu(x)
        return x


class Upsample(tf.keras.Model):
    
    def __init__(self, filters, size):
        super(Upsample, self).__init__()

        initializer = tf.random_normal_initializer(0., 0.02)

        self.up_conv = tf.keras.layers.Conv2DTranspose(filters,
                                                       (size, size),
                                                       strides=2,
                                                       padding='valid',
                                                       kernel_initializer=initializer,
                                                       use_bias=False)
        self.batchnorm = tf.keras.layers.BatchNormalization()

    def call(self, x1, x2, training):
        x = self.up_conv(x1)
        x = self.batchnorm(x, training=training)
        x = tf.nn.relu(x)
        if x2 is not None:
            x = tf.concat([x, x2], axis=-1)
        return x


class ResidualBlock(tf.keras.Model):
    
    def __init__(self, filters, size):
        super(ResidualBlock, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters,
                                            (size, size),
                                            padding='same',
                                            kernel_initializer='he_normal',
                                            use_bias=False)
        self.batchnorm1 = tf.keras.layers.BatchNormalization()
        
        self.conv2 = tf.keras.layers.Conv2D(filters,
                                            (size, size),
                                            padding='same',
                                            kernel_initializer='he_normal',
                                            use_bias=False)
        self.batchnorm2 = tf.keras.layers.BatchNormalization()
  
    def call(self, x, training):
        y = self.conv1(x)
        y = self.batchnorm1(y, training=training)
        y = tf.nn.relu(y)
        y = self.conv2(y)
        y = self.batchnorm2(y, training=training)
        y = tf.math.add(x, y)
        return y

class ResNet9Generator(tf.keras.Model):
    
    def __init__(self, noise=False):
        super(ResNet9Generator, self).__init__()
        self.noise_inputs = noise
        initializer = tf.random_normal_initializer(0., 0.02)
        
        self.init = tf.keras.layers.Conv2D(64,
                                           (7, 7),
                                           padding='same',
                                           kernel_initializer=initializer,
                                           use_bias=False)
        self.batchnorm = tf.keras.layers.BatchNormalization()
        
        self.down1 = Downsample(128, 4)
        self.down2 = Downsample(256, 4)
        
        self.res1 = ResidualBlock(256, 3)
        self.res2 = ResidualBlock(256, 3)
        self.res3 = ResidualBlock(256, 3)
        self.res4 = ResidualBlock(256, 3)
        self.res5 = ResidualBlock(256, 3)
        self.res6 = ResidualBlock(256, 3)
        self.res7 = ResidualBlock(256, 3)
        self.res8 = ResidualBlock(256, 3)
        self.res9 = ResidualBlock(256, 3)

        self.up1 = Upsample(128, 4)
        self.up2 = Upsample(64, 4)

        self.last = tf.keras.layers.Conv2D(OUTPUT_CHANNELS,
                                           (7, 7),
                                           padding='same',
                                           kernel_initializer=initializer)
  
    # @tf.contrib.eager.defun
    def call(self, x, training):
        # x shape == (bs, 256, 256, 3)
        if self.noise_inputs:
            z = tf.random.normal(shape=[BATCH_SIZE * 2, IMG_HEIGHT, IMG_WIDTH, 1])
            x = tf.concat([x, z], axis=-1) # (bs, 256, 256, 4)
        
        x1 = self.init(x) # (bs, 256, 256, 64)
        x1 = self.batchnorm(x1, training=training)
        x1 = tf.nn.relu(x1)
        
        x2 = self.down1(x1, training=training) # (bs, 128, 128, 128)
        x3 = self.down2(x2, training=training) # (bs, 64, 64, 256)
        
        x4 = self.res1(x3, training=training)
        x4 = self.res2(x4, training=training)
        x4 = self.res3(x4, training=training)
        x4 = self.res4(x4, training=training)
        x4 = self.res5(x4, training=training)
        x4 = self.res6(x4, training=training)
        x4 = self.res7(x4, training=training)
        x4 = self.res8(x4, training=training)
        x4 = self.res9(x4, training=training)
        
        x5 = self.up1(x4, None, training=training) # (bs, 128, 128, 128)
        x6 = self.up2(x5, None, training=training) # (bs, 256, 256, 64)

        x7 = self.last(x6) # (bs, 256, 256, 3)
        x7 = tf.nn.tanh(x7)

        return x7, z


class PatchDiscriminator(tf.keras.Model):
    
    def __init__(self):
        super(PatchDiscriminator, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)
    
        self.down1 = Downsample(64, 4, False)
        self.down2 = Downsample(128, 4)
        self.down3 = Downsample(256, 4)
    
        # we are zero padding here with 1 because we need our shape to 
        # go from (batch_size, 32, 32, 256) to (batch_size, 31, 31, 512)
        self.zero_pad1 = tf.keras.layers.ZeroPadding2D()
        self.conv = tf.keras.layers.Conv2D(512, 
                                           (4, 4), 
                                           strides=1, 
                                           kernel_initializer=initializer, 
                                           use_bias=False)
        self.batchnorm1 = tf.keras.layers.BatchNormalization()
    
        # shape change from (batch_size, 31, 31, 512) to (batch_size, 30, 30, 1)
        self.zero_pad2 = tf.keras.layers.ZeroPadding2D()
        self.last = tf.keras.layers.Conv2D(1, 
                                           (4, 4), 
                                           strides=1,
                                           kernel_initializer=initializer)
  
    # @tf.contrib.eager.defun
    def call(self, inp, tar, training):
        # concatenating the input and the target
        x = tf.concat([inp, tar], axis=-1) # (bs, 256, 256, channels*2)
        x = self.down1(x, training=training) # (bs, 128, 128, 64)
        x = self.down2(x, training=training) # (bs, 64, 64, 128)
        x = self.down3(x, training=training) # (bs, 32, 32, 256)

        x = self.zero_pad1(x) # (bs, 34, 34, 256)
        x = self.conv(x)      # (bs, 31, 31, 512)
        x = self.batchnorm1(x, training=training)
        x = tf.nn.leaky_relu(x)

        x = self.zero_pad2(x) # (bs, 33, 33, 512)
        # don't add a sigmoid activation here since
        # the loss function expects raw logits.
        x = self.last(x)      # (bs, 30, 30, 1)
        
#         print(x.shape)

        return x


class C_GAN():
    
    def __init__(self, generator, discriminator, generator_learning_rate=2e-4, discriminator_learning_rate=2e-4,LAMBDA=100,BETA=50,ckpt_freq=1, name=None,checkpoint_dir=None):
        self.name = ('' if name is None else name + '_') + generator.__class__.__name__ + '_' + discriminator.__class__.__name__
        self.generator = generator
        self.discriminator = discriminator
        self.ckpt_freq = ckpt_freq
        self.LAMBDA = LAMBDA
        self.BETA = BETA
        self.checkpoint_dir = checkpoint_dir
        
        self.generator_optimizer = tf.compat.v1.train.AdamOptimizer(generator_learning_rate, beta1=0.5)
        self.discriminator_optimizer = tf.compat.v1.train.AdamOptimizer(discriminator_learning_rate, beta1=0.5)
        
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, self.name + '_ckpt')
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)
        
    
    def generator_loss(self, disc_generated_output, gen_output, target, latent_z):
        gan_loss = tf.compat.v1.losses.sigmoid_cross_entropy(multi_class_labels = tf.ones_like(disc_generated_output),
                                                   logits = disc_generated_output)
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        
        ld_loss = tf.reduce_mean(tf.abs(gen_output[:BATCH_SIZE,:,:,:] - gen_output[BATCH_SIZE:,:,:,:]))/tf.reduce_mean(tf.abs(latent_z[:BATCH_SIZE,:,:,:] - latent_z[BATCH_SIZE:,:,:,:]))

        total_gen_loss = gan_loss + (self.LAMBDA * l1_loss) - self.BETA * ld_loss

        return total_gen_loss
    

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = tf.compat.v1.losses.sigmoid_cross_entropy(multi_class_labels = tf.ones_like(disc_real_output), 
                                                    logits = disc_real_output)
        generated_loss = tf.compat.v1.losses.sigmoid_cross_entropy(multi_class_labels = tf.zeros_like(disc_generated_output), 
                                                         logits = disc_generated_output)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss
    
    
    def generate_image(self, input_image, target, name=None):
        input_image = tf.concat([input_image, input_image], axis=0)
        gen_output,_ = self.generator(input_image, training=True)
            
        return gen_output[0] #* 0.5 + 0.5  
        
    
    def restore_from_checkpoint(self):
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
        self.checkpoint.restore(checkpoint_file)
    
    
    def train(self, dataset, epochs):
        training_stats = []
        
        for epoch in range(epochs):
            avg_gen_loss = 0
            avg_disc_loss = 0
            it = 0
            start = time.time()
            
            for target, input_image in dataset:
                target = np.array(target)
                target = np.where(target < 0, -1, 1)
                target = tf.convert_to_tensor(target, dtype=tf.float32)

                input_image = np.array(input_image)
                input_image = np.where(input_image < 0, -1, 1)
                input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)
                
                target = tf.concat([target, target], axis=0)
                input_image = tf.concat([input_image, input_image], axis=0)
                it += 1
                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    gen_output,latent_z = self.generator(input_image, training=True)                
                    
                    disc_real_output = self.discriminator(input_image, target, training=True)
                    disc_generated_output = self.discriminator(input_image, gen_output, training=True)

                    
                    gen_loss = self.generator_loss(disc_generated_output, gen_output, target, latent_z)
                    disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)
                    avg_gen_loss += gen_loss.numpy()
                    avg_disc_loss += disc_loss.numpy()
                
                generator_gradients = gen_tape.gradient(gen_loss, 
                                                        self.generator.variables)
                discriminator_gradients = disc_tape.gradient(disc_loss, 
                                                             self.discriminator.variables)
                
                self.generator_optimizer.apply_gradients(zip(generator_gradients, 
                                                             self.generator.variables))
                self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                                 self.discriminator.variables))
            
            time_taken = time.time() - start
            avg_gen_loss /= it
            avg_disc_loss /= it
            training_stats.append((epoch + 1, time_taken, avg_gen_loss, avg_disc_loss))
            
            if (epoch + 1) % self.ckpt_freq == 0:
                self.checkpoint.save(file_prefix = self.checkpoint_prefix)
                with open(self.name + 'training_stats.pickle', 'wb') as f:
                    pickle.dump(training_stats, f)
                
            
            print('Time taken for epoch {} is {} sec. Gen_Loss: {} Disc_Loss: {} '.format(epoch + 1, time_taken, avg_gen_loss, avg_disc_loss))
            
#%%
import tensorflow as tf

print(tf.test.is_gpu_available)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_dataset = load_dataset()

EPOCHS = 200

gan = C_GAN(generator=ResNet9Generator(noise=True), discriminator=PatchDiscriminator(),   
            generator_learning_rate=2e-4, discriminator_learning_rate=2e-4,LAMBDA=100,BETA=50,ckpt_freq=200,
           checkpoint_dir='./cgan-model/')  #LAMBDA=100,BETA=50

gan.train(train_dataset, EPOCHS)










            