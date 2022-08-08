import tensorflow as tf
tf.compat.v1.enable_eager_execution()

OUTPUT_CHANNELS = 1
BATCH_SIZE = 1
IMG_WIDTH = 250
IMG_HEIGHT = 250

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
    
    def __init__(self, filters, size, apply_dropout=False):
        super(Upsample, self).__init__()
        self.apply_dropout = apply_dropout
        initializer = tf.random_normal_initializer(0., 0.02)

        self.up_conv = tf.keras.layers.Conv2DTranspose(filters,
                                                       (size, size),
                                                       strides=2,
                                                       padding='valid',
                                                       kernel_initializer=initializer,
                                                       use_bias=False)
        self.batchnorm = tf.keras.layers.BatchNormalization()
        if self.apply_dropout:
            self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, x1, x2, training):
        x = self.up_conv(x1)
        x = self.batchnorm(x, training=training)
        if self.apply_dropout:
            x = self.dropout(x, training=training)
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
            z = tf.random.normal(shape=[BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 1])
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

class C_GAN():
    
    def __init__(self, generator, name=None):
        self.generator = generator       
        self.checkpoint = tf.train.Checkpoint(generator=self.generator)    
    
    def generate_image(self, input_image, name=None):
        gen_output,_ = self.generator(input_image, training=False)            
        return gen_output
           
    def restore_from_checkpoint(self):
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
        self.checkpoint.restore(checkpoint_file)