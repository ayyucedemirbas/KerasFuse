import tensorflow as tf
from tensorflow.keras import layers, Model

class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, filters, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.multi_head_self_attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = self.create_ffn(ff_dim, d_model)
        
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.conv1x1_1 = layers.Conv3D(filters, kernel_size=1, padding='same')
        self.conv1x1_2 = layers.Conv3D(filters, kernel_size=1, padding='same')
    
    def create_ffn(self, ff_dim, d_model):
        return tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(d_model)
        ])
    
    def call(self, inputs):
        attn_output = self.multi_head_self_attention(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        attn_output = self.conv1x1_1(attn_output)
        out1 = self.layer_norm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        ffn_output = self.conv1x1_2(ffn_output)
        
        return self.layer_norm2(out1 + ffn_output)

def UNETR(input_shape, num_classes, d_model=256, num_heads=8, ff_dim=2048, num_transformer_blocks=12, filters=64):
    inputs = tf.keras.Input(input_shape)
    
    # Encoding Path (U-Net)
    encoder = inputs
    skips = []
    
    for _ in range(4):
        encoder = layers.Conv3D(filters, kernel_size=3, activation='relu', padding='same')(encoder)
        skips.append(encoder)
        encoder = layers.MaxPooling3D(pool_size=2, strides=2)(encoder)
    
    encoder = layers.Conv3D(filters, kernel_size=3, activation='relu', padding='same')(encoder)
    
    # Transform the encoder features using transformers
    for _ in range(num_transformer_blocks):
        encoder = TransformerBlock(d_model, num_heads, ff_dim, filters)(encoder)
    
    # Decoding Path (U-Net)
    decoder = encoder
    for skip in reversed(skips):
        decoder = layers.Conv3DTranspose(filters, kernel_size=2, strides=2)(decoder)
        decoder = layers.Concatenate()([decoder, skip])
        decoder = layers.Conv3D(filters, kernel_size=3, activation='relu', padding='same')(decoder)
    
    outputs = layers.Conv3D(num_classes, kernel_size=1, activation='softmax')(decoder)
    
    model = Model(inputs, outputs)
    return model
