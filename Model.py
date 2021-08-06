
import tensorflow as tf
import os
from tensorflow.keras.models import Model
#import h5py



class CNNAE:
    
  def __init__(self):
        # Input layer

      input_img = tf.keras.layers.Input(shape=(252, 252, 3))

      # Encoder network
      # Convert images into a compressed, encoded representation
      x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img) #64
      x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
      x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x) #128
      x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
      x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x) #256
      encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
      #decoder network
      x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
      x = tf.keras.layers.UpSampling2D((2, 2))(x)
      x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
      x = tf.keras.layers.UpSampling2D((2, 2))(x)
      x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')(x)
      x = tf.keras.layers.UpSampling2D((2, 2))(x)
      decoded = tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same',name="Dec")(x)
      #Classifier
      #xAvg= tf.keras.layers.GlobalAveragePooling2D()(encoded)
      xAvg=tf.keras.layers.Flatten()(encoded)
      xFC = tf.keras.layers.Dense(4, activation='softmax',name="FC")(xAvg)
      self.model= Model(inputs= input_img, outputs=[decoded,xFC])
    
      
  def train_AE(self,trainGen,valGen,Nepoch=15):
    
    self.train_AE_weights()
    self.fix_FC_weights()
    self.model.summary()
    loss_fn = tf.keras.losses.MeanAbsoluteError()
    callBack= tf.keras.callbacks.EarlyStopping(
                                      monitor="val_Dec_mean_absolute_error",
                                      min_delta=0,
                                      patience=3,
                                      verbose=0,
                                      mode="min",
                                      baseline=None,
                                      restore_best_weights=True,
                                  )
    rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', patience= 3, factor= 0.5, min_lr= 1e-6, verbose=1,min_delta=0.05)
    self.model.compile(optimizer='adam', loss={"Dec":'binary_crossentropy'}, metrics={"Dec":loss_fn})
    #self.model.fit_generator(trainGen,validation_data =valGen,epochs=15) 
    self.AE_history = self.model.fit(trainGen,validation_data =valGen,epochs=Nepoch, callbacks=[callBack, rlrop])
      

  def fix_AE_weights(self):

      for i,_ in enumerate(self.model.layers[:-1]):
        self.model.layers[i].trainable= False

  def fix_FC_weights(self):


    self.model.layers[-1].trainable= False

  def train_AE_weights(self):

      for i,_ in enumerate(self.model.layers[:-2]):
        self.model.layers[i].trainable= True

  def train_FC_weights(self):

    self.model.layers[-1].trainable= True 


  def train_Classifier(self,trainGen,valGen,Nepoch=30):
      self.fix_AE_weights()
      self.train_FC_weights()
      callBack= tf.keras.callbacks.EarlyStopping(
                                  monitor="val_FC_recall",
                                  min_delta=0,
                                  patience=10,
                                  verbose=0,
                                  mode="max",
                                  baseline=None,
                                  restore_best_weights=True,
                              )
      rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', patience= 3, factor= 0.5, min_lr= 1e-6, verbose=1,min_delta=0.05)
      self.model.summary()
      self.model.compile(optimizer='adam', loss={"FC" :'categorical_crossentropy'}, metrics={"FC" :[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]})
      self.FC_history = self.model.fit(trainGen,validation_data =valGen,epochs=Nepoch,callbacks=[callBack, rlrop])
      
  def save_model(self,path,modelname="Model"):
    self.model.save(os.path.join(path,modelname)+".h5")

  def save_weights(self,path,modelname="Model_weights"):

    self.model.save_weights(os.path.join(path,modelname)+".h5")

  def load_weights(self,path,checkpoint="Model_weights"):
    self.model.load_weights(os.path.join(path,checkpoint)+".h5")


class EFFNET:
    
  def __init__(self):
        # Input layer

      input_img = tf.keras.layers.Input(shape=(252, 252, 3))

      Backbone = tf.keras.applications.EfficientNetB0(include_top=False, weights= 'imagenet', input_tensor=input_img,drop_connect_rate =0.2) #, drop_connect_rate =0.4, 'imagenet'
      Backbone.trainable = False

      #x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(Backbone.output)
      x = tf.keras.layers.MaxPooling2D((2, 2), padding='valid')(Backbone.output)
      x = tf.keras.layers.Flatten(name="avg_pool")(x)
      x = tf.keras.layers.BatchNormalization(name='BN_CL')(x)
      #x = tf.keras.layers.Dense(640, activation = 'relu')(x) #Added Dense Layer
      #x = tf.keras.layers.Dense(300, activation = 'relu')(x) #Added Dense Layer
      x = tf.keras.layers.Dense(10, activation = 'relu')(x) #Added Dense Layer
      x = tf.keras.layers.Dropout(0.2)(x)
      xFC = tf.keras.layers.Dense(4, activation='softmax',name="FC")(x)
      self.model= Model(inputs= input_img, outputs=[xFC])
    
  def fix_Backbone_weights(self):

      for i,layer in enumerate(self.model.layer):
        if layer.name == 'BN_CL' or layer.name == "FC" or layer.name == "avg_pool":
          self.model.layers[i].trainable= True
        else:
          self.model.layers[i].trainable= False

  def unfix_layers_Backbone_weights(self, Nlayer=3):

      for i,_ in enumerate(self.model.layers[-5-Nlayer:]):
        self.model.layers[i].trainable= True

  def train_Classifier_only(self,trainGen,valGen,Nepoch=30):
      #self.fix_Backbone_weights()
      callBack= tf.keras.callbacks.EarlyStopping(
                                  monitor="val_recall",
                                  min_delta=0,
                                  patience=10,
                                  verbose=0,
                                  mode="max",
                                  baseline=None,
                                  restore_best_weights=True,
                              )
      rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', patience= 3, factor= 0.5, min_lr= 1e-6, verbose=1,min_delta=0.05)
      self.model.summary()
      self.model.compile(optimizer='adam', loss={"FC" :'categorical_crossentropy'}, metrics={"FC" :[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]})
      self.FC_history = self.model.fit(trainGen,validation_data =valGen,epochs=Nepoch,callbacks=[callBack,rlrop])

  def train_Classifier_withBackbone(self,trainGen,valGen,Nepoch=30,Nlayers=3):
      self.unfix_layers_Backbone_weights(Nlayer= Nlayers)

      callBack= tf.keras.callbacks.EarlyStopping(
                                  monitor="val_recall_1",
                                  min_delta=0,
                                  patience=10,
                                  verbose=0,
                                  mode="max",
                                  baseline=None,
                                  restore_best_weights=True,
                              )
      rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', patience= 3, factor= 0.5, min_lr= 1e-6, verbose=1,min_delta=0.05)
      self.model.summary()
      self.model.compile(optimizer='adam', loss={"FC" :'categorical_crossentropy'}, metrics={"FC" :[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]})
      self.FC_B_history = self.model.fit(trainGen,validation_data =valGen,epochs=Nepoch,callbacks=[callBack,rlrop])
      
  def save_model(self,path,modelname="Model"):
    self.model.save(os.path.join(path,modelname)+".h5")

  def save_weights(self,path,modelname="Model_weights"):

    self.model.save_weights(os.path.join(path,modelname)+".h5")

  def load_weights(self,path,checkpoint="Model_weights"):
    self.model.load_weights(os.path.join(path,checkpoint)+".h5")

