
## NN Sine wave with arduino

In the following repository we will implement a neural network with keras in order to reproduce a sinusoidal wave, once the model is trained, we will export the weights of our network in order to implement it in an arduino code.First we will create our training data, where the function **sin(x)** will be applied to a sample of 20 data from 0 to 2PI.

![image](https://user-images.githubusercontent.com/115313115/206610558-5534c828-b1c3-4bc9-8ec6-30d2592ab9b2.png)

We build our model using the keras library

```python
#We create our model
model=keras.models.Sequential()
model.add(keras.layers.Dense(10,input_shape=(1,),activation='tanh'))
model.add(keras.layers.Dense(1,activation='linear'))

#Set optimizers and model metrics
model.compile(keras.optimizers.Adam(),'mse',metrics=['mse'])

#Training
hist=model.fit(x_train,yd_train,epochs=10000,verbose=0)

```
## model implemented
![image](https://user-images.githubusercontent.com/115313115/206611358-9d354478-ac9b-410b-8689-04d544085ea4.png)

## Result

![image](https://user-images.githubusercontent.com/115313115/206611450-a26a3ec2-06dd-4f9e-a190-611239546c99.png)

## Export weights to arduino
With the following commands in colab, we can extract the weights of our neural network, as well as those of the hidden layer and those of the output layer.

```python
#Weights hidden layer
Whl = model.layers[0].get_weights()[0]

#Bias hidden layer
bhl = model.layers[0].get_weights()[1]

#Weights output layer
Wol = model.layers[1].get_weights()[0]

#Bias output layer
bol = model.layers[1].get_weights()[1]

```
Once our weights are obtained, we define them in our arduino file as follows:
```arduino
const int HiddenNodes = 10;
const int InputNodes = 1;
const int OutputNodes = 1;

// The neural network weights were obtained in TensorFlow and copied to this program.
// Hidden layer weights

const float HiddenWeights[InputNodes+1][HiddenNodes]= {
{-0.62555987, -0.60016775,  0.6546977,   0.33472893, -0.48347482, -0.33484796,   -0.64942,    -0.22847535, -0.1807958,   0.5953621},
{1.1536413,   1.9326534,  -1.3416654,  -0.32951242,  0.37019846,  1.9247391,   1.3141906,   1.0192639,   0.4849291,  -2.393269}
    };
    

// Output layer weights

const float OutputWeights[HiddenNodes+1][OutputNodes]  = {
 { 0.40681854},
 { 1.8112999 },
 {-0.6973271 },
 { 1.6000099 },
 {-1.0351793 },
 {-2.2122264 },
 { 0.65181553},
 {-2.0956979 },
 {-1.7131217 },
 {-1.8370938 },
 {0.33191314}
    }; 
```
After we have defined our weights, we implement the following two loops to update our weights
```arduino
/******************************************************************
* Hidden layer output calculation
******************************************************************/
    for( i = 0 ; i < HiddenNodes ; i++ ) {    
      Accum = HiddenWeights[InputNodes][i] ;
      for( j = 0 ; j < InputNodes ; j++ ) {
        Accum += HiddenWeights[j][i]*Input[j];
      }
      // tangent hyperbolic activation function
        Hidden[i] = (exp(Accum)-exp(-Accum))/(exp(Accum)+exp(-Accum));
    }

/******************************************************************
* Neural network output calculation
******************************************************************/

    for( i = 0 ; i < OutputNodes ; i++ ) {    
      Accum = OutputWeights[HiddenNodes][i] ;
      for( j = 0 ; j < HiddenNodes ; j++ ) {
        Accum +=  OutputWeights[j][i]*Hidden[j];
      }
      // linear activation function
        Output[i] = Accum; 
    }
```
Finally we implement our network in the arduino device at our disposal, in my case an arduino UNO and we see how our sinusoidal signal is generated in the serial monitor and serial plotter.
![image](https://user-images.githubusercontent.com/115313115/206614348-5377e389-2489-428c-bc5e-e4f422228d23.png)
