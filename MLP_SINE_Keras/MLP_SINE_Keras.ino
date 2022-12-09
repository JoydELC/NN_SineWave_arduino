/******************************************************************
 * Program that generates a sinewave using a neural network
 * Surface MLP
 * Author:
 * Joyd lasprilla
 * joyd.lasprilla@gmail.com
 ******************************************************************/

#include <math.h>

/******************************************************************
* Definition of network structure
 ******************************************************************/

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

int i, j, p, q, r;
float Accum;
float Hidden[HiddenNodes];
float Output[OutputNodes];
float Input[InputNodes];

void setup(){
  //start serial connection
  Serial.begin(9600);
}

void loop(){
  
  float Entry;
  float Out;
  float Time;

Time=millis();

//trick to generate an input between 0 and 2*pi

Entry=(fmod(Time,6283))/1000; 

Input[0]=Entry;
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

Out=Output[0];

Serial.println(Out);      

delay(50);  

}
