# Softmax Regression
###[Author: Hussain Mir Ali]
This module contains logic to run the Softmax Regression algorithm.

##External Libraries Used:
* math License: https://github.com/josdejong/math/blob/master/LICENSE
* mocha License: https://github.com/mochajs/mocha/blob/master/LICENSE
* sinon Licencse: https://github.com/sinonjs/sinon/blob/master/LICENSE
* yuidocjs License: https://github.com/yui/yuidoc/blob/master/LICENSE
* nodeJS License: https://github.com/nodejs/node/blob/master/LICENSE

##Installation:
*  Download the project and unzip it.
*  Copy the 'softmaxregression' folder to your node_modules folder in your project directory.

###Sample usage:

```javascript
var epoch = 1
var callback = function(data) {
    console.log(data);
};

var sft = new window.SoftmaxRegression({
    'notify_count': 1,
    'momentum': 0.5,
    'parameter_size': [78, 12], //[number of  input features, total number of  output classes]
    'max_epochs': 100,
    'weight_initialization_range': [-0.5, 0.5],
    'threshold': 0.1,
    'batch_size': 256,
    'iteration_callback': callback,
    'learningRate': 0.009,
    'regularization_parameter': Math.exp(-4)
  });

var sample_y = []; //one-hot encoded classes: [1,2,3] -> [[1, 0,0],[0, 1,0],[0, 0,1]]

for(let i =0 ; i< 100; i++){
  sample_y[i] = [];
  for(let j =0; j< 12; j++){
    if(i==j){
      sample_y[i][j]= 1;
    }
    else{
      sample_y[i][j]= 0;
    }
  }  
}


var y = [],
  num = 0;


for (let i = 0; i < 1000; i++) {
  num = Math.floor((Math.random() * 100) + 0);
  y.push(sample_y[num]);
}

var x = (math.random(math.matrix([1000, 78]), -1, 1));

console.log('Start training.');
sft.startRegression(x, math.matrix(y));
```
<!--index.html-->
<!doctype html>
<html>
  <head>
  </head>
  <body >
        <script src="softmaxression/lib/q.js"></script>
        <script src="softmaxregression/lib/math.js"></script>
        <script src="softmaxregression/softmax.js"></script>
         <!--Include the main.js file where you use the algorithm.-->
+        <script src="main.js"></script>
</body>
</html>

*/
```

##Testing:
* For unit testing Mocha and Sinon have been used. 
* On newer computers run the command 'mocha --timeout 50000', the 50000 ms timeout is to give enough time for tests to complete as they might not process before timeout. 
* On older computers run the command 'mocha --timeout 300000', the 300000 ms timeout is to give enough time for tests to complete as they might not process before timeout on older computers. 
* If need be more than 300000 ms should be used to run the tests depending on the processing power of the computer. 


##Documentation
*  The documentation is available in the 'out' folder of this project. Open the 'index.html' file under the 'out' folder with Crhome or Firefox.
*  To generate the documentation run 'yuidoc .' command in the main directory of this project.


