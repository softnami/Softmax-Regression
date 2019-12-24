# Softmax Regression
### [Author: Hussain Mir Ali]
This module contains logic to run the Softmax Regression algorithm.

## External Libraries Used:
* math License: https://github.com/josdejong/math/blob/master/LICENSE
* mocha License: https://github.com/mochajs/mocha/blob/master/LICENSE
* sinon Licencse: https://github.com/sinonjs/sinon/blob/master/LICENSE
* yuidocjs License: https://github.com/yui/yuidoc/blob/master/LICENSE
* nodeJS License: https://github.com/nodejs/node/blob/master/LICENSE

## Installation:
*  run 'npm i @softnami/softmaxregression'

### Sample usage:

```javascript
import {SoftmaxRegression} from '@softnami/softmaxregression';
import * as math from 'mathjs';

const callback = (data)=> {
    console.log(data);
};

const generateData = ()=>{
    let sample_y = []; //one-hot encoded classes: [1,2,3] -> [[1, 0,0],[0, 1,0],[0, 0,1]]
    let y = [], num = 0, 
    x = (math.random(math.matrix([1000, 78]), -1, 1));

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

    for (let i = 0; i < 1000; i++) {
      num = Math.floor((Math.random() * 100) + 0);
      y.push(sample_y[num]);
    }

    return [x, y];
};

const softmaxRegression = new SoftmaxRegression({
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

let [x, y] = generateData();

console.log('Start training.');
softmaxRegression.startRegression(x, math.matrix(y));

```

## Testing:
* For unit testing Mocha and Sinon have been used. 
* Run 'npm test', if timeout occurs then increase timeout in test script.


## Documentation
*  The documentation is available in the 'out' folder of this project. Open the 'index.html' file under the 'out' folder with Crhome or Firefox.
*  To generate the documentation run 'yuidoc .' command in the main directory of this project.


