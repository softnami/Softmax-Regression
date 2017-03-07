

var mathjs = require('mathjs');
var epoch = 1;
var callback = function(data) {
    console.timeEnd('timer');
    console.log("Cost: "+data.cost,"Epoch :"+epoch);
    epoch++;
    console.time('timer');
};

var sft = new SoftmaxRegression({
  'notify_count': 1,
  'momentum': 0.5,
  'parameter_size': [784, 100], //[number of  input features, total number of  output classes]
  'max_iterations': 1000000000,
  'weight_initialization_range': [-0.05, 0.05],
  'threshold': 0.1,
  'batch_size': 256,
  'iteration_callback': callback,
  'learningRate': 0.009,
  'regularization_parameter': Math.exp(-4)
});

var sample_y = []; //one-hot encoded classes: [1,2,3] -> [[1, 0,0],[0, 1,0],[0, 0,1]]

for(let i =0 ; i< 100; i++){
  sample_y[i] = [];
  for(let j =0; j< 100; j++){
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


for (let i = 0; i < 10000; i++) {
  num = Math.floor((Math.random() * 100) + 0);
  y.push(sample_y[num]);
}

var x = (mathjs.random(mathjs.matrix([10000, 784]), -1, 1));

console.time('timer');
sft.startRegression(x, mathjs.matrix(y));