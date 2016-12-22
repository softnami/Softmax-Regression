var mathjs = require('mathjs');
var counter = 0,
  cost_previous = 0;
var callback = function(data) {
  if (counter === 1) {
    counter = 0;
    console.log(cost_previous / 1, data.iterations, data.X._data[0]);
    cost_previous = 0;
  }
  cost_previous += data.cost;
  counter++;
};

var sft = new SoftmaxRegression({
  'notify_count': 1,
  'momentum': 0.9,
  'parameter_size': [12, 4], //[number of features, total number of classes]
  'max_iterations': 1000000000,
  'weight_initialization_range': [-1.5, 1.5],
  'threshold': 0.1,
  'batch_size': 17,
  'iteration_callback': callback,
  'learningRate': 0.08,
  'regularization_parameter': Math.exp(-3)
});

var sample_y = [
  [1, 0, 0, 0],
  [0, 1, 0, 0],
  [0, 0, 1, 0],
  [0, 0, 0, 1]
]; //one-hot encoded classes: [0,1,2,3] -> [[1, 0,0,0],[0, 1,0,0],[0, 0,1,0],[0, 0,0,1]]
var y = [],
  num;


for (let i = 0; i < 10000; i++) {
  num = Math.floor((Math.random() * 4) + 0);
  y.push(sample_y[num]);
}

var x = mathjs.floor(mathjs.random(mathjs.matrix([10000, 12]), 0, 2));

for (let i = 0; i < x.size()[0]; i++) {
  x._data[i][0] = 1;
}

sft.startRegression(x, mathjs.matrix(y));