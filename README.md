

var mathjs = require('mathjs');
var epoch = 1;
var callback = function(data) {
  if (data.iterations*25%10000 === 0) {
    console.timeEnd('timer');
    console.log("Cost: "+data.cost,"Epoch :"+epoch);
    epoch++;
    console.time('timer');
  }
};

var sft = new SoftmaxRegression({
  'notify_count': 1,
  'momentum': 0.9,
  'parameter_size': [256, 15], //[number of  input features, total number of  output classes]
  'max_iterations': 1000000000,
  'weight_initialization_range': [-0.00000000001, 0.00000000001],
  'threshold': 0.1,
  'batch_size': 25,
  'decay_constant': 5,
  'iteration_callback': callback,
  'learningRate': 0.001,
  'regularization_parameter': Math.exp(-3)
});

var sample_y = []; //one-hot encoded classes: [1,2,3] -> [[1, 0,0],[0, 1,0],[0, 0,1]]

for(let i =0 ; i< 15; i++){
  sample_y[i] = [];
  for(let j =0; j< 15; j++){
    if(i==j){
      sample_y[i][j]= 1;
    }
    else{
      sample_y[i][j]= 0;
    }
  }  
}


var y = [],
  num;


for (let i = 0; i < 10000; i++) {
  num = Math.floor((Math.random() * 15) + 0);
  y.push(sample_y[num]);
}

var x = mathjs.floor((mathjs.random(mathjs.matrix([10000, 256]), 0, 2)));

console.time('timer');
sft.startRegression(x, mathjs.matrix(y));