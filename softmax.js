"use strict";

import * as math from 'mathjs';

export class SoftmaxRegression {

  /**
   * This method serves as the constructor for the SoftmaxRegression class.
   * @class SoftmaxRregression
   * @constructor
   * @param {Object} args These are the required arguments.
   * @param {Number} args.learningRate Learning rate for BackPropogation.
   * @param {Number} args.threshold Threshold value for cost.
   * @param {Number} args.regularization_parameter Regularization parameter to prevent overfitting. 
   * @param {Number} args.notify_count Value to execute the iteration_callback after every x number of epochs. 
   * @param {Function} args.iteration_callback Callback that can be used for getting cost and epoch value on every notify count. 
   * @param {Number} args.maximum_epochs Maximum epochs to be allowed before the optimization is complete. Defaults to 1000.
   * @param {Array} args.weight_initilization_range Array contains the lower bound and upper bound for random weight initilization: [lower bound, upper bound].
   * @param {Array} args.parameter_size Array contains the number of input features and total number of output classes: [number of  input features, total number of  output classes].
   */
  constructor(args) {
    if (args.notify_count === undefined || args.weight_initialization_range === undefined || args.threshold === undefined || args.momentum === undefined || args.batch_size === undefined || args.learningRate === undefined || args.parameter_size === undefined || args.max_epochs === undefined || args.iteration_callback === undefined) {
      throw ({
        'name': "Invalid Param",
        'message': "The required constructor parameters cannot be empty."
      });
    }

    this.initArgs = args;
    this.MathJS = math;
    this.batch_size = this.initArgs.batch_size;
    this.momentum = this.initArgs.momentum;
    this.notify_count = this.initArgs.notify_count;
    this.max_epochs = this.initArgs.max_epochs;
    this.weight_initialization_range = this.initArgs.weight_initialization_range;
    this.iteration_callback = this.initArgs.iteration_callback;
    this.parameter_size = this.initArgs.parameter_size;
    this.learningRate = this.initArgs.learningRate;
    this.threshold = this.initArgs.threshold;
    this.regularization_parameter = this.initArgs.regularization_parameter;
    this.random_vals = {};
    this.X = {};
    this.Y = {};
    this.W = {};
  }

   /**
   * This method serves as the logic for generating the natural exponent matrix in softmax.
   *
   * @method exp_matrix
   * @param {matrix} _W The matrix to be used as the weights for the exp_matrix.
   * @param {matrix} _X The matrix to be used as the input data for the exp_matrix.
   * @param {matrix} _bias The matrix to be used as the bias for the exp_matrix.
   * @return {matrix} Returns the exp_term matrix.
   */
  exp_matrix(_W, _X, _bias) {
    let scope = {
        x: _X,
        W: (_W||this.W),
        bias: _bias || this.bias,
        max: 0
      },
      exp_term, product;

    if(scope.bias.size()[0]!==scope.x.size()[0]){
      scope.ones =  this.MathJS.ones(scope.x.size()[0], scope.bias.size()[0]);
      scope.resized_bias = this.MathJS.eval('ones*bias',scope);
      this.bias = scope.resized_bias;
      product = this.MathJS.eval('(x*W + resized_bias)', scope);
    }else{
      product = this.MathJS.eval('(x*W + bias)', scope);
    }

    scope.max = this.MathJS.max(product);
    scope.product = product;
    exp_term = this.MathJS.eval('e.^(product-max)', scope);

    return exp_term;
  }

  /**
   * This method serves as the logic for hypothesis.
   *
   * @method hypothesis
   * @param {matrix} exp_term The matrix to be used as the exp_term.
   * @return {matrix} Returns the hypothesis matrix.
   */
  hypothesis(exp_term) {

    let sum = this.MathJS.mean(exp_term,0);
    let scope = {
      exp_term: exp_term,
      sum: sum,
      ones: {},
      size: sum.size()[0]
    };

        scope.ones = this.MathJS.squeeze(this.MathJS.ones(1, scope.exp_term.size()[1]));
        scope.sum = this.MathJS.eval('sum*ones*size',scope);
        let result = this.MathJS.eval('exp_term.*(1/(sum))', scope);

    return result;
  }



  /**
   * This method serves as the logic for the costFunction.
   *
   * @method costFunction
   * @param {matrix} _W The matrix to be used as the weights.
   * @param {matrix} _X The matrix to be used as the input data.
   * @param {matrix} _Y The matrix to be used as the label data.
   * @return {Numer} Returns the cost.
   */
  costFunction(_W, _X, _Y) {

    let cost = 0,
      exp_matrix, probability_matrx = [];
    let W = _W || this.W;
    let X = _X || this.X;
    let Y = _Y || this.Y;

    let scope = {};
    scope.cost = 0;
    scope.cost_n = 0;
    exp_matrix = this.exp_matrix(W, X);


    scope.p_matrx = (this.hypothesis(exp_matrix));

    scope.y = (Y);

    scope.p_matrx = scope.p_matrx.map(function(value) { //case for numerical instability.
      if (value <= 0.001) {
        return 0.001;
      }
      if (value >= 1) {
        return 0.999;
      }

      return value;
    });

    scope.cross_entropy = this.MathJS.eval('(y.*log(p_matrx))', scope);
    scope.size = X.size()[0];
    scope.W = W;
    scope.weights_sum = this.MathJS.sum(this.MathJS.eval('W.^2', scope));
    scope.regularization_constant = this.regularization_parameter;
    scope.regularization_matrix = this.MathJS.eval('(regularization_constant/ 2) * weights_sum', scope);

    scope.cross_entropy = this.MathJS.mean(scope.cross_entropy);
    scope.cost = this.MathJS.sum(scope.regularization_matrix) - scope.cross_entropy;

    return scope.cost;
  }

  /**
   * This method serves as the logic for the cost function gradient.
   *
   * @method costFunctionGradient
   * @param {matrix} _W The matrix to be used as the weights.
   * @param {matrix} _X The matrix to be used as the input data.
   * @param {matrix} _Y The matrix to be used as the label data.
   
   * @return {matrix} Returns the cost gradient.
   */
  costFunctionGradient(_W, _X, _Y) {
    let cost = 0;
    let gradient = [],
      probability_matrx = [],
      exp_matrix;
    let X = _X || this.X;
    let Y = _Y || this.Y;
    let W = _W || this.W;
    W = this.MathJS.squeeze(W);

    exp_matrix = this.exp_matrix(W, X);
    probability_matrx = (this.hypothesis(exp_matrix));

    // w-> n_feat x n_classes
    let scope = {};
    scope.gradient = [];

    scope.probability_matrx = this.MathJS.squeeze(this.MathJS.matrix(probability_matrx));
    scope.y = this.MathJS.matrix(Y);
    scope.x = this.MathJS.transpose(X); //num of features * num of samples
    scope.difference = this.MathJS.eval('y-probability_matrx', scope); //num of samples x num of classes

    scope.gradient = this.MathJS.multiply(scope.x, scope.difference); //number of features x number of classes

    scope.difference_mean = this.MathJS.mean(scope.difference,0);
    scope.difference_mean = this.MathJS.matrix([scope.difference_mean]);

    scope.ones = this.MathJS.ones(this.batch_size, 1);
    scope.size = scope.difference.size()[0];
    scope.learningRate = this.learningRate;

    scope.bias_update = this.MathJS.eval('ones*difference_mean*size',scope);

    scope.size = X.size()[0];
    scope.regularization_constant = this.regularization_parameter;
    scope.W = W;
    scope.regularization_matrix = this.MathJS.squeeze(this.MathJS.eval('W.*regularization_constant', scope));

    return [this.MathJS.eval('(-1*gradient/size)+regularization_matrix', scope), scope.bias_update];
  }

  /**
   * This method serves as the logic for validating X, Y and the Weights(W).
   *
   * @method validateXYW
   **/
  validateXYW() {
    let self = this;

    if (self.X.size()[1] != self.W.size()[0]) {
      throw ({
        'name': "Invalid size",
        'message': "The number of columns in X should be equal to the number of rows in parameter_size."
      });
    }

    if (self.Y.size()[1] != self.W.size()[1]) {
      throw ({
        'name': "Invalid size",
        'message': "The number of columns in Y should be equal to the number of columns in parameter_size."
      });
    }

    if (self.Y.size()[0] != self.X.size()[0]) {
      throw ({
        'name': "Invalid size",
        'message': "The number of rows in X should be equal to the number of rows in Y."
      });
    }

    if (!(self.Y.size()[0] > 2) || !(self.X.size()[0] > 2)) {
      throw ({
        'name': "Invalid size",
        'message': "The number of rows in X andy Y should be greater than 2."
      });
    }
  }


/**
 * This method returns all the parameters passed to the constructor.
 *
 * @method getInitParams
 * @return {Object} Returns the constructor parameters.
 */
getInitParams() {
  return {
    'notify_count': this.notify_count,
    'momentum': this.momentum,
    'parameter_size': this.parameter_size,
    'max_epochs': this.max_epochs,
    'weight_initialization_range': this.weight_initialization_range,
    'threshold': this.threshold,
    'learningRate': this.learningRate,
    'batch_size': this.batch_size,
    'regularization_parameter': this.regularization_parameter,
    'max_epochs': this.max_epochs,
    'iteration_callback': this.iteration_callback
  };
};


  /**
 * This method serves as the logic for softmax function.
 *
 * @method process
 * @param {matrix} X The matrix to be used as the input data.
 * @param {matrix} W The matrix to be used as the weights.
 * @return {matrix} Returns the softmax matrix.
 */
  predict(X, W, bias) {
    let expMatrix;

    this.setWeights();

    let _X = X;
    let _W = W || this.W;
    let _bias = bias || this.bias;

    expMatrix= this.exp_matrix(_W, _X, _bias);
    
    let softmaxMatrix = this.hypothesis(expMatrix);

    return new Promise((resolve, reject) => {
      resolve(softmaxMatrix);
    });
  }


/**
 *This method is responsible for setting bias for the Softmax model.
 *
 * @method setBias 
 * @param {Number} bias The bias for Softmax model.
 */
setBias(bias) {
  this.bias = bias;
};


/**
 *This method is responsible for setting weights and biases for the Softmax model from storage.
 *
 * @method setWeights 
 * @return {Object} Returns a resolved promise after successfuly setting weights and biases.
 */
 setWeights(){
    let self = this;
    let weights, biases;

    weights = JSON.parse(localStorage.getItem("Weights"));
    biases = JSON.parse(localStorage.getItem("Biases"));

    if(weights!==null && weights!==undefined){

      self.W = this.MathJS.matrix(weights.data);
      self.bias= this.MathJS.matrix(biases.data);
      return [self.W._data, self.bias];
    }

    return [];
}


  /**
 *This method is responsible for saving the trained weights and biases for the Softmax model.
 *
 * @method saveWeights 
 * @param {Matrix} weights The weights for the Softmax model.
 * @param {Matrix} biases The biases for the Softmax model.
 * @return {Boolean} Returns true after succesfuly saving the weights.
 */
saveWeights(weights, biases) {
 
  localStorage.setItem("Weights", JSON.stringify(weights));
  localStorage.setItem("Biases", JSON.stringify(biases));

  console.log("\nWeights were successfuly saved.");
  return true;
}


/**
* Randomize matrix element order in-place using Durstenfeld shuffle algorithm.
*/
 shufflematrix(matrix, matrix2) {
            for (let i = matrix.length - 1; i > 0; i--) {
                let j = Math.floor(Math.random() * (i + 1));
                
                let temp = matrix[i];
                matrix[i] = matrix[j];
                matrix[j] = temp;

                let temp2 = matrix2[i];
                matrix2[i] = matrix2[j];
                matrix2[j] = temp2;
            }
            return [matrix, matrix2];
  }


  /**
   * This method serves as the logic for the gradient descent.
   *
   * @method startRegression
   * @param {matrix} Y The matrix to be used as the label data for training.
   * @param {matrix} X The matrix to be used as the input data for training.
   */
  startRegression(X, Y) {
    let iterations = 0,
      counter = 0;
    let scope = {};

    this.X = X;
    this.Y = Y;
    let self = this;
    this.W = this.MathJS.random(this.MathJS.matrix([self.parameter_size[0], self.parameter_size[1]]), this.weight_initialization_range[0], this.weight_initialization_range[1]);
    this.bias = this.MathJS.matrix([this.MathJS.ones(self.parameter_size[1])._data]);
    scope.v = this.MathJS.random(this.MathJS.matrix([self.parameter_size[0], self.parameter_size[1]]), 0, 0);
    scope.epochs = 0;
    scope.W = this.W;
    scope.cost = 0;
    this.validateXYW();

    while (true) {

      this.W._data[0] = this.MathJS.zeros(1, self.parameter_size[1])._data[0];
      scope.X = this.MathJS.matrix(this.X._data.slice(counter * this.batch_size, counter * this.batch_size + this.batch_size));
      scope.Y = this.MathJS.matrix(this.Y._data.slice(counter * this.batch_size, counter * this.batch_size + this.batch_size));
      scope.gamma = this.momentum;

      let gradinfo = this.costFunctionGradient(undefined, scope.X, scope.Y);

      scope.gradient = gradinfo[0];
      scope.bias_update = gradinfo[1];
      scope.iterations = iterations;
      scope.learningRate = this.learningRate;
      scope.bias = this.bias;
  	  scope.v = this.MathJS.eval('(v.*gamma)+(gradient.*learningRate)', scope); //momentum for Stochastic Gradient Descent.
      this.W = this.MathJS.eval('W-v', scope);      
      this.bias = this.MathJS.eval('bias-(bias_update.*learningRate)',scope);
      scope.W = this.W;

      iterations++;
      counter++;

       if (((counter) * this.batch_size + this.batch_size) > this.X.size()[0]) {//cost for whole epoch.
        scope.cost = this.costFunction(undefined, scope.X, scope.Y);
        scope.epochs++;
        this.iteration_callback(scope);
        let shuffled = this.shufflematrix(this.X._data, this.Y._data);
        this.X = this.MathJS.matrix(shuffled[0]);
        this.Y = this.MathJS.matrix(shuffled[1]);
         counter = 0;
      }

      if (scope.epochs >= this.max_epochs || (scope.cost < this.threshold && scope.cost > 0)) {
        console.log("\nTraining completed.\n");
        this.saveWeights(this.W, this.bias);
        return new Promise((resolve, reject) => {
          resolve(scope);
        });
      }
    }
  }

}

