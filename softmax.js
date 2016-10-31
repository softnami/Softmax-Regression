"use strict";


class SoftmaxRegression {

  /**
   * This method serves as the constructor for the SoftmaxRegression class.
   * @method constructor
   * @param {Object} args These are the required arguments.
   */
  constructor(args) {
    if (args.notify_count === undefined || args.threshold === undefined || args.momentum === undefined || args.batch_size === undefined || args.learningRate === undefined || args.parameter_size === undefined || args.max_iterations === undefined || args.iteration_callback === undefined) {
      throw ({
        'name': "InvalidParam",
        'message': "The required constructor parameters cannot be empty."
      });
    }
    this.MathJS = require('mathjs');
    this.initArgs = args;
    this.batch_size = this.initArgs.batch_size;
    this.momentum = this.initArgs.momentum;
    this.notify_count = this.initArgs.notify_count || 100;
    this.max_iterations = this.initArgs.max_iterations || 1000;
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
   * This method serves as the logic for generating the natural exponent matrix in softmax regression.
   *
   * @method exp_matrix
   * @param {matrix} W The matrix to be used as the weights for the exp_matrix.
   * @param {matrix} X The matrix to be used as the input data for the exp_matrix.
   * @return {matrix} exp_term Returns the exp_term matrix.
   */
  exp_matrix(W, X) {

    var scope = {
        x: X[0],
        W: (typeof(W) === "number") ? this.MathJS.matrix([
          [W]
        ]) : this.W,
        W_transpose: {}
      },
      exp_term;

    scope.W_transpose = this.MathJS.transpose(scope.W);
    exp_term = this.MathJS.eval('e.^((W_transpose*x))', scope);

    return exp_term;
  }

  /**
   * This method serves as the logic for hypothesis.
   *
   * @method hypothesis
   * @param {matrix} exp_term The matrix to be used as the exp_term.
   * @return {matrix} result Returns the hypothesis matrix.
   */
  hypothesis(exp_term) {
    var sum = this.MathJS.sum(exp_term);
    var scope = {
      exp_term: exp_term,
      sum: sum
    };
    var result = this.MathJS.eval('exp_term.*(1/sum)', scope);

    return result;
  }

  /**
   * This method serves as the logic for the costFunction.
   *
   * @method costFunction
   * @param {matrix} W The matrix to be used as the weights.
   * @param {matrix} X The matrix to be used as the input data.
   * @param {matrix} Y The matrix to be used as the label data.
   * @return {Numer} Returns the cost.
   */
  costFunction(W, X, Y) {

    var cost = 0,
      exp_matrix, probability_matrx = [];
    var W = W || this.W;
    var X = X || this.X;
    var Y = Y || this.Y;

    var scope = {};
    scope.cost = 0;

    for (let i = 0; i < X.size()[0]; i++) {
      exp_matrix = this.exp_matrix(W, X._data[i]);
      probability_matrx[i] = this.hypothesis(exp_matrix)._data;
    }

    for (let j = 0; j < W.size()[0]-1; j++) {
      for (let n = 0; n < X.size()[0]; n++) {
        if (Y._data[n][0] === j) {
          scope.p_matrx = (probability_matrx[n][0][j]);
          scope.cost = this.MathJS.eval('cost + log(p_matrx)', scope);
        }
      }
    }

    scope.size = X.size()[0];
    scope.W = W;
    scope.weights_sum = this.MathJS.sum(this.MathJS.eval('W.^2', scope));
    scope.regularization_constant = this.regularization_parameter;
    scope.regularization_matrix = this.MathJS.eval('(regularization_constant/ 2) * weights_sum', scope);

    var finalcost = this.MathJS.eval('(-1*cost./size)+regularization_matrix', scope);


    return this.MathJS.sum(finalcost);
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
    var cost = 0;
    var gradient = [],
      probability_matrx = [
        []
      ],
      exp_matrix;
    var X = _X || this.X;
    var Y = _Y || this.Y;
    var W = _W || this.W;

    for (let i = 0; i < X.size()[0]; i++) {
      exp_matrix = this.exp_matrix(W, X._data[i]);
      probability_matrx[i] = this.MathJS.transpose(this.hypothesis(exp_matrix))._data;
    }

    for (let j = 0; j < W.size()[0]; j++) {
      let scope = {};
      scope.gradient_previous = this.MathJS.zeros(W.size()[0], W.size()[1]);

      scope.gradient = [];
      scope.gradient[j] = [];
      for (let i = 0; i < X.size()[0]; i++) {
        scope.x = X._data[i][0];
        if (Y._data[i][0] === j) {
          scope.probability_matrx = (probability_matrx[i]);
          scope.gradient[j][i] = this.MathJS.eval('(1-probability_matrx).*x', scope);
        } else if (Y._data[i][0] !== j) {
          scope.probability_matrx = (probability_matrx[i]);
          scope.gradient[j][i] = this.MathJS.eval('(probability_matrx.*-1).*x', scope);
        }

        scope.gradient_current = this.MathJS.matrix(scope.gradient[j][i]);
        gradient[j] = this.MathJS.eval('(gradient_current+gradient_previous)', scope)._data;//Summing up gradients for all examples for a certain class j.
        scope.gradient_previous = gradient[j];
      }
    }


    let scope = {};
    scope.gradient = gradient[0];
    scope.size = X.size()[0];
    scope.regularization_constant = this.regularization_parameter;
    scope.W = W;
    scope.regularization_matrix = this.MathJS.eval('W.*regularization_constant', scope);


    return this.MathJS.eval('(-1*gradient./size)+regularization_matrix', scope);
  }

  /**
   * This method serves as the logic for the gradient descent.
   *
   * @method startRegression
   * @param {matrix} Y The matrix to be used as the label data for training.
   * @param {matrix} X The matrix to be used as the input data for training.
   */
  startRegression(X, Y) {

    var iterations = 0, counter = 0;

    this.X = X;
    this.Y = Y;
    var self = this;
    this.W = this.MathJS.random(this.MathJS.matrix([self.parameter_size[1], self.parameter_size[0]]), 0.001,0.05);
    var scope = {};
        scope.v =  this.MathJS.random(this.MathJS.matrix([self.parameter_size[1], self.parameter_size[0]]), 0,0);//zeros
        scope.cost = 0;

    while (true) {
      
      this.W._data[0] = this.MathJS.zeros(1, self.parameter_size[0])._data[0];
      scope.X = this.MathJS.matrix(this.X._data.slice(counter,counter+this.batch_size));
      scope.W = this.W;
      scope.gamma = this.momentum;
      scope.cost = this.costFunction(undefined, scope.X, undefined);
      scope.gradient = this.costFunctionGradient(undefined, scope.X, undefined);
      scope.iterations = iterations;
      scope.learningRate = this.learningRate;
      scope.v = this.MathJS.eval('(gamma*v)+(gradient.*learningRate)',scope);//momentum for Stochastic Gradient Descent.
      this.W = this.MathJS.eval('W-v', scope); 

      if (iterations % this.notify_count === 0) {
        scope.cost = scope.cost;
        this.iteration_callback(scope);
      }

      iterations++;
      counter=counter+this.batch_size;

      if(counter>this.X.size()[0]-11){
        counter = 0;
      }

      if (iterations > this.max_iterations || (scope.cost < this.threshold)) {
        console.log("\nTraining completed.\n");
        break;
      }

    }

  }

}

module.exports = SoftmaxRegression;


var mathjs = require('mathjs');
var callback = function(data) {

  console.log(data.cost, data.iterations, data.W);
};

var sft = new SoftmaxRegression({
  'notify_count': 1000,
  'momentum': 0.5,
  'parameter_size': [1, 5],//[depth of each class, total number of classes]
  'max_iterations': 1000000000,
  'threshold': 0.1,
  'batch_size': 25,
  'iteration_callback': callback,
  'learningRate': 0.005,
  'regularization_parameter': 0.0001
});


sft.startRegression(mathjs.floor(mathjs.random(mathjs.matrix([10000, 1]), 0, 2)), mathjs.floor(mathjs.random(mathjs.matrix([10000, 1]), 2, 14)));