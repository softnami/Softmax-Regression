"use strict";


class SoftmaxRegression {

  /**
   * This method serves as the constructor for the SoftmaxRegression class.
   * @method constructor
   * @param {Object} args These are the required arguments.
   */
  constructor(args) {
    if (args.notify_count === undefined || args.weight_initialization_range === undefined || args.threshold === undefined || args.momentum === undefined || args.batch_size === undefined || args.learningRate === undefined || args.parameter_size === undefined || args.max_iterations === undefined || args.iteration_callback === undefined) {
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
   * This method serves as the logic for generating the natural exponent matrix in softmax regression.
   *
   * @method exp_matrix
   * @param {matrix} W The matrix to be used as the weights for the exp_matrix.
   * @param {matrix} X The matrix to be used as the input data for the exp_matrix.
   * @return {matrix} exp_term Returns the exp_term matrix.
   */
  exp_matrix(W, X) {

    var scope = {
        x: this.MathJS.matrix(X),
        bias: this.MathJS.matrix(X).size().length===1?this.bias._data[0]:this.bias,
        W: (typeof(W) === "number") ? this.MathJS.matrix([
          [W]
        ]) : this.W,
        W_transpose: {}
      },
      exp_term;

    exp_term = this.MathJS.eval('e.^(x*W+bias)', scope);

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

    var result = this.MathJS.eval('exp_term.*(1/sum)', scope),
      num;

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
    scope.cost_n = 0;
    exp_matrix = this.exp_matrix(W, X);


    scope.p_matrx = this.MathJS.transpose(this.hypothesis(exp_matrix));
    scope.y = (Y);
    scope.cross_entropy = this.MathJS.eval('-1*(log(p_matrx))*y', scope);

    scope.size = X.size()[0];
    scope.W = W;
    scope.weights_sum = this.MathJS.sum(this.MathJS.eval('W.^2', scope));
    scope.regularization_constant = this.regularization_parameter;
    scope.regularization_matrix = this.MathJS.eval('(regularization_constant/ 2) * weights_sum', scope);

    scope.cross_entropy = this.MathJS.eval('cross_entropy+regularization_matrix', scope);
    scope.cost = 0.5 * this.MathJS.mean(scope.cross_entropy);

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
    var cost = 0;
    var gradient = [],
      probability_matrx = [],
      exp_matrix;
    var X = _X || this.X;
    var Y = _Y || this.Y;
    var W = _W || this.W;
    W = this.MathJS.squeeze(W);

    for (let i = 0; i < X.size()[0]; i++) {
      exp_matrix = this.exp_matrix(W, X._data[i]);
      probability_matrx[i] = (this.hypothesis(exp_matrix))._data;
    }

    // w-> n_feat x n_classes
    let scope = {};
    scope.gradient = [];

    scope.probability_matrx = this.MathJS.squeeze(this.MathJS.matrix(probability_matrx));
    scope.y = this.MathJS.matrix(Y);
    scope.x = this.MathJS.transpose(X); //num of features * num of samples
    scope.difference = this.MathJS.eval('probability_matrx-y', scope); //num of samples x num of classes
    scope.gradient = this.MathJS.multiply(scope.x, scope.difference); //number of features x number of classes
    scope.difference_mean = this.MathJS.mean(scope.difference,0);
    scope.difference_mean = this.MathJS.matrix([scope.difference_mean]);
    scope.ones = this.MathJS.ones(this.batch_size, 1);
    scope.size = scope.difference.size()[0];
    scope.learningRate = this.learningRate;
    scope.bias = this.bias;

    this.bias = this.MathJS.eval('bias-ones*difference_mean*size*learningRate',scope);

    scope.size = X.size()[0];
    scope.regularization_constant = this.regularization_parameter;
    scope.W = W;
    scope.regularization_matrix = this.MathJS.squeeze(this.MathJS.eval('W.*regularization_constant', scope));

    return this.MathJS.eval('(gradient)+regularization_matrix', scope);
  }

  /**
   * This method serves as the logic for validating X, Y and the Weights(W).
   *
   * @method validateXYW
   **/
  validateXYW() {
    var self = this;

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
   * This method serves as the logic for the gradient descent.
   *
   * @method startRegression
   * @param {matrix} Y The matrix to be used as the label data for training.
   * @param {matrix} X The matrix to be used as the input data for training.
   */
  startRegression(X, Y) {
    var iterations = 0,
      counter = 0;
    var scope = {};

    this.X = X;
    this.Y = Y;
    var self = this;
    this.W = this.MathJS.random(this.MathJS.matrix([self.parameter_size[0], self.parameter_size[1]]), -5, 5);
    this.bias = this.MathJS.ones(self.batch_size, self.parameter_size[1]);
    scope.ranmatrx = this.MathJS.random(this.MathJS.matrix([self.parameter_size[0], self.parameter_size[1]]), -0.9, 0.001);
    scope.W = this.W;
    this.W = this.MathJS.eval('ranmatrx.*W', scope);
    scope.cost = 0;
    scope.v = this.MathJS.random(this.MathJS.matrix([self.parameter_size[0], self.parameter_size[1]]), 0, 0);

    this.validateXYW();

    while (true) {

      this.W._data[0] = this.MathJS.zeros(1, self.parameter_size[1])._data[0];
      scope.X = this.MathJS.matrix(this.X._data.slice(counter * this.batch_size, counter * this.batch_size + this.batch_size));
      scope.Y = this.MathJS.matrix(this.Y._data.slice(counter * this.batch_size, counter * this.batch_size + this.batch_size));
      scope.gamma = this.momentum;
      scope.cost = this.costFunction(undefined, scope.X, scope.Y);
      scope.gradient = this.costFunctionGradient(undefined, scope.X, scope.Y);
      scope.iterations = iterations;
      scope.learningRate = this.learningRate;
      scope.v = this.MathJS.eval('(gamma*v)+(gradient.*learningRate)', scope); //momentum for Stochastic Gradient Descent.
      this.W = this.MathJS.eval('W-v', scope);
      scope.W = this.W;

      if (iterations % this.notify_count === 0) {
        this.iteration_callback(scope);
      }

      iterations++;
      counter++;

      if ((counter * this.batch_size + this.batch_size) > this.X.size()[0]) {
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

