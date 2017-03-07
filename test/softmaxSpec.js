'use strict';

let Softmax = require('../softmax');
let assert = require('assert');
let mathjs = require('mathjs');
let sinon = require('sinon');

describe('Softmax', function() {

  global.localStorage = (function() {
    let storage = {};

    return {
      setItem: function(key, value) {
        storage[key] = value || '';
      },
      getItem: function(key) {
        return storage[key] || null;
      },
      removeItem: function(key) {
        delete storage[key];
      },
      get length() {
        return Object.keys(storage).length;
      },
      key: function(i) {
        let keys = Object.keys(storage);
        return keys[i] || null;
      }
    };
  })();

  let callback_data;

  let callback = function(data) {
    callback_data = data;
    console.log('Epochs:'+ data.epochs, 'iterations:'+ data.iterations, 'cost: '+ data.cost);
  }

  let softmax = new Softmax({
    'notify_count': 1,
    'momentum': 0.5,
    'parameter_size': [78, 12], //[number of  input features, total number of  output classes]
    'max_epochs': 100,
    'weight_initialization_range': [-0.5, 0.5],
    'threshold': 0.1,
    'batch_size': 25,
    'iteration_callback': callback,
    'learningRate': 0.009,
    'regularization_parameter': Math.exp(-4)
  });

  let getInitParams = softmax.getInitParams();

  it("should correctly set parameters", function() {
    assert.deepStrictEqual(getInitParams.notify_count, 1);
    assert.deepStrictEqual(getInitParams.momentum, 0.5);
    assert.deepStrictEqual(getInitParams.parameter_size, [78, 12]);
    assert.deepStrictEqual(getInitParams.max_epochs, 100);
    assert.deepStrictEqual(getInitParams.weight_initialization_range, [-0.5, 0.5]);
    assert.deepStrictEqual(getInitParams.threshold, 0.1);
    assert.deepStrictEqual(getInitParams.batch_size, 25);
    assert.deepStrictEqual(getInitParams.iteration_callback, callback);
    assert.deepStrictEqual(getInitParams.learningRate, 0.009);
    assert.deepStrictEqual(getInitParams.regularization_parameter, Math.exp(-4));

  });

  describe('when processing input data and weights.', function() {

    let X = (mathjs.random(mathjs.matrix([256, 78]), 0, 1)),
      W = (mathjs.random(mathjs.matrix([78, 12]), 0, 1)),
      bias = mathjs.ones(256, 12);

    it('should successfuly call exp_matrix method.', function(done) {
      let spy = sinon.spy(softmax, "exp_matrix");

      softmax.predict(X, W, bias).then((data) => {
        assert.equal((data.size()[0] === X.size()[0] && data.size()[1] === W.size()[1] && spy.callCount === 1), true);
        spy.restore();
        done();
      });

    });

    it('should successfuly call hypothesis method.', function(done) {
      let spy = sinon.spy(softmax, "hypothesis");

      softmax.predict(X, W, bias).then((data) => {
        assert.equal((data.size()[0] === X.size()[0] && data.size()[1] === W.size()[1] && spy.callCount === 1), true);
        spy.restore();
        done();
      });

    });


    it('should be able process matrices with 0 values.', function(done) {
      let X = (mathjs.random(mathjs.matrix([256, 78]), 0, 0)),
        W = (mathjs.random(mathjs.matrix([78, 12]), 0, 0));
      let flag = false;

      softmax.predict(X, W, bias).then((data) => {
        //check for NaN or infinity
        for (let i = 0; i < data._data.length; i++) {
          for (let j = 0; j < data._data[0].length; j++) {
            if (isNaN(data._data[i][j]) || data._data[i][j] === Number.POSITIVE_INFINITY || data._data[i][j] === Number.NEGATIVE_INFINITY) {
              flag = true;
              break;
            }
          }
        }

        assert.equal((data.size()[0] === X.size()[0] && data.size()[1] === W.size()[1] && !flag), true);
        done();
      });

    });


    it('should be able process matrices with very small values.', function(done) {
      let X = (mathjs.random(mathjs.matrix([256, 78]), 0, 1)),
        W = (mathjs.random(mathjs.matrix([78, 12]), (Number.MIN_VALUE), (Number.MIN_VALUE)));
      let flag = false;

      softmax.predict(X, W, bias).then((data) => {

        //check for NaN or infinity
        for (let i = 0; i < data._data.length; i++) {
          for (let j = 0; j < data._data[0].length; j++) {
            if (isNaN(data._data[i][j]) || data._data[i][j] === Number.POSITIVE_INFINITY || data._data[i][j] === Number.NEGATIVE_INFINITY) {
              flag = true;
              break;
            }
          }
        }

        assert.equal((data.size()[0] === X.size()[0] && data.size()[1] === W.size()[1]), true);
        done();
      });

    });


    it('should be able process matrices with very large values.', function(done) {

      let X = (mathjs.random(mathjs.matrix([256, 78]), 0, 1)),
        W = (mathjs.random(mathjs.matrix([78, 12]), mathjs.sqrt(Number.MAX_VALUE - 100000), mathjs.sqrt(Number.MAX_VALUE)));
      let flag = false;

      softmax.predict(X, W, bias).then((data) => {

        //check for NaN or infinity
        for (let i = 0; i < data._data.length; i++) {
          for (let j = 0; j < data._data[0].length; j++) {
            if (isNaN(data._data[i][j]) || data._data[i][j] === Number.POSITIVE_INFINITY || data._data[i][j] === Number.NEGATIVE_INFINITY) {
              flag = true;
              break;
            }
          }
        }

        assert.equal((data.size()[0] === X.size()[0] && data.size()[1] === W.size()[1]), true);
        done();
      });

    });


    it('should be able process matrices with negative values.', function(done) {
      let X = (mathjs.random(mathjs.matrix([256, 78]), 0, 1)),
        W = (mathjs.random(mathjs.matrix([78, 12]), -50, 0));
      let flag = false;

      softmax.predict(X, W, bias).then((data) => {
        //check for NaN or infinity
        for (let i = 0; i < data._data.length; i++) {
          for (let j = 0; j < data._data[0].length; j++) {
            if (isNaN(data._data[i][j]) || data._data[i][j] === Number.POSITIVE_INFINITY || data._data[i][j] === Number.NEGATIVE_INFINITY) {
              flag = true;
              break;
            }
          }
        }
        assert.equal((data.size()[0] === X.size()[0] && data.size()[1] === W.size()[1]), true);
        done();
      });

    });

  });

  describe('when saving and setting weights', function() {

    let W = (mathjs.random(mathjs.matrix([256, 78]), 0, 1)),
      bias = mathjs.ones(256, 12);


    it('should successfuly save weights', function(done) {
      softmax.saveWeights(W, bias);
      assert.deepStrictEqual([JSON.parse(global.localStorage.getItem("Weights")).data, JSON.parse(global.localStorage.getItem("Biases")).data], [W._data, bias._data]);
      done();
    });

    it('should successfuly set weights', function(done) {
      assert.deepStrictEqual(softmax.setWeights(), [W._data, bias]);
      done();
    });
  });


  describe('when training', function() {

    let X = (mathjs.random(mathjs.matrix([256, 78]), 0, 1));
    let sample_y = []; //one-hot encoded classes: [1,2,3] -> [[1, 0,0],[0, 1,0],[0, 0,1]]

    for (let i = 0; i < 100; i++) {
      sample_y[i] = [];
      for (let j = 0; j < 12; j++) {
        if (i == j) {
          sample_y[i][j] = 1;
        } else {
          sample_y[i][j] = 0;
        }
      }
    }

    let Y = [],
      num = 0;

    for (let i = 0; i < 256; i++) {
      num = Math.floor((Math.random() * 100) + 0);
      Y.push(sample_y[num]);
    }

    Y = mathjs.matrix(Y);


    it("should correctly run costFunctionGradient()", function() {
      var cost = 0;
      var gradient = [],
        probability_matrx = [],
        exp_matrix;
      var X = _X || this.X;
      var Y = _Y || this.Y;
      var W = _W || this.W;
      W = mathjs.squeeze(W);

      exp_matrix = this.exp_matrix(W, X);
      probability_matrx = (this.hypothesis(exp_matrix));

      // w-> n_feat x n_classes
      let scope = {};
      scope.gradient = [];

      scope.probability_matrx = mathjs.squeeze(mathjs.matrix(probability_matrx));
      scope.y = mathjs.matrix(Y);
      scope.x = mathjs.transpose(X); //num of features * num of samples
      scope.difference = mathjs.eval('y-probability_matrx', scope); //num of samples x num of classes

      scope.gradient = mathjs.multiply(scope.x, scope.difference); //number of features x number of classes

      scope.difference_mean = mathjs.mean(scope.difference, 0);
      scope.difference_mean = mathjs.matrix([scope.difference_mean]);

      scope.ones = mathjs.ones(this.batch_size, 1);
      scope.size = scope.difference.size()[0];
      scope.learningRate = this.learningRate;

      scope.bias_update = mathjs.eval('ones*difference_mean*size', scope);

      scope.size = X.size()[0];
      scope.regularization_constant = this.regularization_parameter;
      scope.W = W;
      scope.regularization_matrix = mathjs.squeeze(mathjs.eval('W.*regularization_constant', scope));
      assert.equal(success, true);

    });


    it("should correctly run costFunction()", function() {
      let computed_cost = 0;
      let success = false;

      let W = (mathjs.random(mathjs.matrix([78, 12]), -1, 1)),
        bias = mathjs.matrix([mathjs.ones(12)._data]);

      softmax.setBias(bias);

      let cost = 0,
        exp_matrix, probability_matrx = [];

      let scope = {};
      scope.cost = 0;
      scope.cost_n = 0;
      exp_matrix = softmax.exp_matrix(W, X);


      scope.p_matrx = (softmax.hypothesis(exp_matrix));

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

      scope.cross_entropy = mathjs.eval('(y.*log(p_matrx))', scope);
      scope.size = X.size()[0];
      scope.W = W;
      scope.weights_sum = mathjs.sum(mathjs.eval('W.^2', scope));
      scope.regularization_constant = getInitParams.regularization_parameter;
      scope.regularization_matrix = mathjs.eval('(regularization_constant/ 2) * weights_sum', scope);

      scope.cross_entropy = mathjs.mean(scope.cross_entropy);
      scope.cost = mathjs.sum(scope.regularization_matrix) - scope.cross_entropy;
      let cost_computed = softmax.costFunction(W, X, Y);

      console.log(cost_computed, scope.cost);

      if (cost_computed !== scope.cost)
        success = false;

      assert.equal(success, true);

    });

    it("should call saveWeights() while running startRegression()", function(done) {

      let success = true;
      let spy = sinon.spy(softmax, "saveWeights");

      softmax.startRegression(X, Y).then(function(data) {
        if (!spy.called)
          success = false;
        spy.restore();
        assert.equal(success, true);
        done();

      });

    });

    it("should call costFunctionGradient() while running startRegression()", function(done) {
      let success = true;
      let spy = sinon.spy(softmax, "costFunctionGradient");

      softmax.startRegression(X, Y).then(function(data) {
        if (!spy.called)
          success = false;
        spy.restore();
        assert.equal(success, true);
        done();
      });

    });

    it("should call iteration_callback() after training", function(done) {
      let success = false;

      softmax.startRegression(X, Y).then(function(data) {
        if (data.epochs === callback_data.epochs && (data.cost) === callback_data.cost) {
          success = true;
        }

        assert.equal(success, true);
        done();
      });
    });

    it("should call costFunction() while running startRegression()", function(done) {
      let success = true;
      let spy = sinon.spy(softmax, "costFunction");

      softmax.startRegression(X, Y).then(function(data) {
        if (!spy.called)
          success = false;
        spy.restore();
        assert.equal(success, true);
        done();

      });
    });

    it("should run gradient descent until convergence or iteration limit is reached.", function(done) {
      let success = false;

      softmax.startRegression(X, Y).then(function(data) {
        if ((data.cost <= (1 / mathjs.exp(3))) || (data.cost < (getInitParams.threshold)) || (data.epochs - 1) === getInitParams.max_epochs) {
          success = true;
        }
        assert.equal(success, true);
        done();
      });
    });

    it("should throw an exception if number of columns(features) in X(input) are not equal to the number of rows in W(weights).", function(done) {
      let success = false;
      let X = (mathjs.random(mathjs.matrix([256, 77]), 0, 1)),
        W = (mathjs.random(mathjs.matrix([78, 12]), 0, 1)),
        bias = mathjs.ones(256, 12);

      try {
        softmax.startRegression(X, Y);
      } catch (err) {
        if (err.name === "Invalid size" && err.message === "The number of columns in X should be equal to the number of rows in parameter_size.") {
          success = true;
        }
      }

      assert.equal(success, true);
      done();
    });

     it("should throw an exception if number of rows in Y(input) are not equal to the number of rows in X(input).", function(done) {
      let success = false;
      let X = (mathjs.random(mathjs.matrix([255, 78]), 0, 1)),
        W = (mathjs.random(mathjs.matrix([78, 12]), 0, 1)),
        bias = mathjs.ones(256, 13);
      let sample_y = []; //one-hot encoded classes: [1,2,3] -> [[1, 0,0],[0, 1,0],[0, 0,1]]

      for (let i = 0; i < 100; i++) {
        sample_y[i] = [];
        for (let j = 0; j < 12; j++) {
          if (i == j) {
            sample_y[i][j] = 1;
          } else {
            sample_y[i][j] = 0;
          }
        }
      }

      let Y = [],
        num = 0;

      for (let i = 0; i < 259; i++) {
        num = Math.floor((Math.random() * 100) + 0);
        Y.push(sample_y[num]);
      }

      Y = mathjs.matrix(Y);

      try {
        softmax.startRegression(X, Y);
      } catch (err) { 
        if (err.name === "Invalid size" && err.message === "The number of rows in X should be equal to the number of rows in Y.") {
          success = true;
        }
      }

      assert.equal(success, true);
      done();
    });


    it("should throw an exception if number of columns(features) in Y(input) are not equal to the number of columns in W(weights).", function(done) {
      let success = false;
      let X = (mathjs.random(mathjs.matrix([256, 78]), 0, 1)),
        W = (mathjs.random(mathjs.matrix([78, 12]), 0, 1)),
        bias = mathjs.ones(256, 13);
      let sample_y = []; //one-hot encoded classes: [1,2,3] -> [[1, 0,0],[0, 1,0],[0, 0,1]]

      for (let i = 0; i < 100; i++) {
        sample_y[i] = [];
        for (let j = 0; j < 13; j++) {
          if (i == j) {
            sample_y[i][j] = 1;
          } else {
            sample_y[i][j] = 0;
          }
        }
      }

      let Y = [],
        num = 0;

      for (let i = 0; i < 256; i++) {
        num = Math.floor((Math.random() * 100) + 0);
        Y.push(sample_y[num]);
      }

      Y = mathjs.matrix(Y);

      try {
        softmax.startRegression(X, Y);
      } catch (err) { 
        if (err.name === "Invalid size" && err.message === "The number of columns in Y should be equal to the number of columns in parameter_size.") {
          success = true;
        }
      }

      assert.equal(success, true);
      done();
    });

    it("should throw an exception if number of rows in Y(input) and X(inputs) should be greater than 2.", function(done) {
      let success = false;
      let X = (mathjs.random(mathjs.matrix([2, 78]), 0, 1)),
        W = (mathjs.random(mathjs.matrix([78, 12]), 0, 1)),
        bias = mathjs.ones(256, 13);
      let sample_y = []; //one-hot encoded classes: [1,2,3] -> [[1, 0,0],[0, 1,0],[0, 0,1]]

      for (let i = 0; i < 100; i++) {
        sample_y[i] = [];
        for (let j = 0; j < 12; j++) {
          if (i == j) {
            sample_y[i][j] = 1;
          } else {
            sample_y[i][j] = 0;
          }
        }
      }

      let Y = [],
        num = 0;

      for (let i = 0; i < 2; i++) {
        num = Math.floor((Math.random() * 100) + 0);
        Y.push(sample_y[num]);
      }

      Y = mathjs.matrix(Y);

      try {
        softmax.startRegression(X, Y);
      } catch (err) { 
        if (err.name === "Invalid size" && err.message === "The number of rows in X andy Y should be greater than 2.") {
          success = true;
        }
      }

      assert.equal(success, true);
      done();
    });

    

    describe('when predicting the result', function() {
      let X = (mathjs.random(mathjs.matrix([256, 78]), 0, 1));

      it("should call setWeights(), exp_matrix() and  hypothesis()", function() {
        let spy1 = sinon.spy(softmax, "setWeights");
        let spy2 = sinon.spy(softmax, "exp_matrix");
        let spy3 = sinon.spy(softmax, "hypothesis");

        softmax.predict(X);
        spy1.restore();
        spy2.restore();
        spy3.restore();

        assert.deepStrictEqual(spy1.calledOnce && spy2.calledOnce && spy2.calledOnce, true);
      });
    });


  });
});