import { inverse, Matrix } from 'ml-matrix';

import gradientFunction from './gradientFunction';

/**
 * Matrix function over the samples.
 * @ignore
 * @param {{x:Array<ArrayLike<number>>, y:Array<ArrayLike<number>>}} data - Array of points to fit in the format [x1, x2, ... ], [y1, y2, ... ].
 * @param {ArrayLike<number>} evaluatedData - Array of previous evaluated function values.
 * @return {Matrix}
 */
function matrixFunction(data, evaluatedData) {
  const m = data.x[0].length; // they are m points
  const n = data.x.length; // each point has n coordinates
  let ans = new Matrix(m, n);

  for (let point = 0; point < m; point++) {
    for (let coordinate = 0; coordinate < n; coordinate++) {
      ans.set(
        point,
        0,
        data.y[coordinate][point] - evaluatedData[coordinate][point],
      );
    }
  }
  return ans;
}

/**
 * Iteration for Levenberg-Marquardt
 * @ignore
 * @param {{x:Array<ArrayLike<number>>, y:Array<ArrayLike<number>>}} data - Array of points to fit in the format [x1, x2, ... ], [y1, y2, ... ]
 * @param {Array<number>} params - Array of previous parameter values
 * @param {number} damping - Levenberg-Marquardt parameter
 * @param {number|array} gradientDifference - The step size to approximate the jacobian matrix
 * @param {boolean} centralDifference - If true the jacobian matrix is approximated by central differences otherwise by forward differences
 * @param {function} parameterizedFunction - Returns a function with the independent variable as a parameter
 */
export default function step(
  data,
  params,
  damping,
  gradientDifference,
  parameterizedFunction,
  centralDifference,
  weights,
) {
  let value = damping;
  let identity = Matrix.eye(params.length, params.length, value);

  const func = parameterizedFunction(params);

  let evaluatedData = new Float64Array(data.x.length);
  for (let i = 0; i < data.x.length; i++) {
    evaluatedData[i] = func(data.x[i]);
  }

  let gradientFunc = gradientFunction(
    data,
    evaluatedData,
    params,
    gradientDifference,
    parameterizedFunction,
    centralDifference,
  );
  let residualError = matrixFunction(data, evaluatedData);

  let inverseMatrix = inverse(
    identity.add(
      gradientFunc.mmul(
        gradientFunc.transpose().scale('row', { scale: weights }),
      ),
    ),
  );

  let jacobianWeightResidualError = gradientFunc.mmul(
    residualError.scale('row', { scale: weights }),
  );

  let perturbations = inverseMatrix.mmul(jacobianWeightResidualError);

  return {
    perturbations,
    jacobianWeightResidualError,
  };
}
