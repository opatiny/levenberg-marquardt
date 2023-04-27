import checkOptions from './checkOptions';
import errorCalculation from './errorCalculation';
import step from './step';

/**
 * Curve fitting algorithm for non-linear data using least squares.
 * @param {{x:ArrayLike<ArrayLike<number>>, y:ArrayLike<ArrayLike<number>>}} data - Data for which to find the best fit. Data is in the format: [[x1,y1,z1], [x2,y2,z2],...].
 * @param {function} parameterizedFunction - Takes an array of parameters and returns a function with the independent variable as its sole argument.
 * @param {object} options - Options object.
 * @param {ArrayLike<number>} options.initialValues - Array of initial parameter values.
 * @param {number|ArrayLike<number>} [options.weights = 1] - Weighting vector, if the length does not match with the number of data points, the vector is reconstructed with first value.
 * @param {number} [options.damping = 1e-2] - Levenberg-Marquardt parameter, small values of the damping parameter λ result in a Gauss-Newton update and large
values of λ result in a gradient descent update.
 * @param {number} [options.dampingStepDown = 9] - Factor to reduce the damping (Levenberg-Marquardt parameter) when there is not an improvement when updating parameters.
 * @param {number} [options.dampingStepUp = 11] - Factor to increase the damping (Levenberg-Marquardt parameter) when there is an improvement when updating parameters.
 * @param {number} [options.improvementThreshold = 1e-3] - The threshold to define an improvement through an update of parameters.
 * @param {number|ArrayLike<number>} [options.gradientDifference = 10e-2] - The step size to approximate the jacobian matrix.
 * @param {boolean} [options.centralDifference = false] - If true the jacobian matrix is approximated by central differences otherwise by forward differences.
 * @param {ArrayLike<number>} [options.minValues] - Minimum allowed values for parameters.
 * @param {ArrayLike<number>} [options.maxValues] - Maximum allowed values for parameters.
 * @param {number} [options.maxIterations = 100] - Maximum of allowed iterations.
 * @param {number} [options.errorTolerance = 10e-3] - Minimum uncertainty allowed for each point.
 * @param {number} [options.timeout] - Maximum time running in seconds before throwing.
 * @return {{parameterValues: Array<number>, parameterError: number, iterations: number}}
 */
export function levenbergMarquardt(data, parameterizedFunction, options) {
  let {
    checkTimeout,
    minValues,
    maxValues,
    parameters,
    weightSquare,
    damping,
    dampingStepUp,
    dampingStepDown,
    maxIterations,
    errorTolerance,
    centralDifference,
    gradientDifference,
    improvementThreshold,
  } = checkOptions(data, parameterizedFunction, options);

  let error = errorCalculation(
    data,
    parameters,
    parameterizedFunction,
    weightSquare,
  );
  let optimalError = error;
  let optimalParameters = parameters.slice();

  let converged = error <= errorTolerance;

  let iteration = 0;
  for (; iteration < maxIterations && !converged; iteration++) {
    let previousError = error;

    let { perturbations, jacobianWeightResidualError } = step(
      data,
      parameters,
      damping,
      gradientDifference,
      parameterizedFunction,
      centralDifference,
      weightSquare,
    );

    for (let k = 0; k < parameters.length; k++) {
      parameters[k] = Math.min(
        Math.max(minValues[k], parameters[k] - perturbations.get(k, 0)),
        maxValues[k],
      );
    }

    error = errorCalculation(
      data,
      parameters,
      parameterizedFunction,
      weightSquare,
    );

    if (isNaN(error)) break;

    if (error < optimalError - errorTolerance) {
      optimalError = error;
      optimalParameters = parameters.slice();
    }

    let improvementMetric =
      (previousError - error) /
      perturbations
        .transpose()
        .mmul(perturbations.mul(damping).add(jacobianWeightResidualError))
        .get(0, 0);

    if (improvementMetric > improvementThreshold) {
      damping = Math.max(damping / dampingStepDown, 1e-7);
    } else {
      damping = Math.min(damping * dampingStepUp, 1e7);
    }

    if (checkTimeout()) {
      throw new Error(
        `The execution time is over to ${options.timeout} seconds`,
      );
    }

    converged = error <= errorTolerance;
  }

  return {
    parameterValues: optimalParameters,
    parameterError: optimalError,
    iterations: iteration,
  };
}
